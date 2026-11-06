#!/usr/bin/env python3
import argparse
import random
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAPER_P_VALUES = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
TRAIN_P_VALUES = (0.04, 0.05, 0.06, 0.07, 0.08)

INITIAL_BP_ITERATIONS = 100
BP_MESSAGE_CAP = 25.0
PERTURB_LLR = 25.0
CANDIDATE_K = 128
STEP_PENALTY = 0.002
SHAPING_SCALE = 0.25

TRAIN_FAIL_PER_P = 250
VAL_FAIL_PER_P = 100
TRAIN_EPISODES = 5000
REPLAY_CAPACITY = 100000
REPLAY_MIN = 1024
BATCH_SIZE = 64
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 500
EPS_DECAY_STEPS = 100000
LEARNING_RATE = 1e-4
DISCOUNT = 1.0

Hx = Hz = Lx = Lz = None
Hz_csr = None
m = n = None
EDGE_CHECK = EDGE_BIT = None
N_EDGES = None
VN_DEGREE = None
MAX_ACTIONS = None


def gf2_rank(M):
    A = np.asarray(M, dtype=np.uint8).copy() & 1
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        idx = np.flatnonzero(A[r:, c])
        if len(idx) == 0:
            continue
        pivot = r + int(idx[0])
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        other = np.flatnonzero(A[:, c])
        other = other[other != r]
        if len(other):
            A[other] ^= A[r]
        r += 1
        if r == rows:
            break
    return int(r)


def load_code(path):
    global Hx, Hz, Lx, Lz, Hz_csr, m, n
    global EDGE_CHECK, EDGE_BIT, N_EDGES, VN_DEGREE, MAX_ACTIONS

    z = np.load(path)
    Hx = np.asarray(z["Hx"], dtype=np.uint8)
    Hz = np.asarray(z["Hz"], dtype=np.uint8)
    Lx = np.asarray(z["Lx"], dtype=np.uint8)
    Lz = np.asarray(z["Lz"], dtype=np.uint8)

    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns.")

    k = Hx.shape[1] - gf2_rank(Hx) - gf2_rank(Hz)
    if Hx.shape[1] != 882 or k != 24:
        raise ValueError(f"Expected B1 [[882,24]], got n={Hx.shape[1]}, k={k}.")
    if np.any((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2):
        raise ValueError("Hx Hz^T != 0.")

    Hz_csr = csr_matrix(Hz)
    Hz_coo = coo_matrix(Hz)
    m, n = Hz.shape
    EDGE_CHECK = Hz_coo.row.astype(np.int64)
    EDGE_BIT = Hz_coo.col.astype(np.int64)
    N_EDGES = len(EDGE_CHECK)
    VN_DEGREE = np.bincount(EDGE_BIT, minlength=n).astype(np.float32)
    MAX_ACTIONS = n


def channel_llr(p):
    return float(np.log((1.0 - float(p)) / float(p)))


def channel_vector(p):
    return np.full(n, channel_llr(p), dtype=np.float64)


def sample_x_error(p, rng):
    return (rng.random(n) < float(p)).astype(np.uint8)


def syndrome_of(error):
    return np.asarray((Hz_csr @ np.asarray(error, dtype=np.uint8)) % 2, dtype=np.uint8).reshape(-1)


def hard_syndrome(bits):
    return np.asarray((Hz_csr @ np.asarray(bits, dtype=np.uint8)) % 2, dtype=np.uint8).reshape(-1)


def classify(true_error, correction):
    residual = np.asarray(true_error, dtype=np.uint8) ^ np.asarray(correction, dtype=np.uint8)
    if np.any(hard_syndrome(residual)):
        return "nonconv"
    if not np.any(residual):
        return "same"
    logical = np.asarray(
        (Lz.astype(np.int64) @ residual.astype(np.int64).T) % 2,
        dtype=np.uint8,
    ).reshape(-1)
    return "degen" if not np.any(logical) else "logic"


def check_to_variable(v2c, syndrome):
    means = np.tanh(np.asarray(v2c, dtype=np.float64) / 2.0)
    zero = np.abs(means) < 1e-15
    safe = means.copy()
    safe[zero] = 1.0

    prod = np.ones(m, dtype=np.float64)
    np.multiply.at(prod, EDGE_CHECK, safe)
    zero_count = np.bincount(
        EDGE_CHECK, weights=zero.astype(np.int64), minlength=m
    ).astype(np.int64)

    zc_edge = zero_count[EDGE_CHECK]
    prod_edge = prod[EDGE_CHECK]
    extrinsic = np.zeros(N_EDGES, dtype=np.float64)

    no_zero = zc_edge == 0
    extrinsic[no_zero] = prod_edge[no_zero] / means[no_zero]

    single_zero_target = (zc_edge == 1) & zero
    extrinsic[single_zero_target] = prod_edge[single_zero_target]

    sign = 1.0 - 2.0 * np.asarray(syndrome, dtype=np.float64)
    extrinsic *= sign[EDGE_CHECK]
    extrinsic = np.clip(extrinsic, -1.0 + 1e-15, 1.0 - 1e-15)
    return 2.0 * np.arctanh(extrinsic)


def one_bp_iteration(v2c, syndrome, channel):
    c2v = check_to_variable(v2c, syndrome)
    incoming = np.bincount(EDGE_BIT, weights=c2v, minlength=n)
    posterior = np.asarray(channel, dtype=np.float64) + incoming
    new_v2c = np.clip(
        posterior[EDGE_BIT] - c2v,
        -BP_MESSAGE_CAP,
        BP_MESSAGE_CAP,
    )
    return posterior, new_v2c


def hard_from_bias(posterior):
    return (np.asarray(posterior) <= 0.0).astype(np.uint8)


def run_bp(syndrome, channel, max_iterations, init_v2c=None):
    channel = np.asarray(channel, dtype=np.float64).reshape(-1)
    syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)

    if init_v2c is None:
        v2c = channel[EDGE_BIT].copy()
    else:
        v2c = np.clip(
            np.asarray(init_v2c, dtype=np.float64).copy(),
            -BP_MESSAGE_CAP,
            BP_MESSAGE_CAP,
        )

    posterior = channel.copy()
    bits = hard_from_bias(posterior)

    # No iteration-0 convergence check, matching the experiment.
    for it in range(1, int(max_iterations) + 1):
        posterior, v2c = one_bp_iteration(v2c, syndrome, channel)
        bits = hard_from_bias(posterior)
        if np.array_equal(hard_syndrome(bits), syndrome):
            return {
                "converged": True,
                "iterations": it,
                "posterior": posterior,
                "v2c": v2c,
                "bits": bits,
            }

    return {
        "converged": False,
        "iterations": int(max_iterations),
        "posterior": posterior,
        "v2c": v2c,
        "bits": bits,
    }


def posterior_from_v2c(v2c, syndrome, channel):
    c2v = check_to_variable(v2c, syndrome)
    incoming = np.bincount(EDGE_BIT, weights=c2v, minlength=n)
    return np.asarray(channel, dtype=np.float64) + incoming


def build_state(post, syndrome, intervention_count, step, candidate_k=CANDIDATE_K):
    post = np.asarray(post, dtype=np.float64)
    bits = hard_from_bias(post)
    unsat = hard_syndrome(bits) ^ np.asarray(syndrome, dtype=np.uint8)

    unsat_count = np.bincount(
        EDGE_BIT,
        weights=unsat[EDGE_CHECK],
        minlength=n,
    ).astype(np.float32)
    unsat_frac = unsat_count / np.maximum(VN_DEGREE, 1.0)
    max_degree = max(float(VN_DEGREE.max()), 1.0)

    node = np.stack([
        post / BP_MESSAGE_CAP,
        np.abs(post) / BP_MESSAGE_CAP,
        unsat_frac,
        bits.astype(np.float32),
        np.clip(np.asarray(intervention_count, dtype=np.float32) / 4.0, 0.0, 1.0),
        VN_DEGREE / max_degree,
    ], axis=1).astype(np.float16)

    glob = np.array([
        float(np.mean(unsat)),
        float(np.mean(bits)),
        float(np.mean(np.abs(post)) / BP_MESSAGE_CAP),
        float(step) / float(max(1, MAX_ACTIONS)),
    ], dtype=np.float16)

    active = np.flatnonzero(unsat_count > 0)
    if len(active) == 0:
        active = np.arange(n)

    mask = np.zeros(n, dtype=bool)
    if candidate_k is None or len(active) <= int(candidate_k):
        mask[active] = True
    else:
        suspicion = unsat_frac[active] + 0.25 * (
            1.0 - np.clip(np.abs(post[active]) / BP_MESSAGE_CAP, 0.0, 1.0)
        )
        keep = active[np.argpartition(suspicion, -int(candidate_k))[-int(candidate_k):]]
        mask[keep] = True

    return {
        "node": node,
        "global": glob,
        "mask": mask,
        "unsat_weight": int(unsat.sum()),
    }


class BinaryCorrectiveQNet(nn.Module):
    def __init__(self, node_dim=6, global_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, node, glob, mask):
        single = node.dim() == 2
        if single:
            node = node.unsqueeze(0)
            glob = glob.unsqueeze(0)
            mask = mask.unsqueeze(0)

        batch_size, num_nodes, _ = node.shape
        g = glob.unsqueeze(1).expand(batch_size, num_nodes, -1)
        q = self.net(torch.cat([node, g], dim=-1))
        q = q.masked_fill(~mask.unsqueeze(-1), -1e9)
        q = q.reshape(batch_size, 2 * num_nodes)
        return q.squeeze(0) if single else q


def state_tensors(state, device):
    return (
        torch.from_numpy(state["node"]).float().to(device),
        torch.from_numpy(state["global"]).float().to(device),
        torch.from_numpy(state["mask"]).bool().to(device),
    )


def stack_states(states, device):
    return (
        torch.from_numpy(np.stack([s["node"] for s in states])).float().to(device),
        torch.from_numpy(np.stack([s["global"] for s in states])).float().to(device),
        torch.from_numpy(np.stack([s["mask"] for s in states])).bool().to(device),
    )


Transition = namedtuple("Transition", "state action reward next_state done")


class Replay:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buf = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buf)

    def add(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size, rng):
        ids = rng.choice(len(self.buf), size=int(batch_size), replace=False)
        return [self.buf[int(i)] for i in ids]


def legal_actions(state):
    candidates = np.flatnonzero(state["mask"])
    return (2 * candidates[:, None] + np.arange(2, dtype=np.int64)[None, :]).reshape(-1)


def select_action(net, state, epsilon, rng):
    legal = legal_actions(state)
    if rng.random() < float(epsilon):
        return int(rng.choice(legal))

    device = next(net.parameters()).device
    node, glob, mask = state_tensors(state, device)
    net.eval()
    with torch.no_grad():
        q = net(node, glob, mask)
    return int(torch.argmax(q).item())


def epsilon_at(step):
    fraction = min(1.0, float(step) / float(EPS_DECAY_STEPS))
    return 1.0 + fraction * (0.05 - 1.0)


def optimize_ddqn(online, target, optimizer, replay, rng):
    batch = replay.sample(BATCH_SIZE, rng)
    states = [x.state for x in batch]
    next_states = [x.next_state for x in batch]

    device = next(online.parameters()).device
    actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=device)

    node, glob, mask = stack_states(states, device)
    nnode, nglob, nmask = stack_states(next_states, device)

    online.train()
    q = online(node, glob, mask).gather(1, actions[:, None]).squeeze(1)

    with torch.no_grad():
        next_action = torch.argmax(online(nnode, nglob, nmask), dim=1)
        next_q = target(nnode, nglob, nmask).gather(1, next_action[:, None]).squeeze(1)
        y = rewards + DISCOUNT * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def corrective_episode(case, net, T, epsilon=0.0, rng=None, collect=False):
    if rng is None:
        rng = np.random.default_rng()

    p = float(case["p"])
    true_error = np.asarray(case["true_error"], dtype=np.uint8)
    syndrome = np.asarray(case["syndrome"], dtype=np.uint8)
    base_channel = channel_vector(p)
    v2c = np.asarray(case["v2c"], dtype=np.float64).copy()

    post = posterior_from_v2c(v2c, syndrome, base_channel)
    intervention_count = np.zeros(n, dtype=np.int16)
    transitions = []

    state = build_state(post, syndrome, intervention_count, 0)
    old_unsat = int(state["unsat_weight"])

    for t in range(MAX_ACTIONS):
        action = select_action(net, state, epsilon, rng)
        vn, target_bit = divmod(action, 2)
        intervention_count[vn] += 1

        perturbed = base_channel.copy()
        perturbed[vn] = -PERTURB_LLR if target_bit == 1 else +PERTURB_LLR

        burst = run_bp(
            syndrome,
            perturbed,
            int(T),
            init_v2c=v2c,
        )

        # The prior goes back to normal; only the evolved messages stay.
        v2c = burst["v2c"].copy()
        post_next = posterior_from_v2c(v2c, syndrome, base_channel)
        next_state = build_state(post_next, syndrome, intervention_count, t + 1)

        if burst["converged"]:
            outcome = classify(true_error, burst["bits"])
            reward = (1.0 if outcome in ("same", "degen") else -1.0) - STEP_PENALTY
            if collect:
                transitions.append(Transition(state, action, reward, next_state, True))
            return {
                "outcome": outcome,
                "actions": t + 1,
                "transitions": transitions,
            }

        new_unsat = int(next_state["unsat_weight"])
        reward = -STEP_PENALTY + SHAPING_SCALE * (old_unsat - new_unsat) / float(max(1, m))
        done = t == MAX_ACTIONS - 1
        if done:
            reward -= 1.0

        if collect:
            transitions.append(Transition(state, action, reward, next_state, done))

        if done:
            return {
                "outcome": "nonconv",
                "actions": t + 1,
                "transitions": transitions,
            }

        state = next_state
        post = post_next
        old_unsat = new_unsat

    raise RuntimeError("Unexpected RL episode exit.")


def collect_failure_bank():
    train_cases = []
    val_cases = {}
    for j, p in enumerate(TRAIN_P_VALUES):
        rng = np.random.default_rng(20000 + j)
        needed = TRAIN_FAIL_PER_P + VAL_FAIL_PER_P
        cases = []

        while len(cases) < needed:
            true_error = sample_x_error(p, rng)
            syndrome = syndrome_of(true_error)
            bp = run_bp(syndrome, channel_vector(p), INITIAL_BP_ITERATIONS)

            if not bp["converged"]:
                cases.append({
                    "p": float(p),
                    "true_error": true_error,
                    "syndrome": syndrome,
                    "v2c": bp["v2c"].astype(np.float32),
                })

        train_cases.extend(cases[:TRAIN_FAIL_PER_P])
        val_cases[float(p)] = cases[TRAIN_FAIL_PER_P:]

    return train_cases, val_cases


def eval_cases(cases, net, T):
    outcomes = {"same": 0, "degen": 0, "logic": 0, "nonconv": 0}
    actions = []
    rng = np.random.default_rng(777000 + int(T))

    net.eval()
    with torch.no_grad():
        for case in cases:
            result = corrective_episode(case, net, T, epsilon=0.0, rng=rng)
            outcomes[result["outcome"]] += 1
            actions.append(result["actions"])

    total = len(cases)
    return {
        "rescue_rate": (outcomes["same"] + outcomes["degen"]) / float(max(1, total)),
        "mean_actions": float(np.mean(actions)) if actions else np.nan,
    }


def train_model(T, model_file, episodes=TRAIN_EPISODES):
    train_cases, val_cases = collect_failure_bank()

    online = BinaryCorrectiveQNet().to(DEVICE)
    target = BinaryCorrectiveQNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    replay = Replay()
    rng = np.random.default_rng(SEED + 1000 * int(T))

    env_step = 0
    updates = 0
    best_rate = -1.0
    best_actions = np.inf

    for episode in range(1, int(episodes) + 1):
        case = train_cases[int(rng.integers(0, len(train_cases)))]
        result = corrective_episode(
            case,
            online,
            int(T),
            epsilon=epsilon_at(env_step),
            rng=rng,
            collect=True,
        )

        for tr in result["transitions"]:
            replay.add(*tr)
            env_step += 1

            if len(replay) >= REPLAY_MIN and env_step % TRAIN_EVERY == 0:
                optimize_ddqn(online, target, optimizer, replay, rng)
                updates += 1
                if updates % TARGET_UPDATE_EVERY == 0:
                    target.load_state_dict(online.state_dict())

        if episode % 250 == 0 or episode == int(episodes):
            rows = [eval_cases(cases, online, int(T)) for cases in val_cases.values()]
            val_rate = float(np.mean([r["rescue_rate"] for r in rows]))
            val_actions = float(np.mean([r["mean_actions"] for r in rows]))
            print(
                f"episode={episode:5d}  val_rescue={val_rate:.3f}  "
                f"val_actions={val_actions:.2f}  eps={epsilon_at(env_step):.3f}"
            )

            better = (
                val_rate > best_rate + 1e-12
                or (abs(val_rate - best_rate) <= 1e-12 and val_actions < best_actions)
            )
            if better:
                best_rate = val_rate
                best_actions = val_actions
                torch.save(
                    {
                        "online_state_dict": online.state_dict(),
                        "T": int(T),
                        "n": n,
                        "val_rescue_rate": val_rate,
                        "val_mean_actions": val_actions,
                    },
                    model_file,
                )

    return model_file


def load_model(model_file):
    net = BinaryCorrectiveQNet().to(DEVICE)
    ckpt = torch.load(model_file, map_location=DEVICE)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    return net


def decode_rl_cbp(true_error, p, net, T):
    syndrome = syndrome_of(true_error)
    initial = run_bp(syndrome, channel_vector(p), INITIAL_BP_ITERATIONS)

    if initial["converged"]:
        return classify(true_error, initial["bits"])

    case = {
        "p": float(p),
        "true_error": np.asarray(true_error, dtype=np.uint8),
        "syndrome": syndrome,
        "v2c": initial["v2c"].astype(np.float32),
    }
    result = corrective_episode(
        case,
        net,
        int(T),
        epsilon=0.0,
        rng=np.random.default_rng(123),
        collect=False,
    )
    return result["outcome"]


def evaluate_model(net, T, p_values, target_errors=100, max_runs=None):
    rows = []

    for j, p in enumerate(p_values):
        rng = np.random.default_rng(610000 + j + 1000 * int(T))
        counts = {"same": 0, "degen": 0, "logic": 0, "nonconv": 0}
        runs = 0

        while counts["logic"] + counts["nonconv"] < int(target_errors):
            if max_runs is not None and runs >= int(max_runs):
                break

            true_error = sample_x_error(p, rng)
            outcome = decode_rl_cbp(true_error, p, net, int(T))
            counts[outcome] += 1
            runs += 1

        errors = counts["logic"] + counts["nonconv"]
        rows.append({
            "decoder": f"RL-CBP T={int(T)}",
            "T": int(T),
            "p": float(p),
            "runs": runs,
            **counts,
            "block_errors": errors,
            "BLER": errors / float(runs),
        })
        print(f"p={p:.2f}  runs={runs:,}  BLER={errors / float(runs):.6e}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="RL-CBP for B1 under independent Pauli-X noise.")
    parser.add_argument("--data-file", default="B1_HxHzLxLz.npz")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--target-errors", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    load_code(args.data_file)

    model_file = args.model_file or f"b1_rl_cbp_T{args.T}.pt"
    output = args.output or f"b1_rl_cbp_T{args.T}_bler.csv"

    if args.mode in ("train", "all"):
        train_model(args.T, model_file, episodes=args.episodes)

    if args.mode in ("eval", "all"):
        net = load_model(model_file)
        df = evaluate_model(
            net,
            args.T,
            PAPER_P_VALUES,
            target_errors=args.target_errors,
            max_runs=args.max_runs,
        )
        df.to_csv(output, index=False)
        print("saved:", Path(output).resolve())


if __name__ == "__main__":
    main()
