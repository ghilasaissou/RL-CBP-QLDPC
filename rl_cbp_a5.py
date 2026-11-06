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
TRAIN_P_VALUES = (0.07, 0.08, 0.09, 0.10)

INITIAL_QBP_ITERATIONS = 100
PERTURB_EPS = 1e-10
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

I_PAULI, X_PAULI, Y_PAULI, Z_PAULI = range(4)
PAULI_X = np.array([0, 1, 1, 0], dtype=np.uint8)
PAULI_Z = np.array([0, 0, 1, 1], dtype=np.uint8)

Hx = Hz = Lx = Lz = None
Hx_csr = Hz_csr = None
HX_BASIS = HZ_BASIS = None
HX_PIVOTS = HZ_PIVOTS = None
mx = mz = n = None
EDGE_CHECK = EDGE_BIT = EDGE_LABEL = EDGE_ANTI = None
M_CHECKS = N_EDGES = None
VN_DEGREE = None
MAX_ACTIONS = None

PROB_EPS = 1e-15
ETA_ZERO_EPS = 1e-15


def gf2_rref(M):
    A = np.asarray(M, dtype=np.uint8).copy() & 1
    rows, cols = A.shape
    pivots = []
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

        pivots.append(c)
        r += 1
        if r == rows:
            break

    return A, pivots


def gf2_rank(M):
    return len(gf2_rref(M)[1])


def gf2_rowspace_basis(M):
    R, pivots = gf2_rref(M)
    rank = len(pivots)
    return R[:rank].copy(), np.asarray(pivots, dtype=np.int64)


def gf2_in_rowspace(v, basis, pivots):
    w = np.asarray(v, dtype=np.uint8).copy() & 1
    for row, pivot_col in zip(basis, pivots):
        if w[pivot_col]:
            w ^= row
    return not np.any(w)


def load_code(path):
    global Hx, Hz, Lx, Lz, Hx_csr, Hz_csr, HX_BASIS, HZ_BASIS
    global HX_PIVOTS, HZ_PIVOTS, mx, mz, n
    global EDGE_CHECK, EDGE_BIT, EDGE_LABEL, EDGE_ANTI
    global M_CHECKS, N_EDGES, VN_DEGREE, MAX_ACTIONS

    z = np.load(path)
    Hx = np.asarray(z["Hx"], dtype=np.uint8)
    Hz = np.asarray(z["Hz"], dtype=np.uint8)
    Lx = np.asarray(z["Lx"], dtype=np.uint8)
    Lz = np.asarray(z["Lz"], dtype=np.uint8)

    mx, n = Hx.shape
    mz, n2 = Hz.shape
    if n != n2:
        raise ValueError("Hx and Hz must have the same number of columns.")

    rank_hx = gf2_rank(Hx)
    rank_hz = gf2_rank(Hz)
    k = n - rank_hx - rank_hz

    if Hx.shape != (90, 180) or Hz.shape != (90, 180) or k != 10:
        raise ValueError(f"Expected A5 [[180,10]], got Hx={Hx.shape}, Hz={Hz.shape}, k={k}.")
    if np.any((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2):
        raise ValueError("Hx Hz^T != 0.")

    Hx_csr = csr_matrix(Hx)
    Hz_csr = csr_matrix(Hz)
    HX_BASIS, HX_PIVOTS = gf2_rowspace_basis(Hx)
    HZ_BASIS, HZ_PIVOTS = gf2_rowspace_basis(Hz)

    anti_table = np.zeros((4, 4), dtype=np.uint8)
    for g in range(4):
        for q in range(4):
            anti_table[g, q] = (
                int(PAULI_X[g]) * int(PAULI_Z[q])
                + int(PAULI_Z[g]) * int(PAULI_X[q])
            ) % 2

    hz_coo = coo_matrix(Hz)
    hx_coo = coo_matrix(Hx)

    EDGE_CHECK = np.concatenate([
        hz_coo.row.astype(np.int64),
        (mz + hx_coo.row).astype(np.int64),
    ])
    EDGE_BIT = np.concatenate([
        hz_coo.col.astype(np.int64),
        hx_coo.col.astype(np.int64),
    ])
    EDGE_LABEL = np.concatenate([
        np.full(hz_coo.nnz, Z_PAULI, dtype=np.uint8),
        np.full(hx_coo.nnz, X_PAULI, dtype=np.uint8),
    ])

    M_CHECKS = mz + mx
    N_EDGES = len(EDGE_CHECK)
    EDGE_ANTI = anti_table[EDGE_LABEL].astype(np.uint8)
    VN_DEGREE = np.bincount(EDGE_BIT, minlength=n).astype(np.float32)
    MAX_ACTIONS = n


def depolarizing_prior(p):
    p = float(p)
    one = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0], dtype=np.float64)
    return np.tile(one, (n, 1))


def sample_depolarizing_error(p, rng):
    q = np.zeros(n, dtype=np.int8)
    hit = rng.random(n) < float(p)
    if np.any(hit):
        q[hit] = rng.integers(1, 4, size=int(hit.sum()), dtype=np.int8)
    return q


def pauli_to_xz(q):
    q = np.asarray(q, dtype=np.int64).reshape(-1)
    if q.size != n:
        raise ValueError(f"Expected {n} Pauli symbols.")
    return PAULI_X[q].astype(np.uint8), PAULI_Z[q].astype(np.uint8)


def syndrome_of_pauli(q):
    ex, ez = pauli_to_xz(q)
    sz = np.asarray((Hz_csr @ ex) % 2, dtype=np.uint8).reshape(-1)
    sx = np.asarray((Hx_csr @ ez) % 2, dtype=np.uint8).reshape(-1)
    return np.concatenate([sz, sx])


def classify_pauli(true_error, correction):
    tx, tz = pauli_to_xz(true_error)
    cx, cz = pauli_to_xz(correction)
    rx = tx ^ cx
    rz = tz ^ cz

    if np.any((Hz_csr @ rx) % 2) or np.any((Hx_csr @ rz) % 2):
        return "nonconv"
    if not np.any(rx) and not np.any(rz):
        return "same"

    x_stab = gf2_in_rowspace(rx, HX_BASIS, HX_PIVOTS)
    z_stab = gf2_in_rowspace(rz, HZ_BASIS, HZ_PIVOTS)
    return "degen" if x_stab and z_stab else "logic"


def normalize_rows(x):
    x = np.clip(np.asarray(x, dtype=np.float64), PROB_EPS, None)
    return x / np.sum(x, axis=1, keepdims=True)


def softmax_rows(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def q_check_to_var(v2c, syndrome):
    v2c = normalize_rows(v2c)
    syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)

    anti_probability = np.sum(v2c * EDGE_ANTI, axis=1)
    eta = 1.0 - 2.0 * anti_probability

    is_zero = np.abs(eta) < ETA_ZERO_EPS
    safe_eta = eta.copy()
    safe_eta[is_zero] = 1.0

    product_by_check = np.ones(M_CHECKS, dtype=np.float64)
    np.multiply.at(product_by_check, EDGE_CHECK, safe_eta)
    zero_count = np.bincount(
        EDGE_CHECK,
        weights=is_zero.astype(np.int64),
        minlength=M_CHECKS,
    ).astype(np.int64)

    edge_zero_count = zero_count[EDGE_CHECK]
    edge_product = product_by_check[EDGE_CHECK]
    extrinsic = np.zeros(N_EDGES, dtype=np.float64)

    no_zero = edge_zero_count == 0
    extrinsic[no_zero] = edge_product[no_zero] / eta[no_zero]

    one_zero_here = (edge_zero_count == 1) & is_zero
    extrinsic[one_zero_here] = edge_product[one_zero_here]

    syndrome_sign = (1.0 - 2.0 * syndrome.astype(np.float64))[EDGE_CHECK, None]
    target_sign = 1.0 - 2.0 * EDGE_ANTI.astype(np.float64)

    c2v = 0.5 * (1.0 + syndrome_sign * target_sign * extrinsic[:, None])
    return np.clip(c2v, PROB_EPS, 1.0)


def q_posterior_and_v2c(c2v, priors):
    priors = normalize_rows(priors)
    log_c2v = np.log(np.clip(c2v, PROB_EPS, 1.0))

    incoming_log = np.zeros((n, 4), dtype=np.float64)
    np.add.at(incoming_log, EDGE_BIT, log_c2v)

    log_prior = np.log(np.clip(priors, PROB_EPS, 1.0))
    posterior = softmax_rows(log_prior + incoming_log)

    edge_logits = (
        log_prior[EDGE_BIT]
        + incoming_log[EDGE_BIT]
        - log_c2v
    )
    new_v2c = softmax_rows(edge_logits)
    return posterior, new_v2c


def q_posterior_from_v2c(v2c, syndrome, priors):
    c2v = q_check_to_var(v2c, syndrome)
    posterior, _ = q_posterior_and_v2c(c2v, priors)
    return posterior


def qbp_run(syndrome, priors, max_iterations=100, init_v2c=None):
    syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
    priors = normalize_rows(priors)

    if init_v2c is None:
        v2c = priors[EDGE_BIT].copy()
    else:
        v2c = normalize_rows(np.asarray(init_v2c, dtype=np.float64))

    posterior = priors.copy()
    qhat = np.argmax(posterior, axis=1).astype(np.int8)

    if np.array_equal(syndrome_of_pauli(qhat), syndrome):
        return {
            "converged": True,
            "iterations": 0,
            "v2c": v2c,
            "posterior": posterior,
            "qhat": qhat,
        }

    for iteration in range(1, int(max_iterations) + 1):
        c2v = q_check_to_var(v2c, syndrome)
        posterior, v2c = q_posterior_and_v2c(c2v, priors)
        qhat = np.argmax(posterior, axis=1).astype(np.int8)

        if np.array_equal(syndrome_of_pauli(qhat), syndrome):
            return {
                "converged": True,
                "iterations": iteration,
                "v2c": v2c,
                "posterior": posterior,
                "qhat": qhat,
            }

    return {
        "converged": False,
        "iterations": int(max_iterations),
        "v2c": v2c,
        "posterior": posterior,
        "qhat": qhat,
    }


def target_pauli_prior(target_q, eps=PERTURB_EPS):
    v = np.full(4, float(eps), dtype=np.float64)
    v[int(target_q)] = 1.0 - float(eps)
    return v / v.sum()


def q_unsat_stats(posterior, syndrome):
    qhat = np.argmax(posterior, axis=1).astype(np.int8)
    unsat = syndrome_of_pauli(qhat) ^ np.asarray(syndrome, dtype=np.uint8)
    cnt = np.bincount(
        EDGE_BIT,
        weights=unsat[EDGE_CHECK],
        minlength=n,
    ).astype(np.float32)
    frac = cnt / np.maximum(VN_DEGREE, 1.0)
    return qhat, unsat, cnt, frac


def build_state(posterior, syndrome, intervention_counts, step, candidate_k=CANDIDATE_K):
    posterior = normalize_rows(posterior)
    qhat, unsat, cnt, frac = q_unsat_stats(posterior, syndrome)

    reliability = np.max(posterior, axis=1).astype(np.float32)
    max_degree = max(1.0, float(VN_DEGREE.max()))

    node = np.concatenate([
        posterior.astype(np.float32),
        reliability[:, None],
        frac[:, None].astype(np.float32),
        np.clip(intervention_counts.astype(np.float32) / 4.0, 0.0, 1.0)[:, None],
        (VN_DEGREE / max_degree)[:, None].astype(np.float32),
    ], axis=1).astype(np.float16)

    glob = np.array([
        float(np.mean(unsat)),
        float(np.mean(qhat != I_PAULI)),
        float(np.mean(reliability)),
        float(step) / float(max(1, MAX_ACTIONS)),
    ], dtype=np.float16)

    active = np.flatnonzero(cnt > 0)
    if len(active) == 0:
        active = np.arange(n)

    mask = np.zeros(n, dtype=bool)
    if candidate_k is None or len(active) <= int(candidate_k):
        mask[active] = True
    else:
        suspicion = frac[active] + 0.25 * (1.0 - reliability[active])
        keep = active[np.argpartition(suspicion, -int(candidate_k))[-int(candidate_k):]]
        mask[keep] = True

    return {
        "node": node,
        "global": glob,
        "mask": mask,
        "unsat_weight": int(unsat.sum()),
    }


class QuaternaryCorrectiveQNet(nn.Module):
    def __init__(self, node_dim=8, global_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
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
        q = q.reshape(batch_size, 4 * num_nodes)
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
    return (4 * candidates[:, None] + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)


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
    true_error = np.asarray(case["true_error"], dtype=np.int8)
    syndrome = np.asarray(case["syndrome"], dtype=np.uint8)
    base_prior = depolarizing_prior(p)
    v2c = np.asarray(case["v2c"], dtype=np.float64).copy()

    posterior = q_posterior_from_v2c(v2c, syndrome, base_prior)
    intervention_counts = np.zeros(n, dtype=np.int16)
    transitions = []

    state = build_state(posterior, syndrome, intervention_counts, 0)
    old_unsat = int(state["unsat_weight"])

    for t in range(MAX_ACTIONS):
        action = select_action(net, state, epsilon, rng)
        i, target_q = divmod(action, 4)
        intervention_counts[i] += 1

        perturbed_prior = base_prior.copy()
        perturbed_prior[i] = target_pauli_prior(target_q, eps=PERTURB_EPS)

        burst = qbp_run(
            syndrome,
            perturbed_prior,
            max_iterations=int(T),
            init_v2c=v2c,
        )

        # Same idea as B1: restore the prior, keep the evolved messages.
        v2c = burst["v2c"].copy()
        posterior_next = q_posterior_from_v2c(v2c, syndrome, base_prior)
        next_state = build_state(
            posterior_next,
            syndrome,
            intervention_counts,
            t + 1,
        )

        if burst["converged"]:
            outcome = classify_pauli(true_error, burst["qhat"])
            reward = (1.0 if outcome in ("same", "degen") else -1.0) - STEP_PENALTY
            if collect:
                transitions.append(Transition(state, action, reward, next_state, True))
            return {
                "outcome": outcome,
                "actions": t + 1,
                "transitions": transitions,
            }

        new_unsat = int(next_state["unsat_weight"])
        reward = -STEP_PENALTY + SHAPING_SCALE * (
            old_unsat - new_unsat
        ) / float(M_CHECKS)

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
        posterior = posterior_next
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
            true_error = sample_depolarizing_error(p, rng)
            syndrome = syndrome_of_pauli(true_error)
            bp = qbp_run(
                syndrome,
                depolarizing_prior(p),
                max_iterations=INITIAL_QBP_ITERATIONS,
            )

            if not bp["converged"]:
                cases.append({
                    "p": float(p),
                    "true_error": true_error.astype(np.int8),
                    "syndrome": syndrome.astype(np.uint8),
                    "v2c": bp["v2c"].astype(np.float32),
                })

        train_cases.extend(cases[:TRAIN_FAIL_PER_P])
        val_cases[float(p)] = cases[TRAIN_FAIL_PER_P:]

    return train_cases, val_cases


def eval_cases(cases, net, T):
    outcomes = {"same": 0, "degen": 0, "logic": 0, "nonconv": 0}
    actions = []
    rng = np.random.default_rng(777)

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

    online = QuaternaryCorrectiveQNet().to(DEVICE)
    target = QuaternaryCorrectiveQNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    replay = Replay()
    rng = np.random.default_rng(SEED)

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
    net = QuaternaryCorrectiveQNet().to(DEVICE)
    ckpt = torch.load(model_file, map_location=DEVICE)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    return net


def decode_rl_cbp(true_error, p, net, T):
    syndrome = syndrome_of_pauli(true_error)
    initial = qbp_run(
        syndrome,
        depolarizing_prior(p),
        max_iterations=INITIAL_QBP_ITERATIONS,
    )

    if initial["converged"]:
        return classify_pauli(true_error, initial["qhat"])

    case = {
        "p": float(p),
        "true_error": np.asarray(true_error, dtype=np.int8),
        "syndrome": syndrome.astype(np.uint8),
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
        rng = np.random.default_rng(110000 + j)
        counts = {"same": 0, "degen": 0, "logic": 0, "nonconv": 0}
        runs = 0

        while counts["logic"] + counts["nonconv"] < int(target_errors):
            if max_runs is not None and runs >= int(max_runs):
                break

            true_error = sample_depolarizing_error(p, rng)
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
    parser = argparse.ArgumentParser(description="RL-CBP for A5 under depolarizing noise.")
    parser.add_argument("--data-file", default="A5_HxHzLxLz.npz")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--target-errors", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    load_code(args.data_file)

    model_file = args.model_file or f"a5_rl_cbp_T{args.T}.pt"
    output = args.output or f"a5_rl_cbp_T{args.T}_bler.csv"

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
