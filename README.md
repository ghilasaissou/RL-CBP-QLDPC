# RL-CBP

Reference implementation of the RL-guided corrective belief-propagation (RL-CBP) decoder.

This repository includes two examples:

- `rl_cbp_b1.py`: B1 `[[882,24]]` under independent Pauli-X noise.
- `rl_cbp_a5.py`: A5 `[[180,10]]` under symmetric depolarizing noise.

Only the proposed RL-CBP approach is included here. Baseline implementations are intentionally left out.

## Data files

Put the code matrices in the same directory as the scripts:

```text
B1_HxHzLxLz.npz
A5_HxHzLxLz.npz
```

Each file is expected to contain:

```text
Hx, Hz, Lx, Lz
```

## Install

```bash
pip install -r requirements.txt
```

## B1

Train RL-CBP with `T_bp = 10`:

```bash
python rl_cbp_b1.py --mode train --T 10
```

Evaluate the trained model:

```bash
python rl_cbp_b1.py --mode eval --T 10
```

Train and evaluate:

```bash
python rl_cbp_b1.py --mode all --T 10
```

The B1 script also supports other corrective BP burst lengths, for example:

```bash
python rl_cbp_b1.py --mode all --T 1
python rl_cbp_b1.py --mode all --T 100
```

## A5

Train RL-CBP with `T_bp = 10`:

```bash
python rl_cbp_a5.py --mode train --T 10
```

Evaluate the trained model:

```bash
python rl_cbp_a5.py --mode eval --T 10
```

Train and evaluate:

```bash
python rl_cbp_a5.py --mode all --T 10
```

## Main settings

The scripts use the settings from the experiments:

- initial BP/Q-BP phase: 100 iterations
- maximum corrective actions: `R_max = n`
- candidate cap: 128 VNs
- DDQN hidden layers: 64, 64
- learning rate: `1e-4`
- discount factor: `1`
- replay capacity: `100000`
- batch size: `64`
- training episodes: `5000`

For B1, the temporary binary prior perturbation uses an LLR magnitude of 25.

For A5, the selected Pauli symbol receives probability `1-epsilon`, with `epsilon = 1e-10` for each remaining symbol.
