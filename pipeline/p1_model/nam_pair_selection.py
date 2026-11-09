"""Pair selection for the neural additive model with pairwise terms (GA²M style; 2026-10-18).

As in GA²M (Lou et al. 2013): start from an additive model and add, greedily, the pair of inputs that most reduces the
held-out residual. Target: B0's boundary speed on the logit scale (cells.npz sample from nn_vs_linear.py, every
trained model separately: 5 folds × 2 seeds; fitted on even 2-h steps, scored on odd ones, gains averaged over models).
Additive base: linear in all 37 inputs + cubic-spline curves of the 8 key inputs (6 current AIA features, PFSS f_s,
back-mapped recurrence). Pair basis: x_i·x_j, x_i²·x_j, x_i·x_j² (a low-order surrogate of a smooth 2-input surface).
Candidates: the 28 pairs of the 8 key inputs.

  python pipeline/p1_model/nam_pair_selection.py
Output: runs/sweep4/nn_vs_linear/pair_selection.csv (greedy order, held-out RMS in km/s after each addition)
"""
import numpy as np
import pandas as pd

KEY = [0, 1, 2, 3, 4, 5, 34, 35]
NAMES = {0: 'area f0.30 w20', 1: 'ecl. band 15', 2: 'area f0.35 w12.5', 3: 'mean log I dark', 4: 'hole-bdry dist +10',
         5: 'ESWF area', 34: 'log10 fs', 35: 'recurrence'}
LO, SPAN = 150.0, 650.0
C = dict(np.load('runs/sweep4/nn_vs_linear/cells.npz'))


def logit(v):
    p = np.clip((v - LO) / SPAN, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def spline(u, k=6):
    kn = np.quantile(u, np.linspace(0.05, 0.95, k))
    return np.stack([u ** 2, u ** 3] + [np.maximum(u - t, 0) ** 3 for t in kn], 1)


def pair_basis(x, i, j):
    return np.stack([x[:, i] * x[:, j], x[:, i] ** 2 * x[:, j], x[:, i] * x[:, j] ** 2], 1)


models = []
for f, s in sorted(set(zip(C['fold'], C['seed']))):
    m = (C['fold'] == f) & (C['seed'] == s)
    x = C['x'][m]
    base = np.hstack([x, np.ones((len(x), 1))] + [spline(x[:, i]) for i in KEY])
    models.append(dict(x=x, base=base, z=logit(C['vb'][m]), v=C['vb'][m], tr=C['even'][m]))


def score(pairs):
    out = []
    for M in models:
        X = np.hstack([M['base']] + [pair_basis(M['x'], i, j) for i, j in pairs])
        mu, sd = X[M['tr']].mean(0), X[M['tr']].std(0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        Xs[:, 37] = 1.0
        tr, te = M['tr'], ~M['tr']
        w = np.linalg.solve(Xs[tr].T @ Xs[tr] + 1e-3 * tr.sum() * np.eye(Xs.shape[1]), Xs[tr].T @ M['z'][tr])
        vh = LO + SPAN / (1 + np.exp(-(Xs[te] @ w)))
        out.append(np.sqrt(np.mean((M['v'][te] - vh) ** 2)))
    return float(np.mean(out))


cands = [(a, b) for k, a in enumerate(KEY) for b in KEY[k + 1:]]
chosen, rows = [], [dict(step=0, pair='(additive only)', rms=score([]))]
print(rows[-1], flush=True)
for step in range(1, 6):
    best = min((p for p in cands if p not in chosen), key=lambda p: score(chosen + [p]))
    chosen.append(best)
    rows.append(dict(step=step, pair=f'{NAMES[best[0]]} × {NAMES[best[1]]}', i=best[0], j=best[1], rms=score(chosen)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('runs/sweep4/nn_vs_linear/pair_selection.csv', index=False)
