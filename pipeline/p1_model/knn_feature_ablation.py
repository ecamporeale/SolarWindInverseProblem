"""Look-alike (kNN) degeneracy test per feature: each feature alone, all-but-one, all 8, all 8 + a pure-noise feature.

Inputs from knn_degeneracy_test.py (data/knn_degeneracy/inputs.npz: ballistic matching, CM column, CM-equivalent
set-2a features). Same settings: standardized Euclidean distance, k = 50, |dT| < 13 d excluded, neighbours from any
period, every 2nd hour as query. 'with lags' variants drop / keep the feature in all 5 frames (t, t-12..72 h).

Metrics (lower degeneracy = better):
  knn_rmse, knn_r    error / correlation of the neighbour mean vs observed speed
  slope              slope of the neighbour mean on the observed speed (1 = look-alikes fully distinct, 0 = no information)
  fast_slow_share    observed >= 550: share of look-alikes <= 400
  p600_for_ge650     observed >= 650: share of look-alikes >= 600 (climatology 0.05)
  slow_fast_share    observed <= 400: share of look-alikes >= 550
Output: data/knn_degeneracy/feature_ablation.csv
"""
import json

import numpy as np
import pandas as pd
import torch

K, EXCL_DAYS, STRIDE = 50, 13.0, 2
D = np.load('data/knn_degeneracy/inputs.npz', allow_pickle=True)
X0, XL, v, tday = D['X0'], D['XL'], D['v'], D['tday']
NAMES = ['A f0.30 ±20', 'B band ±15', 'A f0.35 ±12.5', 'D hole darkness', 'B boundary dist',
         'A ESWF ±7.5', 'H weak-field frac', 'H mean |B_r|']
FULL = json.load(open('data/nn1_dataset/meta.json'))['features']


def summary(X):
    good = np.isfinite(X).all(1)
    Xg, vg, tg = X[good], v[good], tday[good]
    Z = torch.tensor((Xg - Xg.mean(0)) / Xg.std(0), dtype=torch.float32)
    tt, vt = torch.tensor(tg), torch.tensor(vg)
    q = np.arange(0, len(Xg), STRIDE)
    nbs = np.empty((len(q), K), np.float32)
    for s in range(0, len(q), 2048):
        qi = torch.tensor(q[s:s + 2048])
        d = torch.cdist(Z[qi], Z)
        d[(tt[None, :] - tt[qi][:, None]).abs() < EXCL_DAYS] = float('inf')
        nbs[s:s + len(qi)] = vt[d.topk(K, largest=False).indices].numpy()
    vo, m = vg[q], nbs.mean(1)
    fast, slow, top = vo >= 550, vo <= 400, vo >= 650
    return dict(n=len(q), knn_rmse=np.sqrt(np.mean((m - vo) ** 2)), knn_r=np.corrcoef(m, vo)[0, 1],
                slope=np.polyfit(vo, m, 1)[0], fast_slow_share=(nbs[fast] <= 400).mean(),
                p600_for_ge650=(nbs[top] >= 600).mean(), slow_fast_share=(nbs[slow] >= 550).mean())


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    cases = [('all 8', 'single', list(range(8))), ('all 8 + noise', 'single', 'noise')]
    cases += [(f'only {NAMES[i]}', 'single', [i]) for i in range(8)]
    cases += [(f'all but {NAMES[i]}', 'single', [j for j in range(8) if j != i]) for i in range(8)]
    cases += [('all 8', 'lags', list(range(8))), ('all 8 + noise', 'lags', 'noise')]
    cases += [(f'all but {NAMES[i]}', 'lags', [j for j in range(8) if j != i]) for i in range(8)]
    rows = []
    for label, kind, cols in cases:
        base = X0 if kind == 'single' else XL
        if cols == 'noise':
            X = np.concatenate([base, rng.standard_normal((len(base), 1))], 1)
        elif kind == 'single':
            X = base[:, cols]
        else:
            X = base[:, [f * 8 + c for f in range(5) for c in cols]]
        r = summary(X)
        rows.append(dict(inputs=kind, case=label, n_dims=X.shape[1], **r))
        print(f"{kind:6s} {label:28s} dims={X.shape[1]:2d} rmse={r['knn_rmse']:6.2f} r={r['knn_r']:.3f} slope={r['slope']:.3f} "
              f"fast->slow={r['fast_slow_share']:.3f} p600|>=650={r['p600_for_ge650']:.3f} slow->fast={r['slow_fast_share']:.3f}",
              flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('data/knn_degeneracy/feature_ablation.csv', index=False)
    print('saved data/knn_degeneracy/feature_ablation.csv')
