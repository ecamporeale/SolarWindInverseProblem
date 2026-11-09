"""Look-alike test consistent with the development folds: look-alikes only from OTHER development batches (as a held-out
contiguous batch would see), vs the standard pool (any time, |dT| >= 13 d). Daily 193/211 (lags 24/48/72), PFSS fs,
v(T-27d), back-mapped recurrence (+-5 deg). Output: data/knn_degeneracy/other_batch_check.csv
"""
import os

import numpy as np
import pandas as pd
import torch

# Paper alignment (decision 2026-10-23): look-alike source cell = central-meridian column 96 h before the ACE hour
# (KNN_ALIGN_BETA=0), as in the feature ranking; KNN_ALIGN_BETA=1 gives the earlier ballistic matching.
os.environ.setdefault('KNN_ALIGN_BETA', '0')
from knn_feature_ablation import D  # noqa: E402
from knn_full_combo_test import match, C1, C2, k1
from knn_backmap_test import backmap, ALIGN_TAG
from eval_matched_baselines import ace_hourly

T = pd.to_datetime(D['time']); v, tday, batch = D['v'], D['tday'], D['batch']; n = len(v)
C_ = lambda *a: np.concatenate(a, 1)
l193 = C_(*[match(C1['A'][k1], C1['obstime'][k1], L) for L in (0, 24, 48, 72)]); c193 = l193[:, :6]
c211 = match(C2['A'][0], C2['obstime'][0], 0)
R = pd.read_csv(f'data/pfss_connectivity/per_hour{ALIGN_TAG}.csv', low_memory=False)
fs = np.full(n, np.nan); fs[R.row.to_numpy()] = R.log10_fs.to_numpy(); fs = fs[:, None]
rec = ace_hourly().reindex(T - pd.Timedelta(days=27)).to_numpy()[:, None]
bm = backmap(5.0)
keep = np.isfinite(l193).all(1) & np.isfinite(c211).all(1) & np.isfinite(fs[:, 0]) & np.isfinite(rec[:, 0]) & np.isfinite(bm[:, 0])
print(f'common rows: {keep.sum()}')


def summary(X, pool, K=50):
    Xg, vg, tg, bg = X[keep], v[keep], tday[keep], batch[keep]
    Z = torch.tensor((Xg - Xg.mean(0)) / Xg.std(0), dtype=torch.float32)
    tt, vt, bb = torch.tensor(tg), torch.tensor(vg), torch.tensor(bg)
    q = np.arange(0, len(Xg), 2); nbs = np.empty((len(q), K), np.float32)
    for s in range(0, len(q), 2048):
        qi = torch.tensor(q[s:s + 2048]); d = torch.cdist(Z[qi], Z)
        d[(tt[None] - tt[qi][:, None]).abs() < 13] = float('inf')
        if pool == 'other':
            d[bb[None] == bb[qi][:, None]] = float('inf')
        nbs[s:s + len(qi)] = vt[d.topk(K, largest=False).indices].numpy()
    vo, m = vg[q], nbs.mean(1)
    return dict(slope=np.polyfit(vo, m, 1)[0], fast_slow_share=(nbs[vo >= 550] <= 400).mean(), knn_rmse=np.sqrt(np.mean((m - vo) ** 2)))


rng = np.random.default_rng(0)
cases = [('193', c193), ('193 + 6 noise', C_(c193, rng.standard_normal((n, 6)))), ('193 + 211', C_(c193, c211)), ('193 + fs', C_(c193, fs)),
         ('193 lags', l193), ('193 + 18 noise', C_(c193, rng.standard_normal((n, 18)))), ('193 + rec', C_(c193, rec)), ('193 + bm', C_(c193, bm)),
         ('193 lags + fs', C_(l193, fs)), ('193 lags + fs + bm', C_(l193, fs, bm)), ('193 lags + 211 + fs + bm', C_(l193, c211, fs, bm)),
         ('193 lags + fs + bm + rec', C_(l193, fs, bm, rec)), ('193 lags + 211 + fs + bm + rec', C_(l193, c211, fs, bm, rec))]
rows = []
for pool in ('any', 'other'):
    for label, X in cases:
        r = summary(X, pool); rows.append(dict(pool=pool, case=label, **r))
        print(f"{pool:5s} {label:32s} slope={r['slope']:.3f} fast->slow={r['fast_slow_share']:.3f} knn_rmse={r['knn_rmse']:.1f}", flush=True)
pd.DataFrame(rows).to_csv(f'data/knn_degeneracy/other_batch_check{ALIGN_TAG}.csv', index=False)
