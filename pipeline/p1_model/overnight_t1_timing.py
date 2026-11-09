"""Test 1: how much of the degeneracy is timing error of the ballistic matching?
A look-alike of a fast query (v >= 550) counts as slow only if its ACE speed stays <= 400 throughout +-W h of its
matched hour; a look-alike of a slow query (v <= 400) counts as fast only if it stays >= 550 throughout +-W h.
W = 0, 6, 12, 24 h. Inputs: 193 single frame, and all the model's inputs (26 values).
Output: data/knn_degeneracy/overnight/t1_timing.csv
"""
import os

import numpy as np
import pandas as pd

# Paper alignment (decision 2026-10-23): look-alike source cell = central-meridian column 96 h before the ACE hour
# (KNN_ALIGN_BETA=0), as in the feature ranking; KNN_ALIGN_BETA=1 gives the earlier ballistic matching.
os.environ.setdefault('KNN_ALIGN_BETA', '0')
import knn_core as K  # noqa: E402

B = K.base_inputs()
X_all = np.concatenate([B['l193'], B['fs'], B['bm']], 1)            # the model's 26 values (2026-10-24; was K.best, + v(T-27 d))
keep = np.isfinite(X_all).all(1)
ace = K.ace_hourly()
full = ace.reindex(pd.date_range(ace.index.min(), ace.index.max(), freq='h'))
rows = []
rng = np.random.default_rng(0)
rows_k = np.flatnonzero(keep)


def random_lookalikes(qi):
    # control: 50 random rows from OTHER development batches, |dT| >= 13 d (no information, same tolerance scoring)
    NB = np.empty((len(qi), 50), np.int64)
    for j, q in enumerate(qi):
        cand = rows_k[(K.batch[rows_k] != K.batch[q]) & (np.abs(K.tday[rows_k] - K.tday[q]) >= 13)]
        NB[j] = rng.choice(cand, 50, replace=False)
    return NB


for label, X in (('random look-alikes (control)', None), ('193 single frame', B['c193']), ('all inputs (193 lags + fs + bm)', X_all)):
    if X is None:
        qi, _ = K.neighbours(B['c193'], keep); NB = random_lookalikes(qi)
    else:
        qi, NB = K.neighbours(X, keep)
    for W in (0, 6, 12, 24):
        vmax = full.rolling(2 * W + 1, center=True, min_periods=1).max().reindex(K.T).to_numpy()
        vmin = full.rolling(2 * W + 1, center=True, min_periods=1).min().reindex(K.T).to_numpy()
        vmax = np.where(np.isfinite(vmax), vmax, K.v); vmin = np.where(np.isfinite(vmin), vmin, K.v)
        # strict: a look-alike of a fast query is 'slow' only if max over +-W <= 400; of a slow query 'fast' only if min >= 550
        r = dict(test='t1', case=label, window_h=W, **K.metrics(qi, NB, vnb_fast=vmax, vnb_slow=vmin))
        rows.append(r)
        print(f"[t1] {label:38s} +-{W:2d} h  fast->slow={r['fast_slow_share']:.3f}  slow->fast={r['slow_fast_share']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(f'{K.OUT}/t1_timing{K.KB.ALIGN_TAG}.csv', index=False)
