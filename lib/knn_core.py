"""Shared machinery for the fold-consistent look-alike (kNN) tests (overnight tests 1-6, 2026-10-10).

Rows: matched ACE hours of data/knn_degeneracy/inputs.npz (ballistic matching; CM column at the source hour t_src).
Look-alikes: k = 50 nearest in standardised input space, |dT| >= 13 d, and only from OTHER development batches
(fold-consistent pool; see README 'period-matching caveat'). Queries: every 2nd row.
Base inputs (best set so far): daily 193 A set-2a AIA quantities at t_src and t_src - 24/48/72 h, PFSS log10 fs,
back-mapped recurrence bm (+-5 deg), in-situ v(T - 27 d).
"""
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(48)                    # also when a caller sets OMP_NUM_THREADS=1 for its worker pool

import knn_backmap_test as KB
from knn_feature_ablation import D
from eval_matched_baselines import ace_hourly

T, v, n, src, col = KB.T, KB.v, KB.n, KB.src, KB.col
tday, batch = D['tday'], D['batch']
OUT = 'data/knn_degeneracy/overnight'
ROT_D = 27.2753


def match_at(A, obst, lag_h=0.0, cols=None):
    """Value of a daily [days, f, 360] feature cube for every row: daily frame nearest t_src - lag_h (<= 12 h),
    Carrington column `cols` (default: the row's CM column)."""
    cols = col if cols is None else np.asarray(cols) % 360
    ot = pd.DatetimeIndex(obst).as_unit('ns'); ok = ~np.asarray(ot.isna())
    tt, Ak = ot[ok].asi8, A[ok]
    tq = (src - pd.Timedelta(hours=float(lag_h))).asi8
    j = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - tq) < np.abs(tt[j] - tq), j - 1, j)
    X = Ak[j, :, cols].astype(np.float64)
    X[np.abs(tt[j] - tq) > 12 * 3600e9] = np.nan
    return X


def daily193():
    C = np.load('data/knn_degeneracy/daily_aia_features.npz', allow_pickle=True)
    k = list(C['variants']).index('193_0.3_0.35')
    return C['A'][k], C['obstime'][k]


def base_inputs():
    import os
    A, O = daily193()
    cache = f'{OUT}/base_inputs{KB.ALIGN_TAG}.npz'
    if os.path.exists(cache):
        C = np.load(cache)
        return dict(A193=A, O193=O, **{k: C[k] for k in C.files})
    l193 = np.concatenate([match_at(A, O, L) for L in (0, 24, 48, 72)], 1)
    R = pd.read_csv(f'data/pfss_connectivity/per_hour{KB.ALIGN_TAG}.csv', low_memory=False)
    fs = np.full(n, np.nan); fs[R.row.to_numpy()] = R.log10_fs.to_numpy()
    foot_lat = np.full(n, np.nan); foot_lat[R.row.to_numpy()] = R.foot_lat.to_numpy()
    rec = ace_hourly().reindex(T - pd.Timedelta(days=27)).to_numpy()
    bm = KB.backmap(5.0)[:, 0]
    out = dict(l193=l193, c193=l193[:, :6], fs=fs[:, None], rec=rec[:, None], bm=bm[:, None], foot_lat=foot_lat)
    os.makedirs(OUT, exist_ok=True); np.savez(cache, **out)
    return dict(A193=A, O193=O, **out)


def best(B):
    return np.concatenate([B['l193'], B['fs'], B['bm'], B['rec']], 1)


def backmap_at(target_lon, dlon=5.0):
    """Back-mapped recurrence at an arbitrary Carrington longitude per row (KB.backmap generalised)."""
    out = np.full(n, np.nan)
    tq = src.asi8 - KB.ROT; issue = (T - pd.Timedelta(days=4)).asi8
    a = np.searchsorted(KB.TS, tq - 3 * 86400e9); b = np.searchsorted(KB.TS, tq + 3 * 86400e9)
    tl = np.asarray(target_lon, float) % 360.0
    for i in range(n):
        if b[i] <= a[i] or not np.isfinite(tl[i]):
            continue
        sl = slice(a[i], b[i])
        d = np.abs((KB.LS[sl] - tl[i] + 180.0) % 360.0 - 180.0)
        m = (d <= dlon) & (KB.TO[sl] <= issue[i])
        if m.any():
            out[i] = KB.VS[sl][m].mean()
    return out[:, None]


def neighbours(X, keep, K=50, excl=13.0, pool='other', stride=2):
    """Global row indices of queries and of their K look-alikes."""
    rows = np.flatnonzero(keep)
    Xg = X[rows]
    Z = torch.tensor((Xg - Xg.mean(0)) / Xg.std(0), dtype=torch.float32)
    tt, bb = torch.tensor(tday[rows]), torch.tensor(batch[rows])
    q = np.arange(0, len(rows), stride)
    NB = np.empty((len(q), K), np.int64)
    for s in range(0, len(q), 2048):
        qi = torch.tensor(q[s:s + 2048])
        d = torch.cdist(Z[qi], Z)
        d[(tt[None] - tt[qi][:, None]).abs() < excl] = float('inf')
        if pool == 'other':
            d[bb[None] == bb[qi][:, None]] = float('inf')
        NB[s:s + len(qi)] = d.topk(K, largest=False).indices.numpy()
    return rows[q], rows[NB]


def metrics(qi, NB, vq=None, vnb_fast=None, vnb_slow=None):
    """Degeneracy metrics. vnb_fast / vnb_slow: per-row outcome used to judge a look-alike as slow (<= 400) for fast
    queries / fast (>= 550) for slow queries (default: its own hourly speed)."""
    vq = v if vq is None else vq
    vf = v if vnb_fast is None else vnb_fast
    vs = v if vnb_slow is None else vnb_slow
    vo, m = vq[qi], v[NB].mean(1)
    fast, slow, top = vo >= 550, vo <= 400, vo >= 650
    return dict(n_rows=len(np.unique(np.concatenate([qi, NB.ravel()]))), n_query=len(qi),
                slope=float(np.polyfit(vo, m, 1)[0]), fast_slow_share=float((vf[NB][fast] <= 400).mean()),
                slow_fast_share=float((vs[NB][slow] >= 550).mean()), p600_for_ge650=float((v[NB][top] >= 600).mean()),
                knn_rmse=float(np.sqrt(np.mean((m - vo) ** 2))))


def run_cases(cases, keep, label_test, extra_cols=None):
    rows = []
    for label, X in cases:
        qi, NB = neighbours(X, keep)
        r = dict(test=label_test, case=label, dims=X.shape[1], **metrics(qi, NB))
        rows.append(r)
        print(f"[{label_test}] {label:42s} dims={X.shape[1]:3d} slope={r['slope']:.3f} fast->slow={r['fast_slow_share']:.3f} "
              f"slow->fast={r['slow_fast_share']:.3f} p600|>=650={r['p600_for_ge650']:.3f} rmse={r['knn_rmse']:.1f}", flush=True)
    return pd.DataFrame(rows)


def noise(k, seed=0):
    return np.random.default_rng(seed).standard_normal((n, k))
