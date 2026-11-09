"""Recurrence mapped back to the Sun (an NN1-compatible, per-longitude input) vs v(T - 27 d) at L1 (NN2 input).

Back-mapping: every measured ACE hour (t_obs, v, ACE Carrington lon) -> source time t_s = t_obs - 193.5 Rs / v,
source longitude = ACE Carrington longitude at t_s (ballistic, as in the look-alike matching).
bm(L, t_src): mean speed of hours whose source fell within +-DLON deg of L at t_s in [t_src - 27.27 d +- 3 d], counting
only hours observed before the issue time T - 4 d. Rows / settings as knn_full_combo_test.py (daily 193/211, lags 24/48/72).
Output: data/knn_degeneracy/backmap_test.csv
"""
import os

import numpy as np
import pandas as pd

from knn_feature_ablation import D, summary
from eval_matched_baselines import ace_hourly

T = pd.to_datetime(D['time']); v = D['v']; n = len(T)
D_KM = (215 - 21.5) * 695700
# Alignment (env KNN_ALIGN_BETA): transit = 96 h + beta (ballistic - 96 h); 1 = ballistic (published), 0 = fixed 96 h.
# The feature-ranking alignment check (2026-10-23) found beta = 0.25 the best alignment for the AIA features.
BETA = float(os.environ.get('KNN_ALIGN_BETA', '1'))
ALIGN_TAG = '' if BETA == 1 else f'_beta{BETA:g}'


def transit(vv):
    if BETA == 1:
        return pd.to_timedelta(D_KM / np.asarray(vv, float), unit='s')
    return pd.to_timedelta(96 * 3600 + BETA * (D_KM / np.asarray(vv, float) - 96 * 3600), unit='s')


src = T - transit(v)
DC = np.load('data/nn1_dataset/dlon_center.npy').astype(np.float32)
t0 = pd.to_datetime(pd.read_csv('data/nn1_dataset/times.csv', usecols=['time'])['time']).iloc[0]
hs = np.clip(np.rint((src - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy(), 0, DC.shape[1] - 1)
col = np.where(np.isfinite(DC[:, hs]), np.abs(DC[:, hs]), np.inf).argmin(0)     # Carrington lon (deg) of the CM column

# back-mapped ACE hours
o = pd.read_csv('data/preprocessed_window/multi_spacecraft_obs_20110101_20211231T235959.csv', parse_dates=['Time'])
o = o[o.Spacecraft_ID == 'ACE'].set_index('Time')
lon = o['HC_lon'].resample('1h').mean()
unw = pd.Series(np.unwrap(np.deg2rad(lon.interpolate(limit=48).to_numpy())), index=lon.index)   # unwrapped, for interpolation
vh = ace_hourly().dropna()
ts = vh.index - transit(vh.to_numpy())
ls = np.rad2deg(np.interp(ts.asi8, unw.index.asi8, unw.to_numpy())) % 360.0
order = np.argsort(ts.asi8)
TS, LS, VS, TO = ts.asi8[order], ls[order], vh.to_numpy()[order], vh.index.asi8[order]
ROT = 27.2753 * 86400e9


def backmap(dlon):
    out = np.full(n, np.nan)
    tq = src.asi8 - ROT; issue = (T - pd.Timedelta(days=4)).asi8
    a = np.searchsorted(TS, tq - 3 * 86400e9); b = np.searchsorted(TS, tq + 3 * 86400e9)
    for i in range(n):
        if b[i] <= a[i]:
            continue
        sl = slice(a[i], b[i])
        d = np.abs((LS[sl] - col[i] + 180.0) % 360.0 - 180.0)
        m = (d <= dlon) & (TO[sl] <= issue[i])
        if m.any():
            out[i] = VS[sl][m].mean()
    return out[:, None]


def match(A, obst, lag_h):
    ot = pd.DatetimeIndex(obst).as_unit('ns'); ok = ~np.asarray(ot.isna())
    tt, Ak = ot[ok].asi8, A[ok]
    tq = (src - pd.Timedelta(hours=lag_h)).asi8
    j = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - tq) < np.abs(tt[j] - tq), j - 1, j)
    X = Ak[j, :, col]; X[np.abs(tt[j] - tq) > 12 * 3600e9] = np.nan
    return X


if __name__ == '__main__':
    bm2, bm5 = backmap(2.0), backmap(5.0)
    rec = ace_hourly().reindex(T - pd.Timedelta(days=27)).to_numpy()[:, None]
    C1 = np.load('data/knn_degeneracy/daily_aia_features.npz', allow_pickle=True)
    C2 = np.load('data/knn_degeneracy/daily_aia211_features.npz', allow_pickle=True)
    k1 = list(C1['variants']).index('193_0.3_0.35')
    C_ = lambda *a: np.concatenate(a, 1)
    l193 = C_(*[match(C1['A'][k1], C1['obstime'][k1], L) for L in (0, 24, 48, 72)])
    c193, c211 = l193[:, :6], match(C2['A'][0], C2['obstime'][0], 0)
    R = pd.read_csv('data/pfss_connectivity/per_hour.csv', low_memory=False)
    fs = np.full(n, np.nan); fs[R.row.to_numpy()] = R.log10_fs.to_numpy(); fs = fs[:, None]
    keep = np.isfinite(l193).all(1) & np.isfinite(c211).all(1) & np.isfinite(fs[:, 0]) & np.isfinite(rec[:, 0]) \
        & np.isfinite(bm2[:, 0]) & np.isfinite(bm5[:, 0])
    print(f'common rows: {keep.sum()} of {n}')
    for nm, x in (('bm2', bm2), ('bm5', bm5), ('rec', rec)):
        print(f'  r({nm}, v(T)) = {np.corrcoef(x[keep, 0], v[keep])[0, 1]:.3f}   r({nm}, rec) = {np.corrcoef(x[keep, 0], rec[keep, 0])[0, 1]:.3f}')
    mask = lambda X: np.where(keep[:, None], X, np.nan)
    base = C_(l193, c211, fs)
    cases = [('193', c193), ('only rec v(T-27d)', rec), ('only bm +-2', bm2), ('only bm +-5', bm5),
             ('193 + rec', C_(c193, rec)), ('193 + bm +-2', C_(c193, bm2)), ('193 + bm +-5', C_(c193, bm5)),
             ('NN1 set: 193 lags + 211 + fs', base), ('NN1 set + rec', C_(base, rec)),
             ('NN1 set + bm +-2', C_(base, bm2)), ('NN1 set + bm +-5', C_(base, bm5)), ('NN1 set + bm +-5 + rec', C_(base, bm5, rec))]
    rows = []
    for label, X in cases:
        r = summary(mask(X)); rows.append(dict(case=label, dims=X.shape[1], **r))
        print(f"{label:30s} dims={X.shape[1]:2d} slope={r['slope']:.3f} fast->slow={r['fast_slow_share']:.3f} "
              f"p600|>=650={r['p600_for_ge650']:.3f} slow->fast={r['slow_fast_share']:.3f} r={r['knn_r']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv('data/knn_degeneracy/backmap_test.csv', index=False)
