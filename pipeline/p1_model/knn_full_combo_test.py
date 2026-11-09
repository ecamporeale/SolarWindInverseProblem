"""Look-alike test: 193 + 211 A (daily frames) with lags 24/48/72 h, PFSS log10 fs, in-situ v(T - 27 d), speed at issue time v(T - 4 d); no HMI.
Lagged values: same Carrington column, daily frame nearest t_src - L (<= 12 h). fs at t_src only.
All cases on the same rows. Output: data/knn_degeneracy/full_combo_test.csv
"""
import numpy as np
import pandas as pd

from knn_feature_ablation import D, summary
from eval_matched_baselines import ace_hourly

T = pd.to_datetime(D['time']); v = D['v']; n = len(T)
from knn_backmap_test import src, col, ALIGN_TAG     # source time and CM column (alignment: env KNN_ALIGN_BETA)
LAGS = [0, 24, 48, 72]


def match(A, obst, lag_h):
    ot = pd.DatetimeIndex(obst).as_unit('ns'); ok = ~np.asarray(ot.isna())
    tt, Ak = ot[ok].asi8, A[ok]
    tq = (src - pd.Timedelta(hours=lag_h)).asi8
    j = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
    j = np.where(np.abs(tt[j - 1] - tq) < np.abs(tt[j] - tq), j - 1, j)
    X = Ak[j, :, col]; X[np.abs(tt[j] - tq) > 12 * 3600e9] = np.nan
    return X


C1 = np.load('data/knn_degeneracy/daily_aia_features.npz', allow_pickle=True)
C2 = np.load('data/knn_degeneracy/daily_aia211_features.npz', allow_pickle=True)
k1 = list(C1['variants']).index('193_0.3_0.35')
F193 = {L: match(C1['A'][k1], C1['obstime'][k1], L) for L in LAGS}
F211 = {L: match(C2['A'][0], C2['obstime'][0], L) for L in LAGS}
R = pd.read_csv(f'data/pfss_connectivity/per_hour{ALIGN_TAG}.csv', low_memory=False)
fs = np.full(n, np.nan); fs[R.row.to_numpy()] = R.log10_fs.to_numpy(); fs = fs[:, None]
rec = ace_hourly().reindex(T - pd.Timedelta(days=27)).to_numpy()[:, None]
pers = ace_hourly().reindex(T - pd.Timedelta(days=4)).to_numpy()[:, None]      # speed at issue time (4-day lead)

C_ = lambda *a: np.concatenate(a, 1)
c193, c211 = F193[0], F211[0]
l193, l211 = C_(*[F193[L] for L in LAGS]), C_(*[F211[L] for L in LAGS])
keep = np.isfinite(l193).all(1) & np.isfinite(l211).all(1) & np.isfinite(fs[:, 0]) & np.isfinite(rec[:, 0]) & np.isfinite(pers[:, 0])
print(f'common rows: {keep.sum()} of {n}')
mask = lambda X: np.where(keep[:, None], X, np.nan)
rng = np.random.default_rng(0)
noise = lambda k: rng.standard_normal((n, k))
cases = [('193', c193),
         ('193 + 211', C_(c193, c211)),
         ('193 + 211 + fs', C_(c193, c211, fs)),
         ('193 lags', l193), ('193 + 18 noise (control for lags)', C_(c193, noise(18))),
         ('193 lags + 211 (current)', C_(l193, c211)),
         ('193 + 211, both lagged', C_(l193, l211)), ('193 + 211 + 36 noise (control)', C_(c193, c211, noise(36))),
         ('193 + 211 lagged + fs', C_(l193, l211, fs)),
         ('193 + 211 + fs + rec', C_(c193, c211, fs, rec)),
         ('193 lags + 211 + fs + rec', C_(l193, c211, fs, rec)),
         ('193 + 211 lagged + fs + rec', C_(l193, l211, fs, rec)),
         ('193 + rec', C_(c193, rec)), ('193 lags + rec', C_(l193, rec)),
         ('only pers v(T-4d)', pers), ('193 + pers', C_(c193, pers)),
         ('193 lags + 211 + fs + pers', C_(l193, c211, fs, pers)),
         ('193 lags + 211 + fs + rec + pers', C_(l193, c211, fs, rec, pers))]
rows = []
for label, X in cases:
    r = summary(mask(X)); rows.append(dict(case=label, dims=X.shape[1], **r))
    print(f"{label:36s} dims={X.shape[1]:2d} slope={r['slope']:.3f} fast->slow={r['fast_slow_share']:.3f} "
          f"p600|>=650={r['p600_for_ge650']:.3f} slow->fast={r['slow_fast_share']:.3f} r={r['knn_r']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(f'data/knn_degeneracy/full_combo_test{ALIGN_TAG}.csv', index=False)
