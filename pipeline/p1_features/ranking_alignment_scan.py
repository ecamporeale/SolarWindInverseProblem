"""Which image-to-ACE alignment do the AIA features want? (alignment check of the feature ranking, 2026-10-23)

Rows: every measured development ACE hour T (ICME-masked, as the training target), paired with the central-meridian
features of the frame on the hour nearest T - lag.

(1) Lag-rule scan: lag = 96 h + beta (ballistic - 96 h), ballistic = 193.5 R_sun / v(T) (the look-alike rule, from
    21.5 R_sun at the observed speed); beta = 0 is the fixed 96-h lag of the ranking, 1 is ballistic; also ballistic
    plus 12/24/36 h, and ballistic with B0's residual acceleration (L = 50 R_sun, V_ref = 40 km/s). Scores on rows
    common to all rules: mean within-batch r of the leading feature; out-of-fold OLS of the six B0 features (r, RMSE,
    and RMSE / bias for v >= 550 and v < 400 km/s).
(2) Superposed epoch, per bin of observed speed: mean of each area feature in the frame L hours before T, L = 24..168 h;
    the lag of the maximum, against the bin's median ballistic transit. No fit and no pooling over speeds, so the
    over-representation of slow wind cannot affect it.

  python pipeline/p1_features/ranking_alignment_scan.py
Output: data/ranking_alignment_scan_pixel.csv, data/ranking_alignment_epoch_pixel.csv
"""
import contextlib
import io
import os
import runpy

import numpy as np
import pandas as pd

os.environ['FEATURE_SOURCE'] = 'pixel'; os.environ['RANK_ALIGN'] = 'lag96'
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path('pipeline/p1_features/final_feature_ranking.py')       # feature table and ACE series
folds, ace, A_FEATS = G['folds'], G['ace'], G['A_FEATS']
RS = 695700.0
SEL = ['A: area f=0.30, w=20, off=0', 'B: ecliptic band |lat|<=15', 'A: area f=0.35, w=12.5, off=0',
       'D: mean log(I/med) of dark pixels, band |lat|<=30', 'B: hole-boundary distance, off=+10',
       'A: ESWF definition (f=0.35, w=7.5, off=0)']
img = G['aia'][['t'] + list(A_FEATS.values())].set_index('t')
Tdev = ace.index[folds.batch_of(ace.index.to_numpy()) >= 0]
v = ace.reindex(Tdev).to_numpy(); bT = folds.batch_of(Tdev.to_numpy())
tau_b = (215 - 21.5) * RS / v / 3600                                          # h
r_ = np.linspace(21.5, 215, 400); acc = 40 * (1 - np.exp(-(r_ - 21.5) / 50))
tau_a = np.trapezoid(1.0 / ((v - acc[-1])[:, None] + acc[None, :]), r_ * RS, axis=1) / 3600


def frames(lag_h, cols):
    return img.reindex((Tdev - pd.to_timedelta(lag_h, unit='h')).round('h'))[cols].to_numpy(float)


# (1) lag-rule scan
cols = [A_FEATS[s] for s in SEL]
modes = {f'beta={b:g}': 96 + b * (tau_b - 96) for b in (0, 0.25, 0.5, 0.75, 1.0)}
modes.update({f'ballistic + {d} h': tau_b + d for d in (12, 24, 36)})
modes['ballistic, B0 acceleration'] = tau_a
modes['ballistic, B0 acceleration + 24 h'] = tau_a + 24
M = {k: frames(t, cols) for k, t in modes.items()}
ok = np.logical_and.reduce([np.isfinite(x).all(1) for x in M.values()])
y, b = v[ok], bT[ok]
fast, slow = y >= 550, y < 400
rows = []
for k, X in M.items():
    X = X[ok]
    pred = np.full(len(y), np.nan)
    for tr, va, te in folds.FOLDS.values():
        A_ = np.c_[np.ones(len(y)), X]; m = np.isin(b, tr)
        pred[b == te] = A_[b == te] @ np.linalg.lstsq(A_[m], y[m], rcond=None)[0]
    e = pred - y
    rows.append(dict(rule=k, n=len(y), median_lag_h=float(np.median(modes[k][ok])),
                     r_leading=np.mean([np.corrcoef(X[b == i, 0], y[b == i])[0, 1] for i in range(5)]),
                     r_six=np.mean([np.corrcoef(pred[b == i], y[b == i])[0, 1] for i in range(5)]),
                     rmse_six=np.sqrt(np.mean(e ** 2)), rmse_fast=np.sqrt(np.mean(e[fast] ** 2)), bias_fast=e[fast].mean(),
                     rmse_slow=np.sqrt(np.mean(e[slow] ** 2)), bias_slow=e[slow].mean()))
S = pd.DataFrame(rows)
S.to_csv('data/ranking_alignment_scan_pixel.csv', index=False)
pd.set_option('display.width', 200)
print(f'(1) lag rules, {ok.sum()} common ACE hours')
print(S.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# (2) superposed epoch per speed bin
F = {'area f=0.30 w=20': A_FEATS['A: area f=0.30, w=20, off=0'],
     'ESWF area': A_FEATS['A: ESWF definition (f=0.35, w=7.5, off=0)'],
     'ecliptic band 15': A_FEATS['B: ecliptic band |lat|<=15']}
lags = np.arange(24, 169, 3)
cube = np.stack([frames(float(L), list(F.values())) for L in lags], 1)      # [n, lag, feature]
ok = np.isfinite(cube).all((1, 2))
cube, vv, tb = cube[ok], v[ok], tau_b[ok]
edges = [250, 325, 375, 425, 475, 525, 575, 625, 900]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (vv >= lo) & (vv < hi)
    m = cube[s].mean(0)
    r = dict(v_lo=lo, v_hi=hi, n=int(s.sum()), ballistic_lag_h=float(np.median(tb[s])))
    for j, k in enumerate(F):
        r[f'peak_lag_h {k}'] = int(lags[m[:, j].argmax()])
    rows.append(r)
E = pd.DataFrame(rows)
E.to_csv('data/ranking_alignment_epoch_pixel.csv', index=False)
print('\n(2) lag of the maximum mean feature, per bin of observed speed (bins below ~425 km/s have no maximum inside the range)')
print(E.round(0).to_string(index=False))
