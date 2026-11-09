"""Final candidate-feature ranking on the development folds (folds.py).

One basis for every row:
  - frames: hourly AIA FITS (QUALITY == 0) joined with the HMI map cache on the hour,
    development window only, all candidate features finite (one common row set);
  - every image feature computed from the map caches (no pixel-level features mixed in);
  - statistic: mean over the 5 batches of the within-batch Pearson r with ACE v(t+4d);
  - 95% CI: block bootstrap, Carrington rotations as blocks, resampled within batch, B = 2000;
  - multi-value latitude profiles: ridge, out-of-sample prediction (each batch predicted by
    the model trained on its fold's 3 training batches), then scored like a single feature.
Columns also give: worse than #1 (paired CI of |r_1| - |r_k| excludes 0), and redundancy
(pooled correlation with the #1 feature).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import folds

B = 2000
CR_DAYS = 27.2753
PW = 'data/preprocessed_window'
CACHE = 'data/ecliptic_map_cache'

import os
SOURCE = os.environ.get('FEATURE_SOURCE', 'cache')   # 'cache' (0.5 deg map cache) or 'pixel'
import channel_cfg as CH
SUF = '' if SOURCE == 'cache' else CH.SUF_PIXEL
q = pd.read_csv(f'{PW}/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
q['t'] = q.obstime.dt.round('h')
good_t = set(q.loc[q.quality == 0, 't'])
if SOURCE == 'pixel':
    good_t = CH.good_hours()
    aia = pd.read_pickle(CH.PIXEL_AIA)
    aia['t'] = pd.to_datetime(aia.obstime).dt.round('h')
    aia = aia[aia['t'].isin(good_t)]
    hmi = pd.read_pickle('data/pixel_features_hmi.pkl')
    hmi['t'] = pd.to_datetime(hmi.obstime).dt.round('h')
else:
    aia = pd.read_pickle('data/aia_features_cache_fits_2011_2021.pkl')
    aia['t'] = pd.to_datetime(aia.obstime).dt.round('h')
    q = pd.read_csv(f'{PW}/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
    q['t'] = q.obstime.dt.round('h')
    good_t = set(q.loc[q.quality == 0, 't'])
    aia['row_aia'] = np.arange(len(aia))
    aia = aia[aia['use'] & aia['t'].isin(good_t)]
    hmi = pd.read_pickle('data/hmi_features_cache_hmi_dev.pkl')
    hmi['t'] = pd.to_datetime(hmi.obstime).dt.round('h')
    hmi['row_hmi'] = np.arange(len(hmi))

    # old NN1 inputs, from the caches: ecliptic trace (|lat|<=3 deg), central +/-7.5 deg
    ga = np.load(f'{CACHE}/grid_fits_2011_2021.npz')
    lat_rows = np.abs(ga['lats']) <= 3
    cen_cols = np.abs(ga['dlons']) <= 7.5
    Ma = np.load(f'{CACHE}/maps_fits_2011_2021.npy', mmap_mode='r')
    Mh = np.load(f'{CACHE}/maps_hmi_dev.npy', mmap_mode='r')
    meta_a = pd.read_csv(f'{CACHE}/meta_fits_2011_2021.csv').sort_values('idx').reset_index(drop=True)
    meta_h = pd.read_csv(f'{CACHE}/meta_hmi_dev.csv').sort_values('idx').reset_index(drop=True)


    def trace(M, absval=False):
        out = np.empty(M.shape[0], np.float32)
        for lo in range(0, M.shape[0], 2000):
            X = M[lo:lo + 2000][:, lat_rows][:, :, cen_cols].astype(np.float32)
            out[lo:lo + 2000] = np.nanmean(np.abs(X) if absval else X, axis=(1, 2))
        return out


    aia['OLD NN1 input: AIA ecliptic-trace intensity'] = trace(Ma)[meta_a['idx'].to_numpy()][aia['row_aia'].to_numpy()]
    hmi['OLD NN1 input: HMI ecliptic-trace |B_los|'] = trace(Mh, absval=True)[meta_h['idx'].to_numpy()][hmi['row_hmi'].to_numpy()]


A_FEATS = {
    'A: ESWF definition (f=0.35, w=7.5, off=0)': 'A_area f=0.35 w=7.5 off=+0',
    'A: area f=0.35, w=12.5, off=0': 'A_area f=0.35 w=12.5 off=+0',
    'A: area f=0.30, w=20, off=0': 'A_area f=0.3 w=20.0 off=+0',
    'A: area f=0.35, w=7.5, off=-10 (east)': 'A_area f=0.35 w=7.5 off=-10',
    'B: ecliptic band |lat|<=5': 'B_eclband |lat|<=5',
    'B: ecliptic band |lat|<=10': 'B_eclband |lat|<=10',
    'B: ecliptic band |lat|<=15': 'B_eclband |lat|<=15',
    'B: ecliptic band |lat|<=20': 'B_eclband |lat|<=20',
    'B: ecliptic band |lat|<=30': 'B_eclband |lat|<=30',
    'B: ecliptic band |lat|<=45': 'B_eclband |lat|<=45',
    'B: reach (min |lat| of hole in strip)': 'B_reach (min |lat| of hole in strip)',
    'B: hole-boundary distance, off=-10': 'B_dchb signed dist off=-10',
    'B: hole-boundary distance, off=0': 'B_dchb signed dist off=+0',
    'B: hole-boundary distance, off=+10': 'B_dchb signed dist off=+10',
    'OLD NN1 input: AIA ecliptic-trace intensity': 'OLD NN1 input: AIA ecliptic-trace intensity',
}
if SOURCE == 'pixel':
    A_FEATS.update({
        'D: mean log(I/med) of dark pixels, band |lat|<=30': 'D_mean log(I/med) of dark pixels, band |lat|<=30',
        'D: mean log(I/med) of dark pixels, strip': 'D_mean log(I/med) of dark pixels, strip',
        'D: 5th pct log(I/med), band |lat|<=30': 'D_5th pct log(I/med), band |lat|<=30',
        'D: 10th pct log(I/med), band |lat|<=30': 'D_10th pct log(I/med), band |lat|<=30',
        'D: 10th pct log(I/med), strip': 'D_10th pct log(I/med), strip',
        'D: integrated darkness, band |lat|<=30': 'D_integrated darkness, band |lat|<=30',
        'D: integrated darkness, strip': 'D_integrated darkness, strip',
        'D: soft-threshold area, band |lat|<=30': 'D_soft-threshold area, band |lat|<=30',
        'D: soft-threshold area, strip': 'D_soft-threshold area, strip',
    })
H_FEATS = {
    'H: PIL distance, sigma=20, off=0': 'H2_pildist sigma=20 off=+0',
    'H: PIL distance, sigma=20, off=-10': 'H2_pildist sigma=20 off=-10',
    'H: PIL distance, sigma=10, off=0': 'H2_pildist sigma=10 off=+0',
    'H: PIL distance, sigma=5, off=0': 'H2_pildist sigma=5 off=+0',
    'H: flux imbalance, w=20, |lat|<=70': 'H1_imbal w=20.0 k=70',
    'H: flux imbalance, w=20, |lat|<=30': 'H1_imbal w=20.0 k=30',
    'H: |net B_r|, w=20, |lat|<=70': 'H1_absnet w=20.0 k=70',
    'H: mean |B_r|, w=7.5, |lat|<=30': 'H1_meanabs w=7.5 k=30',
    'H: weak-field fraction <10 G, w=7.5, |lat|<=30': 'H1_weakfrac<10G w=7.5 k=30',
    'H: low-order SH (l<=2) PIL distance, off=0': 'H4_pildist_lowl off=+0',
    'H: low-order SH dipole tilt': 'H4_dipole_tilt_deg',
    'H: low-order SH |B| at sub-Earth': 'H4_|Blowl| subEarth',
    'OLD NN1 input: HMI ecliptic-trace |B_los|': 'OLD NN1 input: HMI ecliptic-trace |B_los|',
}
c_dark = [c for c in aia.columns if c.startswith('C_dark')]
c_logi = [c for c in aia.columns if c.startswith('C_logI')]
h3 = [c for c in hmi.columns if c.startswith('H3_')]

d = aia[['t', 'obstime'] + list(A_FEATS.values()) + c_dark + c_logi].merge(
    hmi[['t'] + list(H_FEATS.values()) + h3], on='t')
d['obstime'] = pd.to_datetime(d.obstime)
d['batch'] = folds.batch_of(d.obstime.to_numpy())

obs = pd.read_csv(f'{PW}/multi_spacecraft_obs_20110101_20211231T235959.csv', parse_dates=['Time'])
ace = obs[obs.Spacecraft_ID == 'ACE'].drop_duplicates('Time').set_index('Time')['Speed'].sort_index()
ace = ace[np.isfinite(ace)]
ax_, ay_ = ace.index.astype('int64').to_numpy(), ace.to_numpy()


def nearest(qn, tol_s=1800):
    i = np.clip(np.searchsorted(ax_, qn), 1, len(ax_) - 1)
    best = np.where(np.abs(ax_[i] - qn) < np.abs(ax_[i - 1] - qn), i, i - 1)
    return np.where(np.abs(ax_[best] - qn) <= tol_s * 1e9, ay_[best], np.nan)


ALIGN = CH.ALIGN if SOURCE == 'pixel' else 'lag96'
D_KM = (215 - 21.5) * 695700                 # ballistic source distance, as knn_backmap_test.py (look-alike analysis)


def ace_rows(img, tq, T):
    """Rows indexed by ACE hour T: features of the frame on the hour nearest tq (NaN where there is none)."""
    out = img.reindex(pd.DatetimeIndex(tq).round('h')).reset_index(drop=True)
    out['T'] = T.to_numpy()
    out['y'] = ace.reindex(T).to_numpy()
    return out


if ALIGN == 'lag96':
    tn = d.obstime.astype('int64').to_numpy()
    d['y'] = nearest(tn + int(96 * 3.6e12))
    d['REF: 27-day recurrence v(t+4d-27d)'] = nearest(tn + int(96 * 3.6e12) - int(CR_DAYS * 86400e9))
    d['REF: persistence v(t)'] = nearest(tn)
    d['tblock'] = d.obstime
else:
    # one row per measured development ACE hour T, paired with the frame at t_src = T - D_KM / v(T); batch and rotation
    # blocks follow T (as in training). Reference names are kept for the downstream scripts: the recurrence is v(T - 27 d)
    # in both alignments, and "persistence v(t)" is the speed at the image time t_src.
    assert ALIGN == 'backmap', ALIGN
    img = d.drop(columns='batch').set_index('t')
    Tdev = ace.index[folds.batch_of(ace.index.to_numpy()) >= 0]
    tsrc = Tdev - pd.to_timedelta(D_KM / ace.reindex(Tdev).to_numpy(), unit='s')
    d = ace_rows(img, tsrc, Tdev)
    d['batch'] = folds.batch_of(d['T'].to_numpy())
    d['REF: 27-day recurrence v(t+4d-27d)'] = nearest(d['T'].astype('int64').to_numpy() - int(CR_DAYS * 86400e9))
    d['REF: persistence v(t)'] = nearest(pd.DatetimeIndex(tsrc).astype('int64').to_numpy())
    d['tblock'] = d['T']
    print(f'back-mapped: {len(Tdev)} ACE hours, {d.obstime.notna().sum()} with a frame within 1 h of t_src; '
          f'transit median {np.median((Tdev - tsrc) / pd.Timedelta(days=1)):.2f} d')
REF = ['REF: 27-day recurrence v(t+4d-27d)', 'REF: persistence v(t)']
need = ['y'] + list(A_FEATS.values()) + list(H_FEATS.values()) + c_dark + c_logi + h3 + REF
d = d.replace([np.inf, -np.inf], np.nan)
nn = d[d.batch >= 0][need].isna()
print('rows lost per column (top 8):', nn.sum().sort_values(ascending=False).head(8).to_dict())
print('batch 3 losses (top 6):', nn[d.loc[d.batch >= 0, 'batch'] == 3].sum().sort_values(ascending=False).head(6).to_dict())
d = d[(d.batch >= 0)].dropna(subset=need).reset_index(drop=True)
print(f'common rows: {len(d)}; per batch:', d.groupby('batch').size().to_dict())

# ridge out-of-sample predictions for latitude profiles
PROFILES = {'C: AIA dark-fraction latitude profile (ridge)': c_dark,
            'C: AIA log(I/median) latitude profile (ridge)': c_logi,
            'H: HMI latitude profiles, |B_r|, |net|, imbalance (ridge)': h3}
for name, cols in PROFILES.items():
    pred = np.full(len(d), np.nan)
    for tr, va, te in folds.FOLDS.values():
        mtr, mte = d.batch.isin(tr).to_numpy(), (d.batch == te).to_numpy()
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(d.loc[mtr, cols], d.y[mtr])
        pred[mte] = m.predict(d.loc[mte, cols])
    d[name] = pred

FEATS = {**{k: v for k, v in A_FEATS.items()}, **{k: v for k, v in H_FEATS.items()},
         **{k: k for k in PROFILES}, **{k: k for k in REF}}

# block bootstrap via sufficient statistics
y = d.y.to_numpy()
d['block'] = d.batch * 1000 + ((d.tblock - pd.Timestamp('2010-01-01')).dt.total_seconds() // (CR_DAYS * 86400)).astype(int)
blocks, binv = np.unique(d.block.to_numpy(), return_inverse=True)
bb = (blocks // 1000).astype(int)
nb = len(blocks)
bs = lambda v: np.bincount(binv, weights=v, minlength=nb)
Sn, Sy, Syy = bs(np.ones(len(d))), bs(y), bs(y * y)
SX = {k: (bs(d[c].to_numpy(float)), bs(d[c].to_numpy(float) ** 2), bs(d[c].to_numpy(float) * y)) for k, c in FEATS.items()}
rng = np.random.default_rng(0)
W = np.zeros((nb, B))
for b in range(5):
    idx = np.where(bb == b)[0]
    dr = rng.choice(idx, size=(len(idx), B))
    for r in range(B):
        W[:, r] += np.bincount(dr[:, r], minlength=nb)


def mean_r(w):
    out = {}
    for k, (sx, sxx, sxy) in SX.items():
        rs = []
        for b in range(5):
            wb = w * ((bb == b)[:, None] if w.ndim == 2 else (bb == b))
            n, a, aa, axy, ty, tyy = Sn @ wb, sx @ wb, sxx @ wb, sxy @ wb, Sy @ wb, Syy @ wb
            rs.append((n * axy - a * ty) / np.sqrt((n * aa - a ** 2) * (n * tyy - ty ** 2)))
        out[k] = np.array(rs)
    return out


pt = mean_r(np.ones(nb))
bt = mean_r(W)
rows = []
for k in FEATS:
    m_pt, m_bt = pt[k].mean(), bt[k].mean(0)
    lo, hi = np.percentile(m_bt, [2.5, 97.5])
    rows.append(dict(feature=k, mean_r=m_pt, lo=lo, hi=hi, abs_r=abs(m_pt),
                     sign_consistent=bool(np.all(np.sign(pt[k]) == np.sign(m_pt))), **{f'b{b}': pt[k][b] for b in range(5)}))
res = pd.DataFrame(rows).sort_values('abs_r', ascending=False).reset_index(drop=True)
top = res.feature[0]
worse, redund = [], []
for k in res.feature:
    dlt = np.abs(bt[top].mean(0)) - np.abs(bt[k].mean(0))
    lo, hi = np.percentile(dlt, [2.5, 97.5])
    worse.append('—' if k == top else ('yes' if lo > 0 else 'no'))
    redund.append(np.corrcoef(d[FEATS[top]], d[FEATS[k]])[0, 1])
res['worse_than_1'] = worse
res['corr_with_1'] = redund
res.insert(0, 'rank', np.arange(1, len(res) + 1))
res.to_csv(f'data/final_feature_ranking{SUF}.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
show = res.copy()
show['mean r [95% CI]'] = [f'{m:+.3f} [{l:+.3f}, {h:+.3f}]' for m, l, h in zip(show.mean_r, show.lo, show.hi)]
print(show[['rank', 'feature', 'mean r [95% CI]', 'b0', 'b1', 'b2', 'b3', 'b4', 'sign_consistent', 'worse_than_1', 'corr_with_1']]
      .to_string(index=False, float_format=lambda v: f'{v:+.2f}'))

# feature matrix on the common rows, columns named as in the table, for follow-up analyses
X = pd.DataFrame({k: d[c].to_numpy(float) for k, c in FEATS.items()})
X['y'], X['batch'], X['obstime'], X['block'] = y, d.batch.to_numpy(), d.obstime.to_numpy(), d.block.to_numpy()
X.to_pickle(f'data/final_feature_matrix{SUF}.pkl')

if ALIGN == 'backmap':
    # Alignment comparison on ACE hours common to all alignments (single features; point estimates): the ballistic
    # source time; fixed lags of 2-5 d; and a control with the back-mapped lags permuted within each batch (same
    # lag distribution, but not the hour's own speed).
    rng = np.random.default_rng(1)
    vT = ace.reindex(Tdev).to_numpy()
    tau = pd.to_timedelta(D_KM / vT, unit='s').to_numpy()
    bT = folds.batch_of(Tdev.to_numpy())
    tau_p = tau.copy()
    for b in range(5):
        i = np.where(bT == b)[0]
        tau_p[i] = tau[rng.permutation(i)]
    single = list(A_FEATS.values()) + list(H_FEATS.values())
    modes = {'backmap': Tdev - pd.to_timedelta(tau),
             **{f'lag{L}': Tdev - pd.Timedelta(hours=L) for L in (48, 72, 96, 120)},
             'backmap, lags permuted': Tdev - pd.to_timedelta(tau_p)}
    M = {k: ace_rows(img, tq, Tdev) for k, tq in modes.items()}
    ok = np.logical_and.reduce([np.isfinite(m[single].to_numpy(float)).all(1) for m in M.values()]) & (bT >= 0)
    print(f'\nalignment comparison: {ok.sum()} common ACE hours')
    name = {c: k for k, c in {**A_FEATS, **H_FEATS}.items()}
    cmp = {}
    for k, m in M.items():
        m = m[ok]
        cmp[k] = pd.Series({name[c]: np.mean([np.corrcoef(m.loc[m.index[bT[ok] == b], c], m.y[bT[ok] == b])[0, 1]
                                              for b in range(5)]) for c in single})
    cmp = pd.DataFrame(cmp)
    for k in list(modes):
        cmp[f'rank {k}'] = cmp[k].abs().rank(ascending=False).astype(int)
    cmp = cmp.sort_values('rank backmap')
    cmp.to_csv(f'data/ranking_alignment_compare{SUF}.csv')
    print(cmp.to_string(float_format=lambda v: f'{v:+.3f}'))

    # conditional errors of single-feature OLS (out of fold), back-mapped vs fixed 96-h lag, binned by the observed
    # and by the predicted speed
    SEL = ['A: area f=0.30, w=20, off=0', 'B: ecliptic band |lat|<=15', 'A: area f=0.35, w=12.5, off=0',
           'D: mean log(I/med) of dark pixels, band |lat|<=30', 'B: hole-boundary distance, off=+10',
           'A: ESWF definition (f=0.35, w=7.5, off=0)']
    bins = np.r_[250:701:50, 900]
    rows = []
    for k in ('backmap', 'lag96'):
        m = M[k][ok].reset_index(drop=True); bb_ = bT[ok]
        for f in SEL + ['six selected (OLS)']:
            cols = [{**A_FEATS}[s] for s in SEL] if f.startswith('six') else [{**A_FEATS}[f]]
            pred = np.full(len(m), np.nan)
            for tr, va, te in folds.FOLDS.values():
                A_ = np.c_[np.ones(len(m)), m[cols].to_numpy(float)]
                w = np.linalg.lstsq(A_[np.isin(bb_, tr)], m.y[np.isin(bb_, tr)], rcond=None)[0]
                pred[bb_ == te] = A_[bb_ == te] @ w
            e = pred - m.y.to_numpy()
            for by, v_ in (('observed', m.y.to_numpy()), ('predicted', pred)):
                ib = np.digitize(v_, bins)
                for j in range(1, len(bins)):
                    s = ib == j
                    if s.sum() >= 50:
                        rows.append(dict(align=k, feature=f, by=by, lo=bins[j - 1], n=int(s.sum()),
                                         rmse=np.sqrt(np.mean(e[s] ** 2)), bias=e[s].mean()))
            rows.append(dict(align=k, feature=f, by='all', lo=np.nan, n=len(e), rmse=np.sqrt(np.mean(e ** 2)),
                             bias=e.mean()))
    ce = pd.DataFrame(rows)
    ce.to_csv(f'data/ranking_alignment_conditional{SUF}.csv', index=False)
    print('\nsingle-feature / six-feature OLS RMSE (out of fold), by observed speed:')
    print(ce[ce.by.isin(['observed', 'all'])].pivot_table(index=['feature', 'lo'], columns='align', values='rmse',
                                                          dropna=False).round(1).to_string())
