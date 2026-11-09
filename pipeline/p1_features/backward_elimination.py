"""Backward elimination on linear regression (OLS) over the candidate image features of the
final ranking (data/final_feature_matrix.pkl: development folds, common rows).

Elimination order: OLS on all development rows (5 batches pooled); repeatedly drop the
feature with the smallest |t| until one is left. t-statistics are inflated by the strong
autocorrelation of hourly data, so they are used only to ORDER features, never as tests.
Scoring of every subset: out-of-sample OLS predictions (each batch predicted by the model
trained on the 3 training batches of the fold in which it is the test batch), pooled RMSE,
pooled Pearson r, and mean within-batch r. Block-bootstrap 95% CIs (Carrington rotations,
within batch, B = 2000) for the key subsets.
"""
import numpy as np
import pandas as pd

import folds

B = 2000
import os
import channel_cfg as CH
SUF = '' if os.environ.get('FEATURE_SOURCE', 'cache') == 'cache' else CH.SUF_PIXEL
X = pd.read_pickle(f'data/final_feature_matrix{SUF}.pkl')
rank = pd.read_csv(f'data/final_feature_ranking{SUF}.csv').set_index('feature')['rank']
y = X.y.to_numpy()
batch = X.batch.to_numpy()
REF = [c for c in X.columns if c.startswith('REF:')]
IMG = [c for c in rank.index if c in X.columns and not c.startswith('REF:')]
Z = (X[IMG + REF] - X[IMG + REF].mean()) / X[IMG + REF].std()

# Ridge latitude-profile composites rebuilt without test-batch information (nested):
#   Zt   (elimination order): each batch predicted by ridge fit on the other four batches;
#   Zf[te] (scoring the fold whose test batch is te): training rows from an inner
#          leave-one-batch-out within the 3 training batches, test rows from ridge fit on
#          all 3 training batches.
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if SUF:
    _a, _h = pd.read_pickle(CH.PIXEL_AIA), pd.read_pickle('data/pixel_features_hmi.pkl')
else:
    _a, _h = pd.read_pickle('data/aia_features_cache_fits_2011_2021.pkl'), pd.read_pickle('data/hmi_features_cache_hmi_dev.pkl')
_t = pd.to_datetime(X.obstime).dt.round('h')
_a['t'], _h['t'] = pd.to_datetime(_a.obstime).dt.round('h'), pd.to_datetime(_h.obstime).dt.round('h')
_a, _h = _a.drop_duplicates('t').set_index('t'), _h.drop_duplicates('t').set_index('t')
RAW = {'C: AIA dark-fraction latitude profile (ridge)': _a.loc[_t, [c for c in _a.columns if c.startswith('C_dark')]].to_numpy(float),
       'C: AIA log(I/median) latitude profile (ridge)': _a.loc[_t, [c for c in _a.columns if c.startswith('C_logI')]].to_numpy(float),
       'H: HMI latitude profiles, |B_r|, |net|, imbalance (ridge)': _h.loc[_t, [c for c in _h.columns if c.startswith('H3_')]].to_numpy(float)}
RAW = {k: v for k, v in RAW.items() if k in Z.columns}
assert all(np.isfinite(v).all() for v in RAW.values()), 'profile bins must be finite on the common rows'


def _ridge_pred(fit_mask, pred_mask, R):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(R[fit_mask], y[fit_mask])
    return m.predict(R[pred_mask])


def _std(v, ref_mask):
    return (v - v[ref_mask].mean()) / v[ref_mask].std()


Zt = Z.copy()
for name, R in RAW.items():
    comp = np.full(len(y), np.nan)
    for bb in range(5):
        comp[batch == bb] = _ridge_pred(batch != bb, batch == bb, R)
    Zt[name] = _std(comp, np.ones(len(y), bool))
Zf = {}
for tr, va, te in folds.FOLDS.values():
    Zte = Z.copy()
    mtr, mte = np.isin(batch, tr), batch == te
    for name, R in RAW.items():
        comp = np.full(len(y), np.nan)
        for bb in tr:
            comp[batch == bb] = _ridge_pred(np.isin(batch, [k for k in tr if k != bb]), batch == bb, R)
        comp[mte] = _ridge_pred(mtr, mte, R)
        Zte[name] = _std(comp, mtr)
    Zf[te] = Zte
print(f'nested profile composites built for: {list(RAW)}', flush=True)


def tstats(cols):
    Z = Zt
    A = np.column_stack([np.ones(len(Z)), Z[cols].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    s2 = res @ res / (len(y) - A.shape[1])
    cov = s2 * np.linalg.pinv(A.T @ A)
    return pd.Series(beta[1:] / np.sqrt(np.diag(cov)[1:]), index=cols), beta[1:]


def oos(cols):
    p = np.full(len(y), np.nan)
    for tr, va, te in folds.FOLDS.values():
        mtr, mte = np.isin(batch, tr), batch == te
        Z = Zf[te]
        if cols:
            A = np.column_stack([np.ones(mtr.sum()), Z.loc[mtr, cols].to_numpy()])
            beta, *_ = np.linalg.lstsq(A, y[mtr], rcond=None)
            p[mte] = np.column_stack([np.ones(mte.sum()), Z.loc[mte, cols].to_numpy()]) @ beta
        else:
            p[mte] = y[mtr].mean()
    return p


def scores(p):
    within = np.mean([np.corrcoef(p[batch == b], y[batch == b])[0, 1] for b in range(5)])
    return dict(rmse=np.sqrt(np.mean((p - y) ** 2)), r=np.corrcoef(p, y)[0, 1], mean_within_r=within)


def eliminate(pool):
    cols, steps = list(pool), []
    while cols:
        t, _ = tstats(cols)
        sc = scores(oos(cols))
        weakest = t.abs().idxmin()
        steps.append(dict(n_features=len(cols), **sc, removed_next=weakest,
                          removed_rank=rank.get(weakest, np.nan), removed_abs_t=abs(t[weakest])))
        cols.remove(weakest)
    return pd.DataFrame(steps)


clim = scores(oos([]))
print(f'rows {len(y)}; climatology out-of-sample RMSE {clim["rmse"]:.1f} km/s')

PERS = 'REF: persistence v(t)'
POOL = IMG + [PERS]   # all 33 table rows except 27-day recurrence
steps = eliminate(POOL)
steps.to_csv(f'data/backward_elimination_all_but_recurrence{SUF}.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 58)
print(f'\n=== Backward elimination, {len(POOL)} features = all table rows except 27-day recurrence (row = model with n features; removed_next = dropped to get the next row) ===')
print(steps.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

steps_ref = eliminate(POOL + ['REF: 27-day recurrence v(t+4d-27d)'])
rpos = steps_ref.index[steps_ref.removed_next == 'REF: 27-day recurrence v(t+4d-27d)'][0]
print(f'\n=== Side variant: all table rows (incl. 27-day recurrence) ===')
print(f'recurrence is eliminated when {steps_ref.n_features[rpos]} features remain '
      f'(i.e. it survives {len(steps_ref) - steps_ref.n_features[rpos]} eliminations)')
print(steps_ref[steps_ref.n_features <= 10].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
steps_ref.to_csv(f'data/backward_elimination_with_recurrence{SUF}.csv', index=False)

# ---- bootstrap CIs for key subsets
best_n = int(steps.loc[steps.rmse.idxmin(), 'n_features'])
order = list(steps.removed_next)
subset_of = lambda n: order[len(order) - n:]
top1 = [rank.index[rank == 1][0]]
TOP5 = subset_of(5)
KEY = {'elimination top 5': TOP5, f'all {len(POOL)} features (no recurrence)': POOL,
       f'best by RMSE ({best_n} features)': subset_of(best_n), 'rank 1 alone': top1,
       'elimination top 5 + 27-day recurrence': TOP5 + ['REF: 27-day recurrence v(t+4d-27d)'],
       'climatology': []}
P = {k: oos(v) for k, v in KEY.items()}
blocks, binv = np.unique(X.block.to_numpy(), return_inverse=True)
bb = (blocks // 1000).astype(int)
nb = len(blocks)
bs = lambda v: np.bincount(binv, weights=v, minlength=nb)
S = {'n': bs(np.ones(len(y))), 'y': bs(y), 'yy': bs(y * y)}
for k, p in P.items():
    S[k] = (bs((p - y) ** 2), bs(p), bs(p * p), bs(p * y))
rng = np.random.default_rng(0)
W = np.zeros((nb, B))
for b in range(5):
    idx = np.where(bb == b)[0]
    dr = rng.choice(idx, size=(len(idx), B))
    for r in range(B):
        W[:, r] += np.bincount(dr[:, r], minlength=nb)


def boot(k, w):
    se, sp, spp, spy = (v @ w for v in S[k])
    n, sy, syy = S['n'] @ w, S['y'] @ w, S['yy'] @ w
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (n * spy - sp * sy) / np.sqrt((n * spp - sp ** 2) * (n * syy - sy ** 2))
    return np.sqrt(se / n), r


one = np.ones(nb)
rows = []
ref = 'elimination top 5'
rr_ref, cc_ref = boot(ref, W)
for k in KEY:
    rm, cc = boot(k, one)
    rmb, ccb = boot(k, W)
    drm, dcc = rmb - rr_ref, ccb - cc_ref
    ci = lambda v: np.nanpercentile(v, [2.5, 97.5])
    rows.append(dict(model=k, n_features=len(KEY[k]),
                     rmse=f'{rm:.1f} [{ci(rmb)[0]:.1f}, {ci(rmb)[1]:.1f}]',
                     r='—' if k == 'climatology' else f'{cc:.3f} [{ci(ccb)[0]:.3f}, {ci(ccb)[1]:.3f}]',
                     d_rmse_vs_top5='—' if k == ref else f'{rm - boot(ref, one)[0]:+.1f} [{ci(drm)[0]:+.1f}, {ci(drm)[1]:+.1f}]',
                     d_r_vs_top5='—' if k in (ref, 'climatology') else f'{cc - boot(ref, one)[1]:+.3f} [{ci(dcc)[0]:+.3f}, {ci(dcc)[1]:+.3f}]'))
key = pd.DataFrame(rows)
print('\n=== Key models: out-of-sample pooled RMSE and r [95% block-bootstrap CI]; differences vs elimination top 5 ===')
print(key.to_string(index=False))
key.to_csv(f'data/backward_elimination_key_models{SUF}.csv', index=False)
