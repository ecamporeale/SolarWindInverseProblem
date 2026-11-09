"""Backward elimination for high-speed-stream DETECTION (logistic regression), pixel level,
compared with the regression (OLS/RMSE) elimination of backward_elimination.py.

Event: ACE v(t+4d) > threshold (500 and 600 km/s). Same 41 candidates, same common rows,
same leak-free latitude-profile composites (rebuilt fold by fold) as the regression run.
Order: logistic fit (IRLS) on all development rows; drop the smallest |Wald z| until one is
left (z inflated by autocorrelation -> used only to order). Out-of-sample scores per step
(5-fold test protocol): ROC AUC, Brier skill score vs the training base rate, Heidke skill
score (HSS) with the probability threshold chosen on the training batches (max HSS there).
Outputs: data/logistic_elimination_v{500,600}_pixel.csv, data/elimination_order_comparison_pixel.csv,
data/logistic_key_models_pixel.csv, figures/logistic_elimination_{hss,auc}_pixel.png
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import folds

THRESH = [500.0, 600.0]
B = 2000
import channel_cfg as CH
X = pd.read_pickle(f'data/final_feature_matrix{CH.SUF_PIXEL}.pkl')
rank = pd.read_csv(f'data/final_feature_ranking{CH.SUF_PIXEL}.csv').set_index('feature')['rank']
y = X.y.to_numpy()
batch = X.batch.to_numpy()
IMG = [c for c in rank.index if c in X.columns and not c.startswith('REF:')]
POOL = IMG + ['REF: persistence v(t)']
Z = (X[POOL] - X[POOL].mean()) / X[POOL].std()

# ---- leak-free profile composites (as in backward_elimination.py), regression target y
_a, _h = pd.read_pickle(CH.PIXEL_AIA), pd.read_pickle('data/pixel_features_hmi.pkl')
_t = pd.to_datetime(X.obstime).dt.round('h')
_a['t'], _h['t'] = pd.to_datetime(_a.obstime).dt.round('h'), pd.to_datetime(_h.obstime).dt.round('h')
_a, _h = _a.drop_duplicates('t').set_index('t'), _h.drop_duplicates('t').set_index('t')
RAW = {'C: AIA dark-fraction latitude profile (ridge)': _a.loc[_t, [c for c in _a.columns if c.startswith('C_dark')]].to_numpy(float),
       'C: AIA log(I/median) latitude profile (ridge)': _a.loc[_t, [c for c in _a.columns if c.startswith('C_logI')]].to_numpy(float),
       'H: HMI latitude profiles, |B_r|, |net|, imbalance (ridge)': _h.loc[_t, [c for c in _h.columns if c.startswith('H3_')]].to_numpy(float)}
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
std = lambda v, m: (v - v[m].mean()) / v[m].std()
Zt = Z.copy()
for name, R in RAW.items():
    c = np.full(len(y), np.nan)
    for k in range(5):
        c[batch == k] = ridge().fit(R[batch != k], y[batch != k]).predict(R[batch == k])
    Zt[name] = std(c, np.ones(len(y), bool))
Zf = {}
for tr, va, te in folds.FOLDS.values():
    Zte, mtr, mte = Z.copy(), np.isin(batch, tr), batch == te
    for name, R in RAW.items():
        c = np.full(len(y), np.nan)
        for k in tr:
            fit = np.isin(batch, [j for j in tr if j != k])
            c[batch == k] = ridge().fit(R[fit], y[fit]).predict(R[batch == k])
        c[mte] = ridge().fit(R[mtr], y[mtr]).predict(R[mte])
        Zte[name] = std(c, mtr)
    Zf[te] = Zte
print('profile composites rebuilt fold by fold', flush=True)


def logit_fit(A, e, iters=100):
    beta = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-8 * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (e - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta, H


def zstats(cols, e):
    A = np.column_stack([np.ones(len(y)), Zt[cols].to_numpy()])
    beta, H = logit_fit(A, e)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return pd.Series(beta[1:] / se[1:], index=cols)


def hss_counts(pred, obs):
    a, b_, c, d = (pred & obs).sum(), (~pred & obs).sum(), (pred & ~obs).sum(), (~pred & ~obs).sum()
    return a, b_, c, d


def hss(a, b_, c, d):
    den = (a + b_) * (b_ + d) + (a + c) * (c + d)
    return 2 * (a * d - b_ * c) / den if den > 0 else np.nan


def oos(cols, e):
    """out-of-sample probability and binary forecast (threshold tuned on training batches)."""
    prob, fc, brier_ref = np.full(len(y), np.nan), np.zeros(len(y), bool), np.full(len(y), np.nan)
    for tr, va, te in folds.FOLDS.values():
        mtr, mte = np.isin(batch, tr), batch == te
        ZZ = Zf[te]
        Atr = np.column_stack([np.ones(mtr.sum()), ZZ.loc[mtr, cols].to_numpy()]) if cols else np.ones((mtr.sum(), 1))
        Ate = np.column_stack([np.ones(mte.sum()), ZZ.loc[mte, cols].to_numpy()]) if cols else np.ones((mte.sum(), 1))
        beta, _ = logit_fit(Atr, e[mtr])
        ptr = 1 / (1 + np.exp(-np.clip(Atr @ beta, -30, 30)))
        prob[mte] = 1 / (1 + np.exp(-np.clip(Ate @ beta, -30, 30)))
        grid = np.unique(np.quantile(ptr, np.linspace(0.5, 0.995, 100)))
        best = max(grid, key=lambda th: hss(*hss_counts(ptr >= th, e[mtr].astype(bool))))
        fc[mte] = prob[mte] >= best
        brier_ref[mte] = e[mtr].mean()
    return prob, fc, brier_ref


def score(cols, e):
    prob, fc, ref = oos(cols, e)
    bs, bs0 = np.mean((prob - e) ** 2), np.mean((ref - e) ** 2)
    return dict(auc=roc_auc_score(e, prob) if cols else 0.5, bss=1 - bs / bs0, hss=hss(*hss_counts(fc, e.astype(bool))))


def eliminate(pool, e):
    cols, steps = list(pool), []
    while cols:
        z = zstats(cols, e)
        weakest = z.abs().idxmin()
        steps.append(dict(n_features=len(cols), **score(cols, e), removed_next=weakest,
                          removed_rank=rank.get(weakest, np.nan), removed_abs_z=abs(z[weakest])))
        cols.remove(weakest)
        print(f'  v>{int(e_thr)}: {len(cols)+1} features done', flush=True) if len(cols) % 10 == 0 else None
    return pd.DataFrame(steps)


paths = {}
for e_thr in THRESH:
    e = (y > e_thr).astype(float)
    print(f'v>{int(e_thr)}: event rate {e.mean()*100:.1f}%', flush=True)
    st = eliminate(POOL, e)
    st.to_csv(f'data/logistic_elimination_v{int(e_thr)}{CH.SUF_PIXEL}.csv', index=False)
    paths[int(e_thr)] = st
reg = pd.read_csv(f'data/backward_elimination_all_but_recurrence{CH.SUF_PIXEL}.csv')
paths['ols'] = reg

# survival position: number of features remaining when the feature is dropped (1 = last survivor)
pos = pd.DataFrame({k: st.set_index('removed_next')['n_features'] for k, st in paths.items()})
pos.columns = ['logistic v>500', 'logistic v>600', 'OLS (RMSE)']
pos['Table 1 rank'] = [rank[f] for f in pos.index]
pos = pos.sort_values('OLS (RMSE)')
pos.to_csv(f'data/elimination_order_comparison{CH.SUF_PIXEL}.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print('\n=== survival position (1 = last survivor), 15 most important by OLS order ===')
print(pos.head(15).to_string())
from scipy.stats import spearmanr
for a, b_ in [('OLS (RMSE)', 'logistic v>500'), ('OLS (RMSE)', 'logistic v>600'), ('logistic v>500', 'logistic v>600')]:
    print(f'Spearman rank correlation of survival positions, {a} vs {b_}: {spearmanr(pos[a], pos[b_])[0]:.2f}')
for k, st in paths.items():
    if k == 'ols':
        continue
    print(f'\n=== logistic v>{k}: path (n <= 12) ===')
    print(st[st.n_features <= 12].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ---- key models: HSS / AUC with block-bootstrap CIs
top5 = lambda st: st.sort_values('n_features').removed_next.tolist()[:5]
SETS = {'OLS top 5': top5(reg), 'logistic v>500 top 5': top5(paths[500]), 'logistic v>600 top 5': top5(paths[600]),
        'OLS top 5 + HMI PIL distance': top5(reg) + ['H: PIL distance, sigma=20, off=-10'],
        'rank 1 alone': [rank.index[rank == 1][0]]}
ub, inv = np.unique(X.block.to_numpy(), return_inverse=True)
bb = ub // 1000
rng = np.random.default_rng(0)
W = np.array([np.bincount(np.concatenate([rng.choice(np.where(bb == k)[0], (bb == k).sum()) for k in range(5)]),
                          minlength=len(ub)) for _ in range(B)], float)   # [B, n_blocks]
rows = []
for e_thr in THRESH:
    e = (y > e_thr)
    fcs = {k: oos(v, e.astype(float))[1] for k, v in SETS.items()}
    cnt = {k: np.stack([np.bincount(inv, weights=w.astype(float), minlength=len(ub)) for w in
                        (f & e, ~f & e, f & ~e, ~f & ~e)]) for k, f in fcs.items()}   # [4, n_blocks]
    bh = {k: np.array([hss(*(c @ wr)) for wr in W]) for k, c in cnt.items()}
    for k in SETS:
        h = hss(*cnt[k].sum(1))
        dd = bh[k] - bh['OLS top 5']
        rows.append(dict(threshold=int(e_thr), feature_set=k, features='; '.join(SETS[k]), hss=h,
                         hss_lo=np.percentile(bh[k], 2.5), hss_hi=np.percentile(bh[k], 97.5),
                         d_vs_ols_top5=h - hss(*cnt['OLS top 5'].sum(1)), d_lo=np.percentile(dd, 2.5), d_hi=np.percentile(dd, 97.5)))
key = pd.DataFrame(rows)
key.to_csv(f'data/logistic_key_models{CH.SUF_PIXEL}.csv', index=False)
print('\n=== event skill of competing feature sets (out of sample, HSS [95% CI], difference vs OLS top 5) ===')
print(key.drop(columns='features').to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
for k, v in SETS.items():
    print(f'  {k}: {v}')

# ---- plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
COL = {500: '#2a78d6', 600: '#eb6834'}
for metric, lab, fname in [('hss', 'Out-of-sample Heidke skill score', 'logistic_elimination_hss_pixel.png'),
                           ('auc', 'Out-of-sample ROC AUC', 'logistic_elimination_auc_pixel.png')]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for k in (500, 600):
        st = paths[k].sort_values('n_features')
        ax.plot(st.n_features, st[metric], color=COL[k], lw=2, marker='o', ms=4.5, mec=SURF, mew=1, label=f'event v > {k} km/s', zorder=3)
        bi = st[metric].idxmax()
        ax.plot(st.n_features[bi], st[metric][bi], 'o', ms=8, mfc='none', mec=COL[k], mew=1.6, zorder=4)
        ax.annotate(f'best {st[metric][bi]:.3f} at n={int(st.n_features[bi])}', (st.n_features[bi], st[metric][bi]),
                    xytext=(6, 8), textcoords='offset points', fontsize=8, color=INK2)
        ax.annotate(f'v > {k}', (st.n_features.iloc[-1], st[metric].iloc[-1]), xytext=(4, 0), textcoords='offset points', fontsize=8, color=INK, va='center')
    ax.set_xlabel('Number of features kept', color=INK, fontsize=9)
    ax.set_ylabel(lab, color=INK, fontsize=9)
    ax.set_xlim(0, len(POOL) + 5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xticks([1] + list(range(5, len(POOL) + 1, 5)))
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
    ax.set_title(f'{lab.replace("Out-of-sample ", "")} vs number of features (logistic backward elimination)', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(f'figures/{fname}', facecolor=SURF)
print('plots saved')
