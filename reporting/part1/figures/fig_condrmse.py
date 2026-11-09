"""Conditional errors of the Part I models (decision of the author, 2026-10-22: RMSE is shown in bins for every model,
and the conditional-RMSE curve whose area is TIMSE is shown as a figure).

Matched development test hours (46,412; 2-seed means, out of fold, as Table 2 and lasso_refit_scores.py).
(a) Tail-conditional RMSE: RMSE over the hours with observed speed ≥ v, against v, on the percentile grid of the
    observations (K = 100; train_joint_forecast.percentile_grid). TIMSE_N is the area under the squared curve divided by
    the speed range, TIMSE_N = Σ_k (q_{k+1} − q_k) MSE(o ≥ q_k) / (v_max − v_min), and TRMSE = √TIMSE_N; the script checks
    this against the TRMSE of eval_matched_baselines.metrics. Grey: number of hours above v.
(b) RMSE in bins of the observed speed (25 km/s; bins with < 100 hours not shown).
(c) RMSE in bins of each model's own predicted speed (how reliable a forecast of a given speed is).
Models: B0 (S loss), B0 with MSE loss, the 37-input linear map trained through the solver (F_lin), the 4-coefficient
linear model (Q_lin), the 4-curve NAM (Q_nam0.3), the linear comparator (6 AIA + 2 HMI, fixed 4-d lag), 27-day
recurrence.

  python reporting/part1/figures/fig_condrmse.py [--set models|factorial]
--set factorial (2026-10-22): the factorial test of the link instead — B0 (solver), network + forward ballistic link,
network + fixed lag, linear map + solver, linear map + fixed lag — with per-bin ΔRMSE vs B0 (95 % 27-day block
bootstrap, bins of the observed speed) in the CSV.
Output: figures/part1/conditional_rmse[_factorial].png/.pdf, runs/sweep5_lasso/conditional_rmse[_factorial].csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, GRID, INK, INK2, MUTED, save
from b0_skill_metrics import two_seed
from eval_matched_baselines import ace_hourly, metrics
from train_joint_forecast import percentile_grid

import argparse as _ap
_a = _ap.ArgumentParser(); _a.add_argument('--set', choices=['models', 'factorial'], default='models'); ARGS = _a.parse_args()
FACTORIAL = [('B0', 'runs/sweep4/B0', 'reference: network + solver', C['nn'], '-'),
             ('F_ball', 'runs/sweep4/F_ball', 'network + ballistic link', C['nn2'], '-'),
             ('F_lag', 'runs/sweep4/F_lag', 'network + fixed 96-h lag', C['recurrence'], '-'),
             ('F_lin', 'runs/sweep4/F_lin', 'linear map + solver', C['eswf'], '--'),
             ('F_lin_lag', 'runs/sweep4/F_lin_lag', 'linear map + fixed lag', MUTED, '--')]
MODELS = [('B0', 'runs/sweep4/B0', 'reference model (S loss)', C['nn'], '-'),
          ('B0_mse', 'runs/sweep4/B0_mse', 'reference model, MSE loss', C['nn2'], '-'),
          ('F_lin', 'runs/sweep4/F_lin', 'linear map, 37 inputs', C['eswf'], '-'),
          ('Q_lin', 'runs/sweep5_lasso/Q_lin', '4-coefficient linear', INK, '--'),
          ('Q_nam0.3', 'runs/sweep5_lasso/Q_nam0.3', '4-curve NAM', C['climatology'], '--')]
BASE = [('linear', 'linear comparator (6 AIA + 2 HMI)', C['linear'], ':'), ('recurrence', '27-day recurrence', C['recurrence'], ':'),
        ('wsa_huxt', 'WSA 4.5 + HUXt', C['persistence'], '-.')]          # physics-based reference (2026-10-25; part1_wsa_compare.py)
if ARGS.set == 'factorial':                                # the factorial test of the link (worklog §1, §14)
    MODELS, BASE = FACTORIAL, [('linear', 'linear comparator (6 AIA + 2 HMI)', C['linear'], '-')]
    MODELS = [m[:4] + ('-',) for m in MODELS]              # solid lines only; the colours tell the versions apart (2026-10-26)
TAG = '' if ARGS.set == 'models' else '_factorial'

ace = ace_hourly()
D = None
for key, path, *_ in MODELS:
    T = two_seed(path, ace)
    cols = ['time', 'nn'] if D is not None else ['time', 'obs', 'linear', 'recurrence', 'nn']
    T = T[cols].rename(columns={'nn': key})
    D = T if D is None else D.merge(T, on='time', how='inner')
D = D.dropna()
if any(k == 'wsa_huxt' for k, *_ in BASE):
    n0 = len(D)
    W = pd.read_csv('runs/part1_wsa/matched.csv', parse_dates=['time'])[['time', 'wsa_huxt']]
    D = D.merge(W, on='time', how='inner')
    assert len(D) == n0, f'WSA + HUXt changes the hour set ({n0} -> {len(D)})'
o = D.obs.to_numpy()
g = percentile_grid(o)
q = g['q'].numpy()
ALL = [(k, lab, col, ls) for k, _, lab, col, ls in MODELS] + BASE

rows = []
plt.rcParams.update({'font.size': 24, 'axes.labelsize': 24, 'axes.titlesize': 25, 'xtick.labelsize': 21,
                     'ytick.labelsize': 21, 'legend.fontsize': 20})   # printed at text width: ~11 pt (2026-10-26)
fig, ax = plt.subplots(1, 3, figsize=(18, 7.6), layout='constrained')
cnt = np.array([(o >= v).sum() for v in q[:-1]])
for key, lab, col, ls in ALL:
    e2 = (D[key].to_numpy() - o) ** 2
    tail = np.array([e2[o >= v].mean() for v in q[:-1]])
    timse = np.sum(np.diff(q) * tail) / (g['vmax'] - g['vmin'])
    m = metrics(o, D[key].to_numpy())
    assert abs(np.sqrt(timse) - m['trmse']) < 0.05, (key, np.sqrt(timse), m['trmse'])
    ax[0].step(q[:-1], np.sqrt(tail), where='post', color=col, ls=ls, lw=2.4, label=f'{lab} (√TIMSE {np.sqrt(timse):.0f})')
    for cond, x, a in (('observed', o, ax[1]), ('predicted', D[key].to_numpy(), ax[2])):
        edges = np.arange(225, 900, 25)
        b = np.digitize(x, edges)
        mids, r = [], []
        for i in np.unique(b):
            sel = b == i
            if sel.sum() < 100 or i == 0 or i == len(edges):
                continue
            mids.append(edges[i - 1] + 12.5); r.append(np.sqrt(e2[sel].mean()))
            rows.append(dict(model=key, conditioned_on=cond, bin_lo=edges[i - 1], bin_hi=edges[i], n=int(sel.sum()),
                             rmse=r[-1], bias=float(np.mean(D[key].to_numpy()[sel] - o[sel]))))
        a.plot(mids, r, color=col, ls=ls, lw=2.4, marker='o', ms=5)
    rows.append(dict(model=key, conditioned_on='tail (o ≥ v), TRMSE', bin_lo=np.nan, bin_hi=np.nan, n=len(o),
                     rmse=np.sqrt(timse), bias=np.nan))
a2 = ax[0].twinx()
a2.fill_between(q[:-1], cnt, step='post', color=GRID, zorder=0)
a2.set_ylabel('hours ≥ v', color=MUTED); a2.tick_params(colors=MUTED)
ax[0].set_zorder(1); ax[0].patch.set_visible(False)
ax[0].set_xlabel('threshold v (km/s)'); ax[0].set_ylabel('RMSE, observed ≥ v (km/s)')
ax[0].set_title('(a) Tail-conditional RMSE')
ax[1].set_xlabel('observed speed (km/s)'); ax[1].set_ylabel('RMSE in bin (km/s)')
ax[1].set_title('(b) By observed speed')
ax[2].set_xlabel('predicted speed (km/s)'); ax[2].set_ylabel('RMSE in bin (km/s)')
ax[2].set_title('(c) By predicted speed')
fig.legend(*ax[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, handlelength=2.2, columnspacing=1.4)
for a in ax[1:]:
    a.set_ylim(0, None)
save(fig, 'fig_condrmse' if not TAG else 'fig_condrmse_links')
R = pd.DataFrame(rows)
# per-bin ΔRMSE vs B0 in bins of the observed speed, 95 % 27-day block bootstrap (same hours for both models)
t_ = pd.to_datetime(D.time); blk = ((t_ - t_.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
rng = np.random.default_rng(0); edges = np.arange(225, 900, 25); bo = np.digitize(o, edges)
e0 = D['B0'].to_numpy() - o
for key, lab, col, ls in ALL:
    if key == 'B0':
        continue
    ek = D[key].to_numpy() - o
    for i in np.unique(bo):
        sel = np.flatnonzero(bo == i)
        if len(sel) < 100 or i == 0 or i == len(edges):
            continue
        ub = np.unique(blk[sel]); grp = {u: sel[blk[sel] == u] for u in ub}
        d = [np.sqrt(np.mean(ek[s_] ** 2)) - np.sqrt(np.mean(e0[s_] ** 2))
             for s_ in (np.concatenate([grp[u] for u in rng.choice(ub, len(ub))]) for _ in range(500))]
        m = (R.model == key) & (R.conditioned_on == 'observed') & (R.bin_lo == edges[i - 1])
        R.loc[m, 'dRMSE_vs_B0'] = np.sqrt(np.mean(ek[sel] ** 2)) - np.sqrt(np.mean(e0[sel] ** 2))
        R.loc[m, 'lo95'], R.loc[m, 'hi95'] = np.percentile(d, [2.5, 97.5])
R.to_csv(f'runs/sweep5_lasso/conditional_rmse{TAG}.csv', index=False)
print(R[R.conditioned_on.str.startswith('tail')][['model', 'rmse']].round(1).to_string(index=False))
