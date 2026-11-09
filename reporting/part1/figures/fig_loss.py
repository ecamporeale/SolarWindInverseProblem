"""Part I figure `fig_loss` (manuscript label fig:hss; Figure 8 in the 2026-10 draft): what the loss buys. B0 trained with the S loss vs the same model trained with the MSE loss, and
the in-house baselines, all on the matched development test hours (pooled over the 5 test batches; NN = mean of the
seed-42 and seed-43 predictions; the headline convention of the manuscript, as Table 2).

(a) RMSE vs TRMSE = sqrt(TIMSE_N), pooled (large markers, legend: pooled S); small markers = the five test batches.
    Grey curves: constant S = sqrt(RMSE · TRMSE).
(b) TSS for fast wind (≥ 500 and ≥ 600 km/s) at the fixed thresholds.
(c) ROC AUC for the same events (threshold-free).
Data: <B0_RUN>/seed{42,43} and <B0_RUN>_mse/seed{42,43} test predictions (via _b0_hourly), linear and recurrence
baselines of eval_matched_baselines.

  python reporting/part1/figures/fig_loss.py
Output: figures/part1/fig_loss.png/.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

from _b0_hourly import b0_hourly
from _style import C, INK2, LBL, save
from eval_matched_baselines import metrics

H = b0_hourly().dropna(subset=['linear', 'recurrence', 'nn_mse'])
o = H.obs.to_numpy()
METH = [('nn', 'Reference model, S loss', C['nn'], 'o'), ('nn_mse', 'Reference model, MSE loss', C['nn2'], 's'),
        ('linear', LBL['linear'], C['linear'], 'D'), ('recurrence', LBL['recurrence'], C['recurrence'], 'D'),
        ('climatology', LBL['climatology'], C['climatology'], 'D')]
P = {m: metrics(o, H[m].to_numpy()) for m, *_ in METH}
for m, *_ in METH:
    print(f'{m:12s} S {P[m]["S"]:.2f} RMSE {P[m]["rmse"]:.2f} TRMSE {P[m]["trmse"]:.2f} '
          f'TSS500 {P[m]["tss500"]:.3f} TSS600 {P[m]["tss600"]:.3f}')

fig, a = plt.subplots(figsize=(7.0, 5.2), layout='constrained')                 # one panel (2026-10-25): RMSE vs tail error
xr = np.linspace(55, 100, 200)
for s in (90, 100, 110, 120, 130, 140):
    a.plot(xr, s ** 2 / xr, color='#c9c8c3', lw=0.8, zorder=0)
    a.text(98.5, s ** 2 / 98.5, f'S = {s}', fontsize=10, color=INK2, va='bottom', ha='right')
for m, lab, col, mk in METH:
    for f, g in H.groupby('fold'):
        q = metrics(g.obs.to_numpy(), g[m].to_numpy())
        a.plot(q['rmse'], q['trmse'], mk, color=col, ms=4, alpha=0.35)
    a.plot(P[m]['rmse'], P[m]['trmse'], mk, color=col, ms=10, mew=1.6, label=f'{lab}: S = {P[m]["S"]:.1f}')
a.annotate('', xy=(P['nn_mse']['rmse'], P['nn_mse']['trmse']), xytext=(P['nn']['rmse'], P['nn']['trmse']),
           arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0))
a.set_xlim(55, 100); a.set_ylim(70, 220)
a.set_xlabel('RMSE (km/s)'); a.set_ylabel('√TIMSE$_N$ (km/s)')
a.legend(fontsize=10.5, loc='upper left')
save(fig, 'fig_loss')
