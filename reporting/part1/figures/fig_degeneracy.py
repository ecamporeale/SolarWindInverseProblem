"""Part I figure `fig_degeneracy` (manuscript label fig:degeneracy; Figure 5 in the 2026-10 draft): look-alike analysis of the degeneracy of the image inputs (development data).

(a) Share of fast-wind hours (ACE ≥ 550 km/s) whose 50 nearest look-alikes in input space (fold-consistent pool:
    other development batches only, |ΔT| ≥ 13 d) were followed by slow wind (≤ 400 km/s), as input groups are added;
    the control draws random hours.
(b) The same share when the outcome may be shifted in time by up to ± the tolerance.
Data (fixed 96-h lag, the paper's alignment): data/knn_degeneracy/other_batch_check_beta0.csv,
      data/knn_degeneracy/overnight/t1_timing_beta0.csv
      (pipeline/p1_model/knn_other_batch_check.py, overnight_t1_timing.py).

  python reporting/part1/figures/fig_degeneracy.py
Output: figures/part1/fig_degeneracy.png/.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, INK, save

_b = float(os.environ.get('KNN_ALIGN_BETA', '0'))           # paper: fixed 96-h lag (0); 1 = earlier ballistic matching
TAG = '' if _b == 1 else f'_beta{_b:g}'
O = pd.read_csv(f'data/knn_degeneracy/other_batch_check{TAG}.csv')
O = O[O.pool == 'other'].set_index('case')
T1 = pd.read_csv(f'data/knn_degeneracy/overnight/t1_timing{TAG}.csv')
ctrl0 = float(T1[(T1.case.str.startswith('random')) & (T1.window_h == 0)].fast_slow_share.iloc[0])
sets = [('193', 'image features at $t_{src}$'), ('193 lags', '+ same at 24, 48, 72 h earlier'), ('193 lags + fs', '+ PFSS expansion factor'),
        ('193 lags + fs + bm', '+ back-mapped recurrence (all inputs)')]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [1.3, 1]}, layout='constrained')
a = ax[0]
vals = [ctrl0] + [float(O.loc[k, 'fast_slow_share']) for k, _ in sets]
labs = ['random hours (control)'] + [l for _, l in sets]
y = np.arange(len(vals))[::-1]
a.barh(y, np.array(vals) * 100, height=0.6, color=[C['reference']] + [C['nn']] * len(sets))
for yi, v in zip(y, vals):
    a.text(v * 100 + 1, yi, f'{v * 100:.0f} %', va='center', fontsize=10.5, color=INK)
a.set_yticks(y); a.set_yticklabels(labs); a.grid(axis='y', visible=False); a.set_xlim(0, 58)
a.set_xlabel('look-alikes followed by slow wind (%)')
a.set_title('(a) Input groups added one at a time')
a = ax[1]
for case, lab, col, ls in (('random look-alikes (control)', 'random hours (control)', C['reference'], '--'),
                           ('193 single frame', 'image features at $t_{src}$', C['nn2'], '-'),
                           ('all inputs (193 lags + fs + bm)', 'all inputs', C['nn'], '-')):
    q = T1[T1.case == case].sort_values('window_h')
    a.plot(q.window_h, q.fast_slow_share * 100, 'o', ls=ls, color=col, lw=1.4, ms=5, label=lab)
a.set_xticks([0, 6, 12, 24]); a.set_xlabel('timing tolerance (± h)'); a.set_ylabel('(%)')
a.set_ylim(0, 55); a.legend(fontsize=10.5)
a.set_title('(b) Allowing timing errors')
save(fig, 'fig_degeneracy' if _b == 0 else f'fig_degeneracy_beta{_b:g}')
