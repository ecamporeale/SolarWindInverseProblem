"""Curves of the 4-quantity NAM (Q_nam0.3, Q_nam3; 2026-10-21; worklog INTERPRETABLE_BOUNDARY.md §9): one input per
quantity (mean over the current and available lagged images; boundary.ReducedInputsHead), one curve each, curvature
penalty 0.3 or 3. Is any curve nonlinear, reproducibly across folds?

For each of the 10 models (5 folds × 2 seeds) of each penalty, the full effect of quantity k is its linear term plus
its curve, in km/s at a boundary speed of 400 km/s (154 km/s per logit unit), centred over the data, against the
quantity in physical units (each model's own feature statistics). Thin lines: models (colour: penalty); dashed black:
the 4-coefficient linear model Q_lin (mean of its 10 models). Grey: distribution of the quantity over the development
test cells (nn_vs_linear/cells.npz, reduced). Panel title: per penalty, the nonlinear part (RMS over the data of the
effect minus its best straight line, mean over models) and its reproducibility across folds (mean correlation of the
nonlinear parts of models from different folds, evaluated on the same cells).

  python reporting/part1/figures/fig_qnam.py
Output: figures/part1/fig_qnam.png/.pdf, runs/sweep5_lasso/q_nam_curves.csv
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _style import C, GRID, INK, save
from boundary import GROUP_NAMES, ReducedInputsHead

K = 650 * (250 / 650) * (1 - 250 / 650)
LABEL = ['coronal-hole area, f 0.30, ±20° strip', 'dark fraction, |β| ≤ 15°', 'dark-pixel mean log₁₀(I/I_med)',
         'back-mapped recurrence (km/s)']
PHYS = [('feat', 0), ('feat', 1), ('feat', 3), ('extra', 1)]           # (statistics block, index) per quantity
cells = dict(np.load('runs/sweep4/nn_vs_linear/cells.npz'))


def load(run, seed, fold):
    sd = torch.load(f'{run}/seed{seed}/fold{fold}/best.pt', map_location='cpu', weights_only=False)['nn1_state_dict']
    groups = [GROUP_NAMES[i] for i in sd['head.red_groups'].tolist()]
    nam = 'head.inner.nam_key' in sd
    h = ReducedInputsHead(groups, 'nam' if nam else 'linear',
                          tuple(sd['head.inner.nam_hidden'].tolist()) if nam else (16, 8))
    h.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
    h.eval()
    st = {'feat': (sd['feat_mean'].flatten().numpy(), sd['feat_std'].flatten().numpy()),
          'extra': (sd['extra_mean'].flatten().numpy(), sd['extra_std'].flatten().numpy())}
    with torch.no_grad():
        xr = h.reduce(torch.tensor(cells['x'][cells['fold'] == fold], dtype=torch.float32)).numpy()
    return dict(h=h, nam=nam, st=st, xr=xr, fold=fold, seed=seed)


def effect(M, k, xs):
    """Full effect (logit) of quantity k at reduced, standardized values xs."""
    h = M['h'].inner
    with torch.no_grad():
        if not M['nam']:
            return h[0].weight[0, k].item() * xs
        z = torch.tensor(xs, dtype=torch.float32)
        cur = h._run(h.shape_nets, z.view(-1, 1, 1).expand(-1, len(h.nam_key), 1))[:, k].numpy()
        return h.lin.weight[0, k].item() * xs + cur


def phys(M, k, xs):
    blk, i = PHYS[k]
    mu, sd = M['st'][blk]
    return xs * sd[i] + mu[i]


SETS = {'Q_nam0.3': C['nn'], 'Q_nam3': C['nn2'], 'Q_lin': INK}
MODELS = {r: [load(f'runs/sweep5_lasso/{r}', s, f) for s in (42, 43) for f in range(1, 6)] for r in SETS}
plt.rcParams.update({'font.size': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15})  # ~8-9 pt in print (2026-10-26)
fig, axs = plt.subplots(1, 4, figsize=(19, 6.2), layout='constrained')
rows = []
for k, a in enumerate(axs):
    allx = np.concatenate([M['xr'][:, k] for M in MODELS['Q_lin']])
    lo, hi = np.quantile(allx, [0.01, 0.99])
    grid = np.linspace(lo, hi, 200)
    title = []
    for r, col in SETS.items():
        curves, nl = [], []
        for M in MODELS[r]:
            xd = M['xr'][:, k]
            e = effect(M, k, xd) * K
            b = np.polyfit(xd, e, 1)
            nl.append(np.sqrt(np.mean((e - np.polyval(b, xd)) ** 2)))
            eg = effect(M, k, grid) * K - e.mean()
            curves.append(eg)
            if r != 'Q_lin':
                a.plot(phys(M, k, grid), eg, color=col, lw=0.6, alpha=0.4)
        xm = phys(MODELS[r][0], k, grid)
        if r == 'Q_lin':
            a.plot(xm, np.mean(curves, 0), color=INK, lw=1.4, ls='--', label='4-coefficient linear model')
            continue
        a.plot(xm, np.mean(curves, 0), color=col, lw=2.2, label=f'NAM, curvature penalty {r[5:]}')
        # cross-fold reproducibility of the nonlinear parts, on common cells (each fold's cells, other folds' models)
        cors = []
        for M1, M2 in itertools.combinations([M for M in MODELS[r] if M['seed'] == 42], 2):
            xd = M1['xr'][:, k]                                   # M1's own fold cells, in M1's standardization
            xp = phys(M1, k, xd)
            blk, i = PHYS[k]
            x2 = (xp - M2['st'][blk][0][i]) / M2['st'][blk][1][i]  # same cells in M2's standardization
            n1, n2 = effect(M1, k, xd), effect(M2, k, x2)
            n1 = n1 - np.polyval(np.polyfit(xp, n1, 1), xp); n2 = n2 - np.polyval(np.polyfit(xp, n2, 1), xp)
            cors.append(np.corrcoef(n1, n2)[0, 1])
        title.append(f'λ {r[5:]}: nonlinear {np.mean(nl):.1f} km/s, cross-fold r {np.nanmean(cors):.2f}')
        rows.append(dict(quantity=LABEL[k], model=r, nonlinear_rms=np.mean(nl), nonlinear_rms_sd=np.std(nl),
                         crossfold_r=np.nanmean(cors)))
    ax2 = a.twinx()
    ax2.hist(phys(MODELS['Q_lin'][0], k, allx), bins=60, range=(phys(MODELS['Q_lin'][0], k, grid)[0],
             phys(MODELS['Q_lin'][0], k, grid)[-1]), color=GRID, zorder=0)
    ax2.set_yticks([]); ax2.spines[:].set_visible(False); ax2.set_zorder(0); a.set_zorder(1); a.patch.set_visible(False)
    a.set_xlabel(LABEL[k]); a.set_ylabel('effect on boundary speed (km/s)' if k == 0 else '')
    a.set_title('\n'.join(title), fontsize=15)
axs[0].legend(loc='upper left', fontsize=13)
fig.suptitle('4-quantity models: learned effect of each quantity (10 models per penalty; dashed: linear model)', fontsize=19)
save(fig, 'fig_qnam')
R = pd.DataFrame(rows)
R.to_csv('runs/sweep5_lasso/q_nam_curves.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
