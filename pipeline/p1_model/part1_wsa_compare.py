"""Part I comparison with a physics model (author's request, 2026-10-24): B0 vs WSA 4.5 + HUXt on B0's test periods,
run exactly as B0 is evaluated there (development 2011-01 .. 2020-06; not a forecast from issue times).

Inputs: B0's matched test table (b0_skill_metrics.two_seed: hourly time, obs, nn = mean of seeds 42/43, linear,
recurrence; measured ACE hours of the test batches, warm-up excluded) and the WSA 4.5 runs over the same test chunks:
  wsa_huxt    WSA 4.5 + HUXt, time-dependent boundary (part1_wsa_huxt_test.py; headline)
  wsa_solver  WSA 4.5 boundary through B0's own solver, mean of the seed-42/43 solvers (part1_wsa_solver_test.py;
              boundary-only check: NN1 vs WSA with the propagation fixed)
  wsa_ballistic  WSA 4.5 + forward ballistic mapping, Arge & Pizzo (2000) (part1_wsa_ballistic_test.py)
The public WSA 5.4 archive of Part II has only four maps in this window; WSA 4.5 (GONG) is public hourly 2006 .. 2020-11
(CCMC iSWA), the same format and speed equation, the version the UK Met Office runs.
Rows: hours where every method is finite. Scores as Part I's matched tables (metrics: S with the TIMSE grid of these
hours, RMSE, bias, r, TSS/HSS); paired 27-day block bootstrap of ΔS (B = 1000); per fold; errors binned by the observed
and by the predicted speed; bias-free scores (ROC AUC, best-threshold TSS) for B0 vs each WSA version.

  python pipeline/p1_model/part1_wsa_compare.py [--B 1000] [--cuda-index 3]
2026-10-25: + WSA through the factorial test's ballistic link (fball_test.csv, if present); + WSA + HUXt recalibrated
per fold (a + b·WSA by least squares on the fold's training batches, as the linear comparator); + skill.csv, the
bias-aware measures of the skill table.
Output: runs/part1_wsa/{matched,scores,bootstrap_dS,scores_by_fold,binned,biasfree,skill}.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'pipeline/p2_comparison')
from b0_skill_metrics import _bf, roc_auc_score, tss, two_seed  # noqa: E402
import folds                                                   # noqa: E402
from eval_matched_baselines import ace_hourly, metrics         # noqa: E402
from frozen_model import B0_RUN                                # noqa: E402
from heldout_scores import boot_dS                             # noqa: E402
from train_joint_forecast import percentile_grid               # noqa: E402

OUT = 'runs/part1_wsa'
EDGES = np.arange(225, 900, 25)
NAMES = {'nn': 'B0', 'wsa_huxt': 'WSA 4.5 + HUXt', 'wsa_hux': 'WSA 4.5 + HUX (steady)', 'wsa_solver': 'WSA 4.5 boundary + our solver',
         'wsa_ballistic': 'WSA 4.5 + ballistic', 'wsa_fball': 'WSA 4.5 boundary + our ballistic link',
         'wsa_huxt_recal': 'WSA 4.5 + HUXt, recalibrated',
         'linear': 'linear (6 AIA features)', 'recurrence': '27-day recurrence'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--cuda-index', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device('cpu' if a.cuda_index < 0 else f'cuda:{a.cuda_index}')
    D = two_seed(B0_RUN, ace_hourly())
    Hx = pd.read_csv(f'{OUT}/huxt_test.csv', parse_dates=['time'])[['time', 'wsa_huxt', 'map_age_h']]
    Sv = pd.read_csv(f'{OUT}/solver_test.csv', parse_dates=['time']).groupby('time').wsa_solver.mean().reset_index()
    Bl = pd.read_csv(f'{OUT}/ballistic_test.csv', parse_dates=['time'])[['time', 'wsa_ballistic']]
    D = D.merge(Hx, on='time', how='left').merge(Sv, on='time', how='left').merge(Bl, on='time', how='left')
    import os
    if os.path.exists(f'{OUT}/hux_test.csv'):                         # WSA 4.5 + steady HUX (part1_wsa_hux_test.py, 2026-10-26)
        D = D.merge(pd.read_csv(f'{OUT}/hux_test.csv', parse_dates=['time'])[['time', 'wsa_hux']], on='time', how='left')
    if os.path.exists(f'{OUT}/fball_test.csv'):                       # WSA through the factorial test's ballistic link (2026-10-25)
        Fb = pd.read_csv(f'{OUT}/fball_test.csv', parse_dates=['time']).groupby('time').wsa_fball.mean().reset_index()
        D = D.merge(Fb, on='time', how='left')
    cols = [c for c in NAMES if c in D.columns]
    M = D.dropna(subset=['obs'] + cols).sort_values('time').reset_index(drop=True)
    # WSA + HUXt recalibrated like the linear comparator: a + b·WSA, least squares on each fold's three training batches
    # (hours that are test hours of other folds; WSA is not trained, so these predictions are out of sample), applied
    # to the fold's test batch
    test_batch = {k: te for k, (tr, va, te) in folds.FOLDS.items()}
    hb = M.fold.map(test_batch).to_numpy()
    M['wsa_huxt_recal'] = np.nan; coef = []
    for k, (tr, va, te) in folds.FOLDS.items():
        fit = np.isin(hb, tr)
        b_, a_ = np.polyfit(M.wsa_huxt[fit], M.obs[fit], 1)
        M.loc[M.fold == k, 'wsa_huxt_recal'] = a_ + b_ * M.wsa_huxt[M.fold == k]
        coef.append(dict(fold=k, intercept=a_, slope=b_))
    print('recalibration a + b·WSA per fold:', pd.DataFrame(coef).round(3).to_dict('records'), flush=True)
    cols = cols + ['wsa_huxt_recal']
    M.to_csv(f'{OUT}/matched.csv', index=False)
    print(f'{len(M)} matched hours of {len(D)} scored test hours; map older than 48 h: {(M.map_age_h > 48).mean():.3f}', flush=True)
    o = M.obs.to_numpy(); grid = percentile_grid(o)
    t = pd.to_datetime(M.time); blk = ((t - t.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
    rows = [dict(method=NAMES[c], n=len(M), **metrics(o, M[c].to_numpy(), grid)) for c in cols]
    brows = []
    pairs = [('nn', 'wsa_huxt'), ('nn', 'wsa_solver'), ('nn', 'wsa_ballistic'), ('wsa_huxt', 'wsa_ballistic'),
             ('wsa_huxt', 'linear'), ('wsa_huxt', 'recurrence'), ('nn', 'wsa_huxt_recal'), ('wsa_huxt_recal', 'linear')]
    if 'wsa_fball' in cols:
        pairs += [('wsa_solver', 'wsa_fball'), ('wsa_huxt', 'wsa_fball')]
    if 'wsa_hux' in cols:
        pairs += [('nn', 'wsa_hux'), ('wsa_hux', 'wsa_huxt'), ('wsa_hux', 'linear'), ('wsa_hux', 'recurrence')]
    for x, y in pairs:
        ds = boot_dS(o, (M[x].to_numpy() - o) ** 2, (M[y].to_numpy() - o) ** 2, blk, a.B, dev, grid=grid)
        sa, sb = metrics(o, M[x].to_numpy(), grid)['S'], metrics(o, M[y].to_numpy(), grid)['S']
        lo, hi = np.percentile(ds, [2.5, 97.5])
        brows.append(dict(a=NAMES[x], b=NAMES[y], n=len(M), S_a=sa, S_b=sb, dS=sa - sb, lo=lo, hi=hi, share_a_better=float(np.mean(ds < 0))))
    frows = []
    for fold, g in M.groupby('fold'):
        og = g.obs.to_numpy()
        for c in cols:
            frows.append(dict(fold=fold, method=NAMES[c], n=len(g), **metrics(og, g[c].to_numpy())))
    crows = []
    for c in cols:
        p = M[c].to_numpy(); e = p - o
        for cond, x in (('observed', o), ('predicted', p)):
            b = np.digitize(x, EDGES)
            for j in np.unique(b):
                sel = b == j
                if sel.sum() < 50 or j == 0 or j == len(EDGES):
                    continue
                crows.append(dict(method=NAMES[c], conditioned_on=cond, bin_lo=EDGES[j - 1], bin_hi=EDGES[j], n=int(sel.sum()),
                                  rmse=float(np.sqrt(np.mean(e[sel] ** 2))), bias=float(e[sel].mean())))
    rng = np.random.default_rng(0)
    ub = np.unique(blk); idx = {u: np.flatnonzero(blk == u) for u in ub}
    boots = [np.concatenate([idx[u] for u in rng.choice(ub, len(ub))]) for _ in range(a.B)]
    qrows = []
    for y in [c for c in cols if c.startswith('wsa')]:
        for v in (500, 600):
            pn, pw = M.nn.to_numpy(), M[y].to_numpy()
            x1, x2 = _bf(o, pn, v), _bf(o, pw, v)
            d = np.array([[p_ - q_ for p_, q_ in zip(_bf(o[s], pn[s], v)[:3], _bf(o[s], pw[s], v)[:3])] for s in boots])
            lo, hi = np.percentile(d, [2.5, 97.5], 0)
            qrows.append(dict(model=NAMES[y], threshold=v, n_events=int((o >= v).sum()), n_b0_fcst_above=int((pn >= v).sum()),
                              n_model_fcst_above=int((pw >= v).sum()), tss_b0=x1[0], tss_model=x2[0], auc_b0=x1[1], auc_model=x2[1],
                              best_tss_b0=x1[2], best_tss_model=x2[2], best_thr_b0=x1[3], best_thr_model=x2[3],
                              d_tss=x1[0] - x2[0], d_tss_lo=lo[0], d_tss_hi=hi[0], d_auc=x1[1] - x2[1], d_auc_lo=lo[1], d_auc_hi=hi[1],
                              d_best=x1[2] - x2[2], d_best_lo=lo[2], d_best_hi=hi[2]))
    # the bias-aware measures of Part I's skill table (b0_skill_metrics), for every method on these hours
    srows = []
    for c in cols:
        p = M[c].to_numpy(); b = float(np.mean(p - o)); pd_ = p - b
        row = dict(method=NAMES[c], n=len(o), bias=b, rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                   rmse_debiased=float(np.sqrt(np.mean((pd_ - o) ** 2))), r=float(np.corrcoef(p, o)[0, 1]), amp=float(np.std(p) / np.std(o)))
        for v in (500, 600):
            row[f'tss{v}'] = tss(p, o, v, v); row[f'tss{v}_debiased'] = tss(pd_, o, v, v)
            row[f'tss{v}_best'] = max(tss(p, o, th, v) for th in np.arange(300, 801, 5)); row[f'auc{v}'] = roc_auc_score(o >= v, p)
        srows.append(row)
    for name, X in (('scores', rows), ('bootstrap_dS', brows), ('scores_by_fold', frows), ('binned', crows), ('biasfree', qrows),
                    ('skill', srows)):
        pd.DataFrame(X).to_csv(f'{OUT}/{name}.csv', index=False)
    pd.set_option('display.width', 220)
    print(pd.DataFrame(rows)[['method', 'n', 'S', 'rmse', 'bias', 'r', 'amp', 'tss500', 'tss600']].round(2).to_string(index=False))
    print(pd.DataFrame(brows)[['a', 'b', 'S_a', 'S_b', 'dS', 'lo', 'hi']].round(1).to_string(index=False))
    print(pd.DataFrame(frows).pivot(index='fold', columns='method', values='S').round(1).to_string())
    print(pd.DataFrame(qrows).round(2).to_string(index=False))
    print(pd.DataFrame(srows).round(3).to_string(index=False))
