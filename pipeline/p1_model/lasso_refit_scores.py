"""Relaxed refits of the group-lasso selections (2026-10-20; worklog INTERPRETABLE_BOUNDARY.md §8).

R_A (6 groups: a0, a1, a3, a4, fs, rec), R_B (4: a0, a1, a3, rec), R_C (5: R_B + fs), each the linear map trained
without penalty on the selected groups only (runs/sweep5_lasso/R_*, 5 folds × 2 seeds); since 2026-10-21 also the
reduced models Q_lin / Q_nam* (one lag-averaged input per quantity of R_B, §9 of the worklog) once evaluated; compared
with B0 and with the
full linear map F_lin (37 inputs) on the matched development test hours: headline convention (2-seed mean, pooled),
paired 27-day block-bootstrap 95 % intervals of ΔS against B0 and against F_lin (as factorial_matched.py).
Coefficient stability: every weight of every model (standardized inputs), mean and s.d. over the 10 models, and the
number of models agreeing with the sign of the mean.

  python pipeline/p1_model/lasso_refit_scores.py
Reduced linear model Q_lin: each coefficient (logit per standardized unit) and its effect on the boundary speed at
400 km/s per standard deviation (×154 km/s) and per physical unit (÷ the feature's standard deviation).
Output: runs/sweep5_lasso/refit_matched.csv, runs/sweep5_lasso/refit_coefficients.csv, runs/sweep5_lasso/q_lin_coefficients.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from b0_skill_metrics import two_seed
from eval_matched_baselines import ace_hourly, metrics
from heldout_scores import boot_dS
from train_joint_forecast import LINEAR_GROUPS

MODELS = [('B0', 'runs/sweep4/B0', 'B0 (network, 37 inputs)'), ('F_lin', 'runs/sweep4/F_lin', 'linear, 37 inputs'),
          ('R_A', 'runs/sweep5_lasso/R_A', 'linear, 6 groups (23 weights)'),
          ('R_C', 'runs/sweep5_lasso/R_C', 'linear, 5 groups (18 weights)'),
          ('R_B', 'runs/sweep5_lasso/R_B', 'linear, 4 groups (17 weights)')]
import glob as _glob, sys as _sys
for _key, _lab in (('Q_lin', 'linear, 4 lag-averaged quantities (10 weights)'),
                   ('Q_nam0.3', 'NAM, 4 curves, curvature penalty 0.3'), ('Q_nam3', 'NAM, 4 curves, curvature penalty 3'),
                   ('H_hinge2', 'piecewise linear, 2 breakpoints/quantity, start 25/75'),
                   ('H_start_10_50', 'piecewise linear, start 10/50'), ('H_start_50_90', 'piecewise linear, start 50/90')):
    if len(_glob.glob(f'runs/sweep5_lasso/{_key}/seed*/matched_baselines.csv')) == 2:    # include once evaluated
        MODELS.append((_key, f'runs/sweep5_lasso/{_key}', _lab))
LAGS = ['t', 't-12h', 't-24h', 't-48h', 't-72h']
NAMES = {'a0': 'dark area f0.30 ±20°', 'a1': 'dark fraction |β|≤15°', 'a2': 'dark area f0.35', 'a3': 'dark-pixel intensity',
         'a4': 'hole-boundary distance', 'a5': 'ESWF area', 'fs': 'log10 f_s', 'rec': 'recurrence'}

if __name__ == '__main__':
    ace = ace_hourly()
    D = None
    for key, path, _ in MODELS:
        T = two_seed(path, ace)
        cols = ['time', 'nn'] if D is not None else ['time', 'obs', 'linear', 'recurrence', 'nn']
        T = T[cols].rename(columns={'nn': key})
        D = T if D is None else D.merge(T, on='time', how='inner')
    D = D.dropna().sort_values('time')
    o = D.obs.to_numpy()
    t = pd.to_datetime(D.time)
    blk = ((t - t.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rows = []
    for key, _, lab in MODELS:
        p = D[key].to_numpy()
        m = metrics(o, p)
        row = dict(model=key, label=lab, n=len(o), S=m['S'], rmse=m['rmse'], trmse=m['trmse'], bias=m['bias'], r=m['r'],
                   amp=m['amp'], tss500=m['tss500'], tss600=m['tss600'], auc500=roc_auc_score(o >= 500, p),
                   auc600=roc_auc_score(o >= 600, p))
        for ref in ('B0', 'F_lin'):
            q = D[ref].to_numpy()
            row[f'dS_vs_{ref}'] = m['S'] - metrics(o, q)['S']
            if key != ref:
                ds = boot_dS(o, (p - o) ** 2, (q - o) ** 2, blk, 1000, dev)
                row[f'dS_vs_{ref}_lo'], row[f'dS_vs_{ref}_hi'] = np.percentile(ds, [2.5, 97.5])
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv('runs/sweep5_lasso/refit_matched.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.drop(columns='label').round(3).to_string(index=False))

    C = []
    for key in ('R_A', 'R_C', 'R_B'):
        W = []
        for seed in (42, 43):
            for fold in range(1, 6):
                sd = torch.load(f'runs/sweep5_lasso/{key}/seed{seed}/fold{fold}/best.pt', map_location='cpu',
                                weights_only=False)['nn1_state_dict']
                W.append(sd['head.0.weight'][0].numpy())
        W = np.array(W)
        for g, idx in LINEAR_GROUPS.items():
            if np.all(W[:, idx] == 0):
                continue
            for lag, i in zip(LAGS if len(idx) == 5 else (['value', 'valid flag'] if g == 'rec' else ['value']), idx):
                w = W[:, i]
                C.append(dict(model=key, group=NAMES[g], input=lag, mean=w.mean(), sd=w.std(),
                              same_sign=int((np.sign(w) == np.sign(w.mean())).sum())))
            s = W[:, idx[:5] if len(idx) == 5 else idx[:1]].sum(1)
            C.append(dict(model=key, group=NAMES[g], input='sum over lags' if len(idx) == 5 else 'total', mean=s.mean(),
                          sd=s.std(), same_sign=int((np.sign(s) == np.sign(s.mean())).sum())))
    Ct = pd.DataFrame(C)
    Ct.to_csv('runs/sweep5_lasso/refit_coefficients.csv', index=False)
    print(Ct[Ct.input.isin(['sum over lags', 'total', 'value'])].round(3).to_string(index=False))

    # the reduced linear model: one coefficient per quantity
    import os
    if os.path.exists('runs/sweep5_lasso/Q_lin/seed42/fold1/best.pt'):
        K = 650 * (250 / 650) * (1 - 250 / 650)
        names = ['dark area f0.30 ±20° (lag mean)', 'dark fraction |β|≤15° (lag mean)', 'dark-pixel intensity (lag mean)',
                 'recurrence (km/s)', 'recurrence valid flag', 'image at t−12 h present', 't−24 h present',
                 't−48 h present', 't−72 h present']
        unit_sd = []                                    # physical s.d. per input, per model (feature stats of its fold)
        W, SD = [], []
        for seed in (42, 43):
            for fold in range(1, 6):
                sd = torch.load(f'runs/sweep5_lasso/Q_lin/seed{seed}/fold{fold}/best.pt', map_location='cpu',
                                weights_only=False)['nn1_state_dict']
                W.append(sd['head.inner.0.weight'][0].numpy())
                fs_ = sd['feat_std'].flatten().numpy(); es = sd['extra_std'].flatten().numpy()
                SD.append([fs_[0], fs_[1], fs_[3], es[1], es[2], 1, 1, 1, 1])
        W, SD = np.array(W), np.array(SD)
        Q = pd.DataFrame(dict(input=names, coef=W.mean(0), coef_sd=W.std(0),
                              same_sign=[int((np.sign(W[:, i]) == np.sign(W[:, i].mean())).sum()) for i in range(W.shape[1])],
                              kms_per_sd=W.mean(0) * K, kms_per_sd_sd=W.std(0) * K,
                              kms_per_unit=(W / SD).mean(0) * K, phys_sd=SD.mean(0)))
        Q.to_csv('runs/sweep5_lasso/q_lin_coefficients.csv', index=False)
        print(Q.round(4).to_string(index=False))
