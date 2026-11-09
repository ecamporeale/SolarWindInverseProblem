"""Does the solver beat the ballistic link where stream interaction acts? (2026-10-22; worklog INTERPRETABLE_BOUNDARY.md §14)

The solver (B0) and the forward ballistic link (F_ball: same network, inputs, loss; parcels travel at their own speed
plus ΔV(r), overlapping parcels averaged, no interaction) differ only where fast wind overtakes slow wind. The global
score dilutes such hours, so the models are compared inside the interfaces, with the fixed lag (F_lag) as reference.
Out-of-fold development test predictions, mean of seeds 42 and 43 (two_seed), ACE-hourly, ICME-masked.

Interface hours, two definitions:
  observed   steep rises of the measured speed: T0 with v(T0 + 24 h) − v(T0) ≥ 150 km/s (both measured); consecutive
             qualifying T0 merged into one event at the largest rise; window [T0 − 12 h, T0 + 48 h]. Conditions on the
             outcome, but every model is scored on the same hours (as in event verification).
  overtaking from B0's own out-of-fold boundary (seed 42, fold of the test batch): forward ballistic arrival times
             A(t_d) of the parcels departing 1–7 d before each hour t from the longitude that then faced the
             spacecraft (as train_joint_forecast.kinematic_track); the hour is multi-valued if a parcel arriving
             within ±3 h of t has dA/dt_d < 0 (a later departure arrives earlier: overtaking). No observations used.
Control: hours in neither set.
Per set: RMSE and bias of each model; ΔRMSE(B0 − F_ball) and ΔRMSE(B0 − F_lag) with 95 % 27-day block-bootstrap
intervals. Per observed event: lag (h) of the best cross-correlation of each model with the observation within the
window (shifts ±24 h), mean |lag| and mean signed lag (positive = model late).

  python pipeline/p1_model/interface_check.py [--cuda-index 0]
Output: runs/sweep4/interface_check.csv, runs/sweep4/interface_timing.csv, runs/sweep4/interface_hours.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import folds
from b0_skill_metrics import two_seed
from eval_matched_baselines import ace_hourly

MODELS = {'B0': 'runs/sweep4/B0', 'F_ball': 'runs/sweep4/F_ball', 'F_lag': 'runs/sweep4/F_lag'}
RISE, WIN_BEFORE, WIN_AFTER = 150.0, 12, 48


def overtaking_hours(dev):
    from plot_best_checkpoint import chunks_of, load_fold
    from train_joint_forecast import make_batch_record
    flags = {}
    for fold in range(1, 6):
        F = load_fold(fold, Path(f'runs/sweep4/B0/seed42/fold{fold}'), dev)
        cfg, fm = F['config'], F['forward_model']
        L, Vref = (x.to(dev, torch.float32) for x in fm.learned_radial_accel.physical_parameters())
        Lk, r0, Om, dt = L * 695700.0, float(cfg.r0), float(cfg.Omega), float(cfg.dt)
        for cid in chunks_of(F, 'test'):
            rec = make_batch_record(F['specs'][cid], cfg, dev)
            with torch.no_grad():
                v = F['nn1'](rec['image_slice'])                                   # [Nphi, Nt]
                Nphi, Nt = v.shape
                t = rec['time_obs_t'].to(dev, torch.float32)
                ph = rec['phi_obs_t'].to(dev, torch.float32)
                r = rec['radius_obs_t'].to(dev, torch.float32)
                r = r.expand_as(t) if r.dim() == 0 else r
                ok = torch.isfinite(t) & torch.isfinite(ph) & torch.isfinite(r)
                k = torch.arange(int(round(86400 / dt)), int(round(7 * 86400 / dt)) + 1, device=dev, dtype=torch.float32)
                lag = k * dt
                td = t[:, None] - lag[None, :]
                x = torch.remainder(ph[:, None] + Om * lag[None, :], 2 * math.pi) / (2 * math.pi / Nphi)
                j0 = torch.floor(x).long() % Nphi; a = x - torch.floor(x); j1 = (j0 + 1) % Nphi
                tc = torch.clamp(td / dt, 0, Nt - 1); n0 = torch.clamp(torch.floor(tc), max=Nt - 2).long(); b = tc - n0
                vb = (1 - a) * ((1 - b) * v[j0, n0] + b * v[j0, n0 + 1]) + a * ((1 - b) * v[j1, n0] + b * v[j1, n0 + 1])
                X = ((r - r0) * 695700.0)[:, None]
                A = vb + Vref
                tau = (X + Lk * torch.log((A - Vref * torch.exp(-X / Lk)) / vb)) / A
                arr = td + tau                                                     # arrival time of each parcel
                # departures are ordered from latest (k small) to earliest; overtaking: dA/dt_d < 0, i.e. along
                # increasing lag (earlier departure) the arrival time increases
                dA = arr[:, :-1] - arr[:, 1:]                                      # A(later dep) − A(earlier dep)
                near = (arr[:, :-1] - t[:, None]).abs() <= 3 * 3600
                multi = ((dA < 0) & near & (td[:, 1:] >= 0)).any(1) & ok
            t0 = pd.Timestamp(F['specs'][cid]['batch_start_time'])
            hours = (t0 + pd.to_timedelta(t.cpu().numpy(), unit='s')).round('h')
            for h, m in zip(hours, multi.cpu().numpy()):
                flags.setdefault(h, bool(m))
        print(f'fold {fold}: {len(flags):,} hours flagged so far, {np.mean(list(flags.values())):.3f} multi-valued', flush=True)
    return pd.Series(flags, name='overtaking')


def observed_events(obs):
    s = obs.asfreq('1h')
    rise = s.shift(-24) - s
    cand = rise[rise >= RISE].index
    events, cur = [], []
    for t in cand:
        if cur and t - cur[-1] > pd.Timedelta(hours=1):
            events.append(max(cur, key=lambda x: rise[x])); cur = []
        cur.append(t)
    if cur:
        events.append(max(cur, key=lambda x: rise[x]))
    return events


def boot_drmse(e_a, e_b, blk, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blk); idx = {u: np.flatnonzero(blk == u) for u in ub}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
        out.append(np.sqrt(np.mean(e_a[s] ** 2)) - np.sqrt(np.mean(e_b[s] ** 2)))
    return np.percentile(out, [2.5, 97.5])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda-index', type=int, default=0)
    a = ap.parse_args()
    ace = ace_hourly()
    D = None
    for key, path in MODELS.items():
        T = two_seed(path, ace)
        cols = ['time', 'nn'] if D is not None else ['time', 'obs', 'nn']
        T = T[cols].rename(columns={'nn': key})
        D = T if D is None else D.merge(T, on='time', how='inner')
    D = D.dropna().drop_duplicates('time').set_index('time').sort_index()
    obs_all = ace[(ace.index >= folds.DEV_START) & (ace.index <= folds.DEV_END)]
    ev = [e for e in observed_events(obs_all)]
    win = pd.Series(False, index=D.index)
    for e in ev:
        win[(D.index >= e - pd.Timedelta(hours=WIN_BEFORE)) & (D.index <= e + pd.Timedelta(hours=WIN_AFTER))] = True
    ov = overtaking_hours(torch.device(f'cuda:{a.cuda_index}'))
    D['observed_interface'] = win
    D['overtaking'] = ov.reindex(D.index).fillna(False).astype(bool)
    D[['obs', 'observed_interface', 'overtaking']].to_csv('runs/sweep4/interface_hours.csv')
    blk = ((D.index - D.index.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
    rows = []
    sets = {'all hours': np.ones(len(D), bool), 'observed interfaces': D.observed_interface.to_numpy(),
            'overtaking (B0 boundary)': D.overtaking.to_numpy(),
            'control (neither)': ~(D.observed_interface.to_numpy() | D.overtaking.to_numpy()),
            'both': D.observed_interface.to_numpy() & D.overtaking.to_numpy()}
    o = D.obs.to_numpy()
    for name, m in sets.items():
        e = {k: D[k].to_numpy()[m] - o[m] for k in MODELS}
        row = dict(set=name, hours=int(m.sum()), share=m.mean(), obs_mean=o[m].mean())
        for k in MODELS:
            row[f'rmse_{k}'] = np.sqrt(np.mean(e[k] ** 2)); row[f'bias_{k}'] = e[k].mean()
        for ref in ('F_ball', 'F_lag'):
            row[f'dRMSE_B0_vs_{ref}'] = row['rmse_B0'] - row[f'rmse_{ref}']
            row[f'lo_{ref}'], row[f'hi_{ref}'] = boot_drmse(e['B0'], e[ref], blk[m])
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv('runs/sweep4/interface_check.csv', index=False)
    tim = []
    for e0 in ev:
        w = D.loc[e0 - pd.Timedelta(hours=WIN_BEFORE + 24): e0 + pd.Timedelta(hours=WIN_AFTER + 24)]
        if len(w) < 60:
            continue
        y = w.obs.reindex(pd.date_range(w.index[0], w.index[-1], freq='1h'))
        row = dict(event=e0, n=int(y.notna().sum()))
        for k in MODELS:
            p = w[k].reindex(y.index)
            best, bl = -2, np.nan
            for L_ in range(-24, 25):
                c = pd.concat([y, p.shift(L_)], axis=1).dropna()
                if len(c) > 40:
                    r_ = np.corrcoef(c.iloc[:, 0], c.iloc[:, 1])[0, 1]
                    if r_ > best:
                        best, bl = r_, L_
            row[f'lag_{k}'] = -bl if np.isfinite(bl) else np.nan   # best when the model is delayed by bl h: model early by bl, late by −bl
            row[f'r_{k}'] = best
        tim.append(row)
    Tm = pd.DataFrame(tim)
    Tm.to_csv('runs/sweep4/interface_timing.csv', index=False)
    pd.set_option('display.width', 250)
    print(f'observed rise events: {len(ev)}; with a usable window: {len(Tm)}')
    print(R.round(2).to_string(index=False))
    for k in MODELS:
        print(f'{k}: mean |lag| {Tm[f"lag_{k}"].abs().mean():.1f} h, mean lag {Tm[f"lag_{k}"].mean():+.1f} h, '
              f'median r {Tm[f"r_{k}"].median():.2f}')
