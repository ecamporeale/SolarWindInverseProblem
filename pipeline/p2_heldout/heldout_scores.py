"""One-time held-out scores (cycle 25, 2020-06-01 12:00 .. 2025-07-31 23:00) of the issue-time forecasts made by
realtime_forecast.py --heldout with the 10 frozen B0 models (runs/sweep3/B0/seed{42,43}/fold{1..5}).

Methods at every (issue t0, lead tau = 1..168 h), target hour T = t0 + tau, measured ACE hours only (ICME-masked):
  lastseen_ens   10-member ensemble mean, last-seen boundary after t0 (the real-time forecast)
  errpers_ens    lastseen_ens + w(tau) * [observed - model] at the anchor (most recent measured hour labelled
                 t0-1..t0-3), w frozen from development (--w-file; the 2026-10-13 B0 file is archived in
                 runs/archive_omega297/realtime_dev/w_frozen.csv; since 2026-10-22 one file per model)
  oracle_ens     ensemble mean with the real future frames (not available in real time; reference only)
  *_member       metrics of single members, averaged over the 10
  persistence    last measured ACE hour labelled <= t0 - 1 h (at most 24 h old)
  recurrence     ACE at T - 27 d
  linear         OLS on the 6 AIA set-2a features fitted on all development rows (image and target in the
                 development window), applied to the image at min(T - 4 d, t0) (newest image <= that time, <= 6 h old)
Matched rows per lead and phase: all methods finite. Phases by target time T: rise = before 2023-12-01, maximum =
from 2023-12-01 (first month with 13-month smoothed SSN >= 80 % of the cycle-25 peak, 160.9 in Oct 2024).
Paired 27-day block bootstrap (B = 1000, S recomputed per resample) of dS = S(a) - S(b) at selected leads.
--fixed-grid: one TIMSE percentile grid and v_max - v_min per phase set, from all measured held-out hours of that phase
(as in training), used for every lead, method and bootstrap resample; without it each lead's matched observations
define their own grid (S then shifts with the sample's observed range). Outputs get the suffix _fixedgrid.

  python heldout_scores.py [--dir runs/realtime_heldout] [--cuda-index 0]
Outputs: <dir>/scores_by_lead.csv, <dir>/bootstrap_dS.csv, <dir>/scores_by_lead.png
"""
import argparse
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval_matched_baselines import SET_2A, metrics
from realtime_wcut_test import _S_nan

HOUR = pd.Timedelta(hours=1)
HELD_OBS = 'data/preprocessed_window/multi_spacecraft_obs_heldout_20200401_20250731.csv'
MAX_START = pd.Timestamp('2023-12-01')
PIX = {'A: area f=0.30, w=20, off=0': 'A_area f=0.3 w=20.0 off=+0', 'B: ecliptic band |lat|<=15': 'B_eclband |lat|<=15',
       'A: area f=0.35, w=12.5, off=0': 'A_area f=0.35 w=12.5 off=+0',
       'D: mean log(I/med) of dark pixels, band |lat|<=30': 'D_mean log(I/med) of dark pixels, band |lat|<=30',
       'B: hole-boundary distance, off=+10': 'B_dchb signed dist off=+10',
       'A: ESWF definition (f=0.35, w=7.5, off=0)': 'A_area f=0.35 w=7.5 off=+0'}
BOOT_LEADS = [6, 12, 24, 48, 72, 96, 120, 144, 168]
BOOT_PAIRS = [('errpers_ens', 'lastseen_ens'), ('errpers_ens', 'persistence'), ('errpers_ens', 'recurrence'),
              ('errpers_ens', 'linear'), ('lastseen_ens', 'recurrence'), ('lastseen_ens', 'linear'), ('lastseen_ens', 'oracle_ens')]


def linear_heldout():
    """Linear model fitted on all development rows; predictions from every available image hour (dev + held-out)."""
    X = pd.read_pickle('data/final_feature_matrix_pixel.pkl')
    A = np.column_stack([np.ones(len(X)), X[SET_2A[:6]].to_numpy(float)])
    ok = np.isfinite(A).all(1) & np.isfinite(X.y.to_numpy(float))
    coef = np.linalg.lstsq(A[ok], X.y.to_numpy(float)[ok], rcond=None)[0]
    imgs = []
    for f, cut in (('data/pixel_features_aia.pkl', None), ('data/pixel_features_aia_heldout.pkl', None)):
        P = pd.read_pickle(f)
        P = pd.DataFrame({'t': pd.to_datetime(P.obstime).dt.round('h'), **{k: P[v].to_numpy(float) for k, v in PIX.items()}})
        imgs.append(P)
    P = pd.concat(imgs).drop_duplicates('t', keep='last').sort_values('t')
    B_ = np.column_stack([np.ones(len(P)), P[SET_2A[:6]].to_numpy(float)])
    good = np.isfinite(B_).all(1)
    return pd.Series(B_[good] @ coef, index=P.t.to_numpy()[good]), coef


def phase_grids(ace, lo, hi):
    """Fixed TIMSE grids from all measured target hours in [lo, hi), per phase set."""
    from train_joint_forecast import percentile_grid
    v = ace[(ace.index >= lo) & (ace.index < hi)].dropna()
    return {'all': percentile_grid(v.to_numpy()), 'rise': percentile_grid(v[v.index < MAX_START].to_numpy()),
            'maximum': percentile_grid(v[v.index >= MAX_START].to_numpy())}


def _S_grid(o, e2, grid):
    """S per column on a fixed grid; o [N, B] with NaN for padded rows, e2 [N, B]."""
    q = grid['q'].to(o.device, o.dtype)
    m = torch.isfinite(o)
    oz = torch.where(m, o, torch.full_like(o, -1e30))
    e2m = torch.where(m, e2, torch.zeros_like(e2))
    width = q[1:] - q[:-1]
    above = (oz[None] >= q[:-1, None, None]).to(o.dtype)                       # [K, N, B]
    cnt = above.sum(1)
    tail = (above * e2m[None]).sum(1) / cnt.clamp(min=1.0)
    use = ((cnt > 0) & (width[:, None] > 0)).to(o.dtype)
    timse = (width[:, None] * tail * use).sum(0) / (grid['vmax'] - grid['vmin'])
    return (e2m.sum(0) / m.sum(0) * timse) ** 0.25


def boot_dS(o, ea, eb, block, B, dev, chunk=50, seed=0, grid=None):
    rng = np.random.default_rng(seed)
    ids = np.unique(block)
    by = {b: np.flatnonzero(block == b) for b in ids}
    idxs = [np.concatenate([by[b] for b in rng.choice(ids, len(ids))]) for _ in range(B)]
    n_max = max(len(i) for i in idxs)
    ot, at, bt = (torch.tensor(x, device=dev, dtype=torch.float64) for x in (o, ea, eb))
    out = []
    for c in range(0, B, chunk):
        sub = idxs[c:c + chunk]
        I = torch.tensor(np.stack([np.resize(i, n_max) for i in sub], 1), device=dev)
        w = torch.tensor(np.stack([np.arange(n_max) < len(i) for i in sub], 1), device=dev)
        oo = ot[I].clone(); oo[~w] = float('nan')
        S_ = (lambda x, e: _S_nan(x, e)) if grid is None else (lambda x, e: _S_grid(x, e, grid))
        out.append(S_(oo, at[I]) - S_(oo, bt[I]))
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='runs/realtime_heldout')
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--obs-file', default=HELD_OBS, help='dry runs on development outputs only')
    ap.add_argument('--fixed-grid', action='store_true', help='one TIMSE grid per phase set from all measured held-out hours')
    ap.add_argument('--cuda-index', type=int, default=0)
    ap.add_argument('--w-file', default='runs/archive_omega297/realtime_dev/w_frozen.csv',
                    help='frozen error-persistence weights of THIS model (2026-10-22 protocol: '
                         'runs/part2_v2/realtime_dev/<model>/w_frozen.csv)')
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.cuda_index}')
    files = sorted(glob.glob(f'{args.dir}/seed*_fold*.npz'))
    Z = [np.load(f, allow_pickle=True) for f in files]
    issue = pd.to_datetime(Z[0]['issue']); lead_all = Z[0]['lead']
    assert all((pd.to_datetime(z['issue']) == issue).all() for z in Z)
    print(f'{len(Z)} members: {[f.split("/")[-1] for f in files]}; {len(issue)} issue times {issue[0]} -> {issue[-1]}', flush=True)
    sc, an = lead_all >= 1, (lead_all >= -3) & (lead_all <= -1)
    lead = lead_all[sc]
    obs_all = Z[0]['obs']
    I, L = len(issue), int(sc.sum())
    T = issue.to_numpy()[:, None] + lead[None, :] * np.timedelta64(1, 'h')
    d = {'lead': np.broadcast_to(lead, (I, L)), 'issue': np.broadcast_to(issue.to_numpy()[:, None], (I, L)), 'T': T,
         'obs': obs_all[:, sc]}
    for m in ('oracle', 'lastseen'):
        P_all = np.stack([z[f'pred_{m}'] for z in Z])
        d[f'{m}_ens'] = P_all[:, :, sc].mean(0)
        for s in range(len(Z)):
            d[f'{m}_mem{s}'] = P_all[s][:, sc]
        if m == 'lastseen':
            d['spread'] = P_all[:, :, sc].std(0, ddof=1)
            E_an = P_all.mean(0)[:, an] - obs_all[:, an]
            e_an = np.full(I, np.nan)
            for c in range(E_an.shape[1]):                                    # most recent measured hour wins
                e_an = np.where(np.isfinite(E_an[:, c]), E_an[:, c], e_an)
            Wf = pd.read_csv(args.w_file).set_index('lead').w.reindex(lead).to_numpy()
            d['errpers_ens'] = d['lastseen_ens'] - Wf[None, :] * np.nan_to_num(e_an)[:, None]
            d['anchor_found'] = np.broadcast_to(np.isfinite(e_an)[:, None], (I, L))
    o = pd.read_csv(args.obs_file, parse_dates=['Time'])
    ace = o[o.Spacecraft_ID == 'ACE'].set_index('Time')['Speed'].sort_index().resample('1h').mean()
    meas = ace.dropna()
    j = np.searchsorted(meas.index.to_numpy(), (issue - HOUR).to_numpy(), side='right') - 1
    pv = np.where(j >= 0, meas.to_numpy()[np.maximum(j, 0)], np.nan)
    pv[(issue.to_numpy() - meas.index.to_numpy()[np.maximum(j, 0)]) > np.timedelta64(24, 'h')] = np.nan
    d['persistence'] = np.broadcast_to(pv[:, None], (I, L))
    d['recurrence'] = ace.reindex(pd.DatetimeIndex(T.ravel()) - pd.Timedelta(days=27)).to_numpy().reshape(I, L)
    lin, coef = linear_heldout()
    t_src = np.minimum(T - np.timedelta64(4, 'D'), issue.to_numpy()[:, None]).ravel()
    li = np.searchsorted(lin.index.to_numpy(), t_src, side='right') - 1
    lv = lin.to_numpy()[np.maximum(li, 0)]
    lv[(li < 0) | ((t_src - lin.index.to_numpy()[np.maximum(li, 0)]) > np.timedelta64(6, 'h'))] = np.nan
    d['linear'] = lv.reshape(I, L)
    D = pd.DataFrame({c: np.asarray(v).ravel() for c, v in d.items()})
    D['phase'] = np.where(pd.to_datetime(D['T']) < MAX_START, 'rise', 'maximum')
    D['block'] = ((pd.to_datetime(D.issue) - issue[0]) / pd.Timedelta(days=27)).astype(int)
    D.to_pickle(f'{args.dir}/rows.pkl')
    sfx = '_fixedgrid' if args.fixed_grid else ''
    G = phase_grids(ace, issue[0], pd.Timestamp(issue[-1]) + 169 * HOUR) if args.fixed_grid else {}
    for k, gg in G.items():
        print(f'fixed grid {k}: {gg["n"]} hours, v_min {gg["vmin"]:.1f}, v_max {gg["vmax"]:.1f} km/s', flush=True)
    print(f'anchor found for {np.isfinite(e_an).mean() * 100:.1f}% of issue times; linear coefficients {np.round(coef, 2)}', flush=True)

    methods = ['oracle_ens', 'lastseen_ens', 'errpers_ens', 'persistence', 'recurrence', 'linear']
    mem = {m: [c for c in D if c.startswith(f'{m}_mem')] for m in ('oracle', 'lastseen')}
    rows, brows = [], []
    for phase, Dp in (('all', D), ('rise', D[D.phase == 'rise']), ('maximum', D[D.phase == 'maximum'])):
        for ld, g in Dp.groupby('lead'):
            g = g[np.isfinite(g[['obs'] + methods + mem['oracle'] + mem['lastseen']].to_numpy(float)).all(1)]
            if len(g) < 200:
                continue
            ob = g.obs.to_numpy(float)
            gr = G.get(phase)
            for m in methods:
                rows.append(dict(phase=phase, lead=ld, method=m, n=len(g), **metrics(ob, g[m].to_numpy(float), gr)))
            for m in mem:
                mm = pd.DataFrame([metrics(ob, g[c].to_numpy(float), gr) for c in mem[m]]).mean()
                rows.append(dict(phase=phase, lead=ld, method=f'{m}_member', n=len(g), **mm.to_dict()))
            rows.append(dict(phase=phase, lead=ld, method='spread', n=len(g), rmse=float(np.sqrt(np.mean(g.spread ** 2)))))
            if ld in BOOT_LEADS:
                for a, b in BOOT_PAIRS:
                    ea, eb = (g[a].to_numpy(float) - ob) ** 2, (g[b].to_numpy(float) - ob) ** 2
                    ds = boot_dS(ob, ea, eb, g.block.to_numpy(), args.B, dev, grid=gr)
                    s_a, s_b = metrics(ob, g[a].to_numpy(float), gr)['S'], metrics(ob, g[b].to_numpy(float), gr)['S']
                    lo_, hi_ = np.percentile(ds, [2.5, 97.5])
                    brows.append(dict(phase=phase, lead=ld, a=a, b=b, n=len(g), S_a=s_a, S_b=s_b, dS=s_a - s_b, lo=lo_, hi=hi_))
        print(f'phase {phase}: done', flush=True)
    R = pd.DataFrame(rows); R.to_csv(f'{args.dir}/scores_by_lead{sfx}.csv', index=False)
    BR = pd.DataFrame(brows); BR.to_csv(f'{args.dir}/bootstrap_dS{sfx}.csv', index=False)

    fig, ax = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    sty = {'oracle_ens': ('k', '-'), 'lastseen_ens': ('C3', '-'), 'errpers_ens': ('C4', '-'), 'persistence': ('C0', '--'),
           'recurrence': ('C2', '--'), 'linear': ('C1', '--')}
    Ra = R[R.phase == 'all']
    for a_, met in zip(ax.ravel(), ['S', 'rmse', 'tss500', 'hss500', 'r', 'bias']):
        for m, (c, ls) in sty.items():
            q = Ra[Ra.method == m]
            a_.plot(q.lead / 24.0, q[met], color=c, ls=ls, label=m)
        a_.set_ylabel(met); a_.grid(alpha=.3)
    for a_ in ax[-1]:
        a_.set_xlabel('lead (days)')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle('Held-out (cycle 25, 2020-06 .. 2025-07): issue-time forecasts, 10-member B0 ensemble, matched hours per lead')
    fig.tight_layout(); fig.savefig(f'{args.dir}/scores_by_lead{sfx}.png', dpi=110)
    pd.set_option('display.width', 220)
    sel = R[R.lead.isin([1, 6, 12, 24, 48, 72, 96, 120, 144, 168])]
    for phase in ('all', 'rise', 'maximum'):
        for met in ('S', 'rmse', 'bias', 'r', 'tss500', 'hss500', 'tss600', 'hss600'):
            q = sel[(sel.phase == phase) & (sel.method != 'spread')]
            print(f'--- {phase}: {met}'); print(q.pivot(index='lead', columns='method', values=met).round(3 if met.startswith(('r', 't', 'h')) else 1).to_string())
    print(R[(R.phase == 'all') & (R.method == 'spread') & R.lead.isin([1, 24, 72, 120, 168])][['lead', 'rmse']].rename(columns={'rmse': 'rms spread'}).to_string(index=False))
    print(BR.round(2).to_string(index=False))
    print(f'saved {args.dir}/scores_by_lead{sfx}.csv, bootstrap_dS{sfx}.csv, scores_by_lead{sfx}.png')


if __name__ == '__main__':
    main()
