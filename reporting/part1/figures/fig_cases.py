"""Part I figure `fig_cases` (manuscript label fig:cases; Figure 10 in the 2026-10 draft): four case studies selected by rule (pipeline/p1_model/dev_cases_select.py; events from the PROSWIN
high-speed-stream protocol on the development out-of-fold predictions, dev_events_*.py): best success, typical
success (median error), largest missed stream, steepest stream interface.

Per case (column; redesigned 2026-10-26, boundary-map row removed):
  row 1  AIA 193 Å frame at the departure time of the observed peak from 21.5 R☉ (peak time minus the travel time at
         the peak speed, with the model's residual acceleration), dark pixels in blue: the source near disk centre;
  row 2  speed at ACE over ±5 d around the peak: measured, 27-day recurrence, the reference model (mean of 2 seeds, at
         every hour from the solved field, as fig_reconstruction), WSA 4.5 + HUXt (runs/part1_wsa/huxt_test.csv);
         no ICME (HELIO4CAST, Wind) overlaps any of the four
         windows, so no hours are masked.

  python reporting/part1/figures/fig_cases.py [--cuda-index 0]
Output: figures/part1/fig_cases.png/.pdf
"""
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from matplotlib.colors import ListedColormap
from sunpy.coordinates import sun
from sunpy.map import Map

import build_pixel_features as BPF
import folds

from _style import BLUES, C, GRID, INK, INK2, LBL, save
from helpers import sample_satellite_track
from eval_matched_baselines import ace_hourly
from plot_best_checkpoint import _transient_skip_steps, chunks_of, load_fold, run_chunk
from frozen_model import B0_RUN

ap = argparse.ArgumentParser()
ap.add_argument('--cuda-index', type=int, default=0)
args = ap.parse_args()
dev = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')

CASES = pd.read_csv('runs/dev_events/cases.csv', parse_dates=['obs_peak', 'pred_peak'])
A = pd.read_csv('runs/dev_events/ace_axis.csv', parse_dates=['time']).set_index('time')
OBS = ace_hourly()
PIX = pd.read_pickle('data/pixel_features_aia.pkl')[['obstime', 'path']]
PIX['obstime'] = pd.to_datetime(PIX.obstime)
HALF, RS, R0, RA = pd.Timedelta(days=5), 695700.0, 21.5, 213.5
TITLE = {'best': 'Best success', 'typical': 'Typical success', 'failure': 'Missed stream',
         'interface': 'Steep stream interface'}
_folds = {}


def fold_of(t, seed=42):
    b = int(folds.batch_of(np.array([t.to_datetime64()]))[0])
    k = [f for f, v in folds.FOLDS.items() if v[2] == b][0]
    if (k, seed) not in _folds:
        _folds[(k, seed)] = load_fold(k, f'{B0_RUN}/seed{seed}/fold{k}', dev)
    return _folds[(k, seed)]


def chunk_for(F, lo, hi):
    """Test chunk whose scored part covers [lo, hi] best."""
    best, cov = None, -1
    for cid in chunks_of(F, 'test'):
        s = F['specs'][cid]
        a0, a1 = pd.Timestamp(s['batch_start_time']) + pd.Timedelta(hours=150), pd.Timestamp(s['batch_end_time'])
        c = (min(a1, hi) - max(a0, lo)) / pd.Timedelta(hours=1)
        if c > cov:
            best, cov = cid, c
    return best


def travel(F):
    """Travel time 21.5 R☉ -> ACE vs boundary speed, with the model's residual acceleration ΔV(r)."""
    L, Vref = [float(x) for x in F['forward_model'].learned_radial_accel.physical_parameters()]
    rr = np.linspace(R0, RA, 400)
    dV = Vref * (1 - np.exp(-(rr - R0) / L))
    vg = np.arange(100, 1001, 1.0)
    return vg, np.array([np.trapz(1.0 / (v + dV), rr * RS) for v in vg]), dV[-1]


def ace_all_hours(F, rec, out):
    """Speed at ACE at every hour of the chunk (warm-up excluded): the solved field sampled at ACE's position,
    interpolated across hours without a position (as fig_reconstruction)."""
    tt = pd.Timestamp(rec['batch_start_time']) + pd.to_timedelta(rec['obs_tracks'][0]['time_obs_seconds'], unit='s')
    k0 = _transient_skip_steps(F['config'], len(tt))
    ph = rec['phi_obs_t'].cpu().numpy().astype(float); ra = rec['radius_obs_t'].cpu().numpy().astype(float)
    ok = np.isfinite(ph) & np.isfinite(ra); x = np.arange(len(ph))
    ph = np.interp(x, x[ok], np.unwrap(ph[ok])) % (2 * np.pi); ra = np.interp(x, x[ok], ra[ok])
    with torch.no_grad():
        v = sample_satellite_track(out['v_full'], F['r'], F['phi'], torch.tensor(ph, dtype=torch.float32, device=dev),
                                   torch.tensor(ra, dtype=torch.float32, device=dev), time_obs=rec['time_obs_t'],
                                   dt=F['config'].dt).cpu().numpy()
    return pd.Series(v[k0:], index=pd.DatetimeIndex(tt[k0:]))


HX = pd.read_csv('runs/part1_wsa/huxt_test.csv', parse_dates=['time']).drop_duplicates('time').set_index('time').wsa_huxt

plt.rcParams.update({'font.size': 20, 'axes.labelsize': 20, 'axes.titlesize': 20, 'xtick.labelsize': 17,
                     'ytick.labelsize': 17, 'legend.fontsize': 18})          # printed at text width: ~8-9 pt
fig, ax = plt.subplots(2, 4, figsize=(18, 9.5), gridspec_kw={'height_ratios': [1.05, 1.0]}, layout='constrained')
for c, q in CASES.iterrows():
    F = fold_of(q.obs_peak); F43 = fold_of(q.obs_peak, 43)
    vg, tau, dv_end = travel(F)
    t_img = q.obs_peak - pd.Timedelta(seconds=float(np.interp(q.obs_peak_v - dv_end, vg, tau)))
    lo, hi = t_img - HALF, q.obs_peak + HALF
    cid = chunk_for(F, lo, hi)
    rec, out = run_chunk(F, cid); rec43, out43 = run_chunk(F43, cid)

    # row 1: AIA frame
    k = (PIX.obstime - t_img).abs().idxmin()
    m = Map(PIX.path[k])
    img = np.asarray(m.data, np.float64) / float(m.meta['exptime'])
    cx, cy, r_pix, e1, e2, nn = BPF.ecliptic_basis(m)
    Z, rho2, *_ = BPF.pixel_coords(img.shape, cx, cy, r_pix, e1, e2, nn)
    med = float(np.median(img[(rho2 <= 0.98 ** 2) & np.isfinite(img)]))
    a = ax[0, c]; a.axis('off')
    ext = (-0.5, img.shape[1] - 0.5, -0.5, img.shape[0] - 0.5)
    a.imshow(np.where(rho2 < 1, np.log10(np.clip(img / med, 1e-3, None)), np.nan)[::2, ::2], origin='lower', cmap='gray',
             vmin=-1.3, vmax=0.7, extent=ext)
    a.imshow(np.where((img < 0.35 * med) & (rho2 < 1), 1.0, np.nan)[::2, ::2], origin='lower', extent=ext,
             cmap=ListedColormap([C['nn']]), alpha=0.5)
    a.set_xlim(cx - 1.05 * r_pix, cx + 1.05 * r_pix); a.set_ylim(cy - 1.05 * r_pix, cy + 1.05 * r_pix); a.set_aspect('equal')
    a.set_title(f'({"abcd"[c]}) {TITLE[q.case]}\n{pd.Timestamp(m.date.datetime):%Y-%m-%d %H:%M} UT')

    # row 2: ACE
    a = ax[1, c]
    win = lambda x: x[(x.index >= q.obs_peak - HALF) & (x.index <= q.obs_peak + HALF)]
    o = win(OBS)
    a.plot(o.index, o.to_numpy(), color=INK, lw=1.2, label='ACE measured')
    W = win(A).asfreq('1h')
    a.plot(W.index, W['recurrence'], color=C['recurrence'], lw=1.4, label=LBL['recurrence'])
    nn_all = win(pd.concat([ace_all_hours(F, rec, out), ace_all_hours(F43, rec43, out43)], axis=1).mean(axis=1))
    a.plot(nn_all.index, nn_all.to_numpy(), color=C['nn'], lw=2.4, label=LBL['nn'] + ' (mean of 2 seeds)')
    hx = win(HX).asfreq('1h')
    a.plot(hx.index, hx.to_numpy(), color=C['persistence'], lw=2.0, label='WSA 4.5 + HUXt')
    a.set_ylim(250, 850)
    a.xaxis.set_major_locator(mdates.DayLocator(interval=4)); a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    if c == 0:
        a.set_ylabel('speed at ACE (km/s)')
        h0, l0 = a.get_legend_handles_labels()
fig.legend(h0, l0, loc='outside lower center', ncol=4, handlelength=2.0, columnspacing=1.5)
save(fig, 'fig_cases')
