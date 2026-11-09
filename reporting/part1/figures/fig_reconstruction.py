"""Part I figure `fig_reconstruction` (manuscript label fig:reconstruction; Figure 7 in the 2026-10 draft): what the model reconstructs.

(a) NN1 inner-boundary speed v(21.5 R☉) per Carrington longitude and hour for one development test chunk
    (B0 seed 42, fold 5, chunk 26: 2017-02-07 to 2017-05-10, declining phase); grey = no AIA input (fill speed);
    black = sub-Earth Carrington longitude.
(b) Speed at ACE over the same interval: measured (ICME-masked), B0 (mean of seeds 42 and 43, chunk evaluation),
    the linear set-2a model and the 27-day recurrence.
(c) Distribution of the speeds over all matched test hours (5 development test batches): observed vs each method.
Data: <B0_RUN>/seed{42,43}/fold*/{best.pt,test_predictions.csv}; the baselines as in
      pipeline/p1_model/eval_matched_baselines.py.

  python reporting/part1/figures/fig_reconstruction.py [--cuda-index 0]       (panel a runs the checkpoint; a GPU is faster)
Output: figures/part1/fig_reconstruction.png/.pdf
"""
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from sunpy.coordinates import sun

from _b0_hourly import b0_hourly
from _style import BLUES, C, GRID, INK, LBL, save
from plot_best_checkpoint import load_fold, run_chunk
from frozen_model import B0_RUN

ap = argparse.ArgumentParser()
ap.add_argument('--cuda-index', type=int, default=0)
args = ap.parse_args()
dev = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')

F = load_fold(5, f'{B0_RUN}/seed42/fold5', dev)
cid = 26
rec, out = run_chunk(F, cid)
cfg = F['config']
V = out['v_inner_hat'].cpu().numpy()
valid = rec['image_slice'][9].cpu().numpy() > 0.5
t0 = pd.Timestamp(rec['batch_start_time'])
tg = t0 + pd.to_timedelta(np.arange(V.shape[1]) * cfg.dt, unit='s')
end = pd.Timestamp(F['specs'][cid]['batch_end_time'])
keep = tg < end
V, valid, tg = V[:, keep], valid[:, keep], tg[keep]
lon = np.arange(V.shape[0]) * 360.0 / V.shape[0]
L0 = sun.L0(Time(tg[::12].to_pydatetime())).deg

D = b0_hourly()
W = D[(D.time >= tg[0]) & (D.time < end)]
M = D.dropna(subset=['linear', 'recurrence'])

fig = plt.figure(figsize=(14, 9.5), layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1.3, 1])
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[1, 0], sharex=a); c = fig.add_subplot(gs[:, 1])
cmap = BLUES.copy(); cmap.set_bad(GRID)
pc = a.pcolormesh(tg, lon, np.ma.masked_where(~valid, V), cmap=cmap, vmin=300, vmax=700, shading='auto', rasterized=True)
seg = np.flatnonzero(np.abs(np.diff(L0)) > 180) + 1
for s in np.split(np.arange(len(L0)), seg):
    a.plot(tg[::12][s], L0[s], color=INK, lw=0.8)
a.set_ylabel('Carrington longitude (°)'); a.set_ylim(0, 360); a.set_yticks([0, 90, 180, 270, 360]); a.grid(False)
cb = fig.colorbar(pc, ax=a, pad=0.01); cb.set_label('boundary speed at 21.5 R☉ (km/s)')
a.set_title('(a) Inner boundary (reference model, seed 42, fold 5)')
W = W.set_index('time').asfreq('1h')                         # missing or masked hours shown as breaks
# the model predicts at every hour of the ACE track; only the scoring skips masked hours: mean of the seed-42 and
# seed-43 fold-5 models over the whole scored span (warm-up excluded), drawn continuously
from plot_best_checkpoint import _transient_skip_steps
from helpers import sample_satellite_track
P = []
F43 = load_fold(5, f'{B0_RUN}/seed43/fold5', dev)
for Fs, (r_, o_) in ((F, (rec, out)), (F43, run_chunk(F43, cid))):
    tt = pd.Timestamp(r_['batch_start_time']) + pd.to_timedelta(r_['obs_tracks'][0]['time_obs_seconds'], unit='s')
    k0 = _transient_skip_steps(Fs['config'], len(tt))
    # ACE's position is missing where its speed is: interpolate the (smooth) position across the gaps and sample the
    # solved field there, so that the model is drawn at every hour (NN2 is disabled: ace_final = the solver at ACE)
    ph = r_['phi_obs_t'].cpu().numpy().astype(float); ra = r_['radius_obs_t'].cpu().numpy().astype(float)
    ok = np.isfinite(ph) & np.isfinite(ra); x = np.arange(len(ph))
    ph = np.interp(x, x[ok], np.unwrap(ph[ok])) % (2 * np.pi); ra = np.interp(x, x[ok], ra[ok])
    with torch.no_grad():
        v_all = sample_satellite_track(o_['v_full'], Fs['r'], Fs['phi'], torch.tensor(ph, dtype=torch.float32, device=dev),
                                       torch.tensor(ra, dtype=torch.float32, device=dev), time_obs=r_['time_obs_t'],
                                       dt=Fs['config'].dt).cpu().numpy()
    sc = o_['ace_final'].cpu().numpy(); fin = np.isfinite(sc)
    assert np.allclose(v_all[fin], sc[fin], atol=0.05), 'resampled track differs from the scored one'
    P.append(pd.Series(v_all[k0:], index=pd.DatetimeIndex(tt[k0:])))
nn_all = pd.concat(P, axis=1).mean(axis=1)
nn_all = nn_all[(nn_all.index >= W.index[0]) & (nn_all.index <= W.index[-1])]
ic = pd.read_csv('data/HELIO4CAST_ICMECAT_v23.csv', parse_dates=['icme_start_time', 'mo_end_time'])
ic = ic[ic.sc_insitu.str.upper() == 'WIND']                   # the ICMEs masked in the ACE target (L1)
for c_ in ('icme_start_time', 'mo_end_time'):
    ic[c_] = pd.to_datetime(ic[c_], utc=True).dt.tz_convert(None)
for r_ in ic[(ic.mo_end_time >= W.index[0]) & (ic.icme_start_time <= W.index[-1])].itertuples():
    b.axvspan(r_.icme_start_time, r_.mo_end_time, color=GRID, alpha=0.9, lw=0, zorder=0)
b.plot(W.index, W.obs, color=INK, lw=1.0, label='ACE (measured)')
for k in ('recurrence', 'linear'):
    b.plot(W.index, W[k], color=C[k], lw=1.1, label=LBL[k])
b.plot(nn_all.index, nn_all.to_numpy(), color=C['nn'], lw=2.0, label=LBL['nn'] + ' (mean of 2 seeds)')
b.fill_between([], [], [], color=GRID, label='ICME, masked (not scored)')
b.set_ylabel('speed at ACE (km/s)'); b.legend(fontsize=10, loc='upper left', ncol=2)
b.xaxis.set_major_locator(mdates.MonthLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
b.set_title('(b) Out-of-fold speed at ACE')
bins = np.arange(250, 860, 15)
c.hist(M.obs, bins=bins, orientation='horizontal', color=GRID, label='ACE observed')
for k in ('nn', 'linear', 'recurrence'):
    amp = M[k].std() / M.obs.std()
    c.hist(M[k], bins=bins, orientation='horizontal', histtype='step', lw=1.6, color=C[k],
           label=f'{LBL[k]} (σ ratio {amp:.2f})')
c.set_xlabel('hours'); c.set_ylabel('speed (km/s)'); c.legend(fontsize=9.5, loc='upper center', bbox_to_anchor=(0.45, -0.1))
c.set_title(f'(c) Distribution, all matched test hours\n     (n = {len(M):,}, 5 test batches)')
save(fig, 'fig_reconstruction')
