"""Figure 1 (Part I; also the graphical abstract): the whole computational pipeline on real data from one development test interval
(B0, fold 5, chunk 26: 2017-02-07 to 2017-05-10).

  AIA 193 Å frame -> longitude-resolved features -> NN1 (shared per-cell MLP) -> inner-boundary speed v(21.5 R☉, φ, t)
  -> differentiable propagation (snapshot of v(r, φ) in the ecliptic) -> speed at ACE vs measurement -> loss S;
  the gradient of S flows back through the solver into NN1.

  python reporting/part1/figures/fig_pipeline.py [--cuda-index 0] [--fits <AIA frame>] [--snap 2017-03-24T12]
Output: figures/part1/fig_pipeline.png/.pdf
"""
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from sunpy.coordinates import sun
from sunpy.map import Map

import build_pixel_features as BPF
from _b0_hourly import b0_hourly
from _style import BLUES, C, GRID, INK, INK2, save
from plot_best_checkpoint import load_fold, run_chunk
from frozen_model import B0_RUN
from data_paths import SOLAR

ap = argparse.ArgumentParser()
ap.add_argument('--cuda-index', type=int, default=0)
ap.add_argument('--fits', default=str(SOLAR / 'SDO/AIA/LVL1.5/hourly/2017/193/AIA20170319_1200_0193.fits'))
ap.add_argument('--snap', default='2017-03-24T12:00')
args = ap.parse_args()
dev = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')

# --- model run on the chunk: boundary and full propagated field
F = load_fold(5, f'{B0_RUN}/seed42/fold5', dev)
rec, out = run_chunk(F, 26)
cfg = F['config']
V = out['v_inner_hat'].cpu().numpy()                                  # [Nphi, Nt]
VF = out['v_full']
VF = (VF[0] if VF.dim() == 4 else VF).cpu().numpy()                    # [Nr, Nphi, Nt]
valid = rec['image_slice'][9].cpu().numpy() > 0.5
t0 = pd.Timestamp(rec['batch_start_time'])
tg = t0 + pd.to_timedelta(np.arange(V.shape[1]) * cfg.dt, unit='s')
end = pd.Timestamp(F['specs'][26]['batch_end_time'])
keep = tg < end
lon = np.arange(V.shape[0]) * 360.0 / V.shape[0]
r = F['r'].cpu().numpy()
snap = pd.Timestamp(args.snap)
isnap = int(np.argmin(np.abs(tg - snap)))
img_t = pd.Timestamp(Map(args.fits).date.datetime)

# --- inputs: channel-0 time-longitude map over the chunk
D = 'data/nn1_dataset'
times = pd.DatetimeIndex(pd.read_csv(f'{D}/times.csv', parse_dates=['time']).time)
win = (times >= tg[keep][0]) & (times < end)
F0 = np.load(f'{D}/features.npy', mmap_mode='r')[0][:, win]
MA = np.load(f'{D}/mask_aia.npy', mmap_mode='r')[:, win]
ch0 = np.where(np.asarray(MA) > 0, np.asarray(F0, np.float32), np.nan)

W = b0_hourly()
W = W[(W.time >= tg[keep][0]) & (W.time < end)].set_index('time').asfreq('1h')

fig = plt.figure(figsize=(16, 6.3))
cmap = BLUES.copy(); cmap.set_bad(GRID)
Y0, H = 0.36, 0.44                                                     # panel row (figure coordinates)
X = [0.010, 0.175, 0.345, 0.460, 0.640, 0.825]                         # panel left edges
WD = [0.135, 0.135, 0.085, 0.145, 0.150, 0.165]
TITLES = ['1  Solar image', '2  Features', '3  NN', '4  Inner boundary', '5  Propagation',
          '6  Speed at ACE']
SUB = ['SDO/AIA 193 Å; dark pixels\n(coronal holes) in blue', '6 coronal-hole features + lags,\nPFSS expansion, recurrence\n(longitude × time)',
       'one MLP shared by\nall longitudes and hours', 'v(21.5 R☉, longitude × time)\n(never observed)',
       '∂v/∂t + v ∂v/∂r + Ω ∂v/∂φ = v dΔV/dr\necliptic snapshot ' + f'{snap:%Y-%m-%d}' + ', 21.5–217 R☉', 'measured vs model;\nloss S on ICME-free hours']


def header(i):
    fig.text(X[i] + WD[i] / 2, Y0 + H + 0.10, TITLES[i], ha='center', va='bottom', fontsize=13.5, weight='bold', color=INK)
    fig.text(X[i] + WD[i] / 2, Y0 - 0.035, SUB[i], ha='center', va='top', fontsize=10.5, color=INK2, linespacing=1.3)


# 1 AIA frame
m = Map(args.fits)
img = np.asarray(m.data, np.float64) / float(m.meta['exptime'])
cx, cy, r_pix, e1, e2, nn = BPF.ecliptic_basis(m)
Z, rho2, *_ = BPF.pixel_coords(img.shape, cx, cy, r_pix, e1, e2, nn)
med = float(np.median(img[(rho2 <= 0.98 ** 2) & np.isfinite(img)]))
s = 2
a = fig.add_axes([X[0], Y0, WD[0], H * 1.2]); a.axis('off')
ext = (-0.5, img.shape[1] - 0.5, -0.5, img.shape[0] - 0.5)
a.imshow(np.where(rho2 < 1, np.log10(np.clip(img / med, 1e-3, None)), np.nan)[::s, ::s], origin='lower', cmap='gray',
         vmin=-1.3, vmax=0.7, extent=ext)
a.imshow(np.where((img < 0.35 * med) & (rho2 < 1), 1.0, np.nan)[::s, ::s], origin='lower', extent=ext,
         cmap=ListedColormap([C['nn']]), alpha=0.5)
a.set_xlim(cx - 1.05 * r_pix, cx + 1.05 * r_pix); a.set_ylim(cy - 1.05 * r_pix, cy + 1.05 * r_pix); a.set_aspect('equal')
a.text(0.5, -0.02, f'{img_t:%Y-%m-%d %H:%M} UT', transform=a.transAxes, ha='center', va='top', fontsize=10, color=INK2)
header(0)

# 2 features (time-longitude, channel 0)
a = fig.add_axes([X[1], Y0 + 0.03, WD[1], H * 1.1])
tx = mdates.date2num(times[win].to_pydatetime())
a.imshow(np.ma.masked_invalid(ch0), aspect='auto', origin='lower', cmap=cmap, extent=[tx[0], tx[-1], 0, 360],
         interpolation='nearest', vmin=0, vmax=np.nanpercentile(ch0, 99))
a.axvline(mdates.date2num(img_t.to_pydatetime()), color=C['nn2'], lw=1.2, ls='--')
a.xaxis_date(); a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a.set_yticks([0, 180, 360], ['', '', '']); a.tick_params(labelsize=9); a.grid(False)
header(1)

# 3 NN1 schematic
a = fig.add_axes([X[2], Y0 + 0.03, WD[2], H * 1.1]); a.axis('off'); a.set_xlim(0, 1); a.set_ylim(0, 1)
layers = [6, 5, 4, 3, 1]
xs = np.linspace(0.08, 0.92, len(layers))
pos = [[(x, y) for y in (np.linspace(0.12, 0.88, n) if n > 1 else [0.5])] for x, n in zip(xs, layers)]
for L1, L2 in zip(pos[:-1], pos[1:]):
    for p in L1:
        for q in L2:
            a.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.6, zorder=1)
for k, L in enumerate(pos):
    for p in L:
        a.plot(*p, 'o', ms=7, color=C['nn2'] if 0 < k < len(pos) - 1 else C['nn'], mec='white', mew=0.8, zorder=2)
a.text(0.5, -0.02, '37 → 100 → 64 → 32 → 1', transform=a.transAxes, ha='center', va='top', fontsize=10, color=INK2)
header(2)

# 4 inner boundary
a = fig.add_axes([X[3], Y0 + 0.03, WD[3], H * 1.1])
pc = a.pcolormesh(tg[keep], lon, np.ma.masked_where(~valid[:, keep], V[:, keep]), cmap=cmap, vmin=300, vmax=700,
                  shading='auto', rasterized=True)
L0 = sun.L0(Time(tg[keep][::12].to_pydatetime())).deg
for sg in np.split(np.arange(len(L0)), np.flatnonzero(np.abs(np.diff(L0)) > 180) + 1):
    a.plot(tg[keep][::12][sg], L0[sg], color=INK, lw=0.6)
a.axvline(snap, color=C['nn2'], lw=1.2, ls='--')
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a.set_yticks([0, 180, 360], ['', '', '']); a.tick_params(labelsize=9); a.grid(False); a.set_ylim(0, 360)
header(3)

# 5 propagated field, polar snapshot (Carrington frame, Sun at the centre)
a = fig.add_axes([X[4], Y0 - 0.02, WD[4], H * 1.3], projection='polar')
th = np.deg2rad(np.append(lon, 360.0))
Vs = np.vstack([VF[:, :, isnap].T, VF[:, :1, isnap].T])                 # [Nphi+1, Nr]
a.pcolormesh(th, r, Vs.T, cmap=cmap, vmin=300, vmax=700, shading='auto', rasterized=True)
L0s = float(sun.L0(Time(tg[isnap].to_pydatetime())).deg)
a.plot(np.deg2rad(L0s), 214.0, 'o', ms=8, color=C['persistence'], mec='white', mew=1.0)
a.annotate('ACE', (np.deg2rad(L0s), 214.0), xytext=(6, 6), textcoords='offset points', fontsize=10.5, color=INK, weight='bold')
a.set_ylim(0, 217); a.set_yticks([]); a.set_xticks([]); a.grid(False); a.spines['polar'].set_color(INK2)
a.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 21.5), color=INK, lw=0.8)
header(4)
cax = fig.add_axes([X[3] + 0.03, Y0 - 0.185, X[4] + WD[4] - X[3] - 0.06, 0.022])
cb = fig.colorbar(pc, cax=cax, orientation='horizontal'); cb.set_label('solar-wind speed (km/s)', fontsize=10.5)
cb.ax.tick_params(labelsize=9)

# 6 ACE
a = fig.add_axes([X[5], Y0 + 0.03, WD[5], H * 1.1])
a.plot(W.index, W.obs, color=INK, lw=0.8, label='ACE measured')
a.plot(W.index, W.nn, color=C['nn'], lw=1.5, label='model')
a.axvline(snap, color=C['nn2'], lw=1.2, ls='--')
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a.tick_params(labelsize=9); a.set_ylim(250, 800); a.yaxis.tick_right(); a.text(1.0, 1.01, 'km/s', transform=a.transAxes, ha='right', va='bottom', fontsize=10, color=INK2)
a.legend(fontsize=9, loc='upper left', ncol=2)
header(5)

# arrows
yc = Y0 + H * 0.6
for i in range(5):
    x0, x1 = X[i] + WD[i] + 0.004, X[i + 1] - 0.006
    fig.add_artist(FancyArrowPatch((x0, yc), (x1, yc), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=16, lw=1.6, color=INK2))
xg0, xg1, yt, yb = X[5] + WD[5] * 0.5, X[2] + WD[2] * 0.5, Y0 - 0.15, 0.085
fig.add_artist(plt.Line2D([xg0, xg0, xg1], [yt, yb, yb], transform=fig.transFigure, color=C['nn2'], lw=1.6, ls=(0, (4, 3))))
fig.add_artist(FancyArrowPatch((xg1, yb), (xg1, yt), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=16,
                               lw=1.6, color=C['nn2'], linestyle=(0, (4, 3))))
fig.text((xg0 + xg1) / 2, 0.045, 'end-to-end training: the gradient of the loss S at ACE flows back through the solver into '
         'the NN weights', ha='center', va='center', fontsize=12.5, color=C['nn2'], weight='bold')
fig.text(0.5, 0.985, 'Learning the solar-wind inner boundary from solar images through a differentiable heliospheric model',
         ha='center', va='top', fontsize=16, color=INK, weight='bold')
save(fig, 'fig_pipeline')
