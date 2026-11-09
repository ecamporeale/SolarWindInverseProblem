"""Part I figure `fig_magnetic_inputs` (proposed 2026-10-23, after fig_aia_features): the magnetic inputs, on the hour of fig_aia_features (2017-01-03 04:00 UT).

(a) HMI hmi.M_720s at the hour: B_r = B_los / μ on the disk (μ ≥ cos 65°), with the region of the two HMI features of
    the linear comparator (|Δlon| ≤ 7.5°, |β| ≤ 30°) and its weak-field pixels (|B_r| < 10 G). Feature values from
    pipeline/p1_data/build_pixel_features.hmi_frame (the code that built them).
(b) The GONG mrzqs synoptic map used for B0's input at that hour (the most recent map at or before it), B_r in
    Carrington longitude and sine latitude; shaded: photospheric footpoints of open field in the PFSS extrapolation
    (sunkit-magex, R_ss = 2.5 R_sun, 30 radial points; traced from a 2° grid); lines: the field lines traced down from
    the source surface at Earth's latitude B0, every 4th of the 128 model longitudes, with their photospheric
    footpoints; the sub-Earth longitude of the AIA hour is marked.
(c) B0's input: log10 f_s along the 128 model longitudes, from pipeline/p1_data/pfss_fs_grid_hourly.map_fs (the code
    that built it) and checked against data/nn1_dataset/pfss_log10fs_hourly_nphi128.npy; open circles: longitudes
    whose line is closed and which take the value of the nearest open longitude.

  python reporting/part1/figures/fig_magnetic_inputs.py
Output: figures/part1/fig_magnetic_inputs.png/.pdf
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunkit_magex.pfss as pfss
import sunpy.map
from astropy.coordinates import SkyCoord
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sunpy.coordinates import sun

import build_pixel_features as BPF
import pfss_fs_grid_hourly as PF
from _style import C, INK, INK2, save
from data_paths import SOLAR

T = pd.Timestamp('2017-01-03 04:00')
HMI = str(SOLAR / 'SDO/HMI/fits/2017/01/03/hmi.M_720s.20170103_040000_TAI.fits')
WEAK, OPEN, LINE, EARTH = '#eb6834', '#cfe3f7', '#eda100', '#1baf7a'

# ---- B0's input at the hour: the most recent GONG map, its log10 f_s, and the stored training input
M = pd.read_csv('data/nn1_dataset/pfss_log10fs_hourly_maps.csv', parse_dates=['map_time'])
row = M[M.map_time <= T].iloc[-1]
_, fs, status, n_open = PF.map_fs((row['map'], row.map_time))
assert status == 'ok', status
times = pd.to_datetime(pd.read_csv('data/nn1_dataset/times.csv', usecols=['time'])['time'])
stored = np.load('data/nn1_dataset/pfss_log10fs_hourly_nphi128.npy', mmap_mode='r')[:, int(np.flatnonzero(times == T)[0])]
assert np.allclose(fs, stored, atol=1e-5), 'recomputed log10 f_s differs from the stored B0 input'

# ---- PFSS for panel (b)
gm = sunpy.map.Map(row['map']).resample([360, 180] * u.pix)
pout = pfss.pfss(pfss.Input(gm, PF.NRHO, PF.RSS))
b0 = float(sun.B0(row.map_time).to_value(u.deg))
tracer = pfss.tracing.PerformanceTracer()
seeds = SkyCoord(PF.LONS * u.deg, np.full(PF.NPHI, b0) * u.deg, (PF.RSS - 0.01) * u.R_sun, frame=pout.coordinate_frame)
fls = tracer.trace(seeds, pout)
is_open = np.array([fl.is_open for fl in fls])
glon, glat = np.meshgrid(np.arange(1, 360, 2.0), np.rad2deg(np.arcsin(np.linspace(-0.99, 0.99, 90))))
foot = SkyCoord(glon.ravel() * u.deg, glat.ravel() * u.deg, 1.01 * u.R_sun, frame=pout.coordinate_frame)
open_ph = np.array([fl.is_open for fl in tracer.trace(foot, pout)]).reshape(glat.shape)
L0 = float(sun.L0(T).to_value(u.deg))                                  # Carrington longitude of the sub-Earth point

# ---- HMI for panel (a)
hm = sunpy.map.Map(HMI)
hfeat = BPF.hmi_frame(HMI, None)
Blos = np.asarray(hm.data, np.float64)
cx, cy, r_pix, e1, e2, nn = BPF.ecliptic_basis(hm)
Z, rho2, ri, ci, ingrid = BPF.pixel_coords(Blos.shape, cx, cy, r_pix, e1, e2, nn)
vis = ingrid & np.isfinite(Blos) & (Z >= BPF.MU_MIN_HMI)
with np.errstate(invalid='ignore', divide='ignore'):
    Br = np.where(vis, Blos / Z, np.nan)
lat_pix, lon_pix = np.full(Blos.shape, np.nan), np.full(Blos.shape, np.nan)
lat_pix[ingrid], lon_pix[ingrid] = BPF.LATS[ri[ingrid]], BPF.DLONS[ci[ingrid]]
region = vis & (np.abs(lon_pix) <= 7.5) & (np.abs(lat_pix) <= 30)
weak = region & (np.abs(Br) < 10)


def to_pix(beta, dlon):
    b, l = np.deg2rad(beta), np.deg2rad(dlon)
    v = np.outer(np.cos(b) * np.cos(l), e1) + np.outer(np.cos(b) * np.sin(l), e2) + np.outer(np.sin(b), nn)
    return cx + r_pix * v[:, 0], cy + r_pix * v[:, 1]


fig = plt.figure(figsize=(14.5, 6.0), layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[2.0, 1])

# (a) HMI
a = fig.add_subplot(gs[:, 0])
s = 4
ext = (-0.5, Blos.shape[1] - 0.5, -0.5, Blos.shape[0] - 0.5)
a.imshow(np.where(rho2 < 1, 0.0, np.nan)[::s, ::s], origin='lower', cmap=ListedColormap(['#e9e8e4']), extent=ext)
a.imshow(Br[::s, ::s], origin='lower', cmap='gray', vmin=-40, vmax=40, extent=ext)
a.imshow(np.where(weak, 1.0, np.nan)[::s, ::s], origin='lower', cmap=ListedColormap([WEAK]), alpha=0.35, extent=ext)
for bnd in ((7.5, 30),):
    w, k = bnd
    edge = np.r_[np.c_[np.linspace(-k, k, 60), np.full(60, -w)], np.c_[np.full(30, k), np.linspace(-w, w, 30)],
                 np.c_[np.linspace(k, -k, 60), np.full(60, w)], np.c_[np.full(30, -k), np.linspace(w, -w, 30)]]
    x, y = to_pix(edge[:, 0], edge[:, 1])
    a.plot(x, y, color=C['nn'], lw=1.6)
th = np.linspace(0, 2 * np.pi, 400)
rmu = r_pix * np.sin(np.arccos(BPF.MU_MIN_HMI))
a.plot(cx + rmu * np.cos(th), cy + rmu * np.sin(th), color=INK2, lw=0.8, ls='--')
a.set_xlim(cx - 1.04 * r_pix, cx + 1.04 * r_pix); a.set_ylim(cy - 1.04 * r_pix, cy + 1.04 * r_pix)
a.set_aspect('equal'); a.axis('off')
a.set_title(f'(a) HMI $B_r = B_{{los}}/\\mu$, {T:%Y-%m-%d %H:%M} UT\n'
            f'weak-field fraction (|$B_r$| < 10 G) {hfeat["H1_weakfrac<10G w=7.5 k=30"]:.2f}; '
            f'mean |$B_r$| {hfeat["H1_meanabs w=7.5 k=30"]:.1f} G', fontsize=11)

# (b) GONG synoptic map, open-field footpoints, traced lines
b = fig.add_subplot(gs[0, 1])
b.grid(False)
lon_e = np.linspace(0, 360, gm.data.shape[1] + 1)
slat_e = np.linspace(-1, 1, gm.data.shape[0] + 1)
b.pcolormesh(lon_e, slat_e, gm.data, cmap='gray', vmin=-20, vmax=20, rasterized=True)
b.contourf(glon, np.sin(np.deg2rad(glat)), open_ph.astype(float), levels=[0.5, 1.5], colors=[OPEN], alpha=0.55)
b.contour(glon, np.sin(np.deg2rad(glat)), open_ph.astype(float), levels=[0.5], colors=[C['nn']], linewidths=0.7)
for j in range(0, PF.NPHI, 4):
    fl = fls[j]
    if not fl.is_open:
        continue
    c = fl.coords
    lo, la = np.asarray(c.lon.to_value(u.deg)), np.sin(np.asarray(c.lat.to_value(u.rad)))
    brk = np.r_[0, np.flatnonzero(np.abs(np.diff(lo)) > 180) + 1, len(lo)]
    for i0, i1 in zip(brk[:-1], brk[1:]):
        b.plot(lo[i0:i1], la[i0:i1], color=LINE, lw=1.3)
    k_ph = 0 if c.radius[0] < c.radius[-1] else -1                  # photospheric end of the line
    b.plot(lo[k_ph], la[k_ph], 'o', ms=3.5, mfc=LINE, mec=INK, mew=0.5)
b.axhline(np.sin(np.deg2rad(b0)), color=INK2, lw=0.8, ls=':')
b.axvline(L0, color=EARTH, lw=1.6)
b.set_xlim(0, 360); b.set_ylim(-1, 1)
b.set_xticks(np.arange(0, 361, 60)); b.tick_params(labelbottom=False)
b.set_yticks(np.sin(np.deg2rad([-60, -30, 0, 30, 60]))); b.set_yticklabels(['−60', '−30', '0', '30', '60'])
b.set_ylabel('latitude (°)')
b.set_title(f'(b) GONG {row.map_time:%Y-%m-%d %H:%M} UT, $B_r$ (±20 G); PFSS open-field footpoints; '
            f'lines from the source surface at $B_0$ = {b0:.1f}°', fontsize=11)

# (c) B0's input along the model longitudes
c_ = fig.add_subplot(gs[1, 1], sharex=b)
c_.plot(PF.LONS, fs, color=INK, lw=1.0, zorder=1)
c_.scatter(PF.LONS[is_open], fs[is_open], color=INK, s=10, zorder=2)
c_.scatter(PF.LONS[~is_open], fs[~is_open], s=16, facecolors='white', edgecolors=INK2, zorder=2)
j_e = int(np.argmin(np.abs((PF.LONS - L0 + 180) % 360 - 180)))
c_.axvline(L0, color=EARTH, lw=1.6)
c_.annotate(f'sub-Earth: {fs[j_e]:.2f}', (L0, fs[j_e]), xytext=(-10, -16), textcoords='offset points', fontsize=10,
            color=EARTH, ha='right', weight='bold')
c_.set_xlim(0, 360); c_.set_xticks(np.arange(0, 361, 60))
c_.set_xlabel('Carrington longitude (°)'); c_.set_ylabel('log$_{10}$ $f_s$')
c_.set_title(f'(c) Model input log$_{{10}}$ $f_s$ at the 128 model longitudes ({n_open} open)', fontsize=11)

handles = [Patch(color=WEAK, alpha=0.35, label='(a) weak-field pixels (|$B_r$| < 10 G) in the feature region'),
           plt.Line2D([], [], color=C['nn'], lw=1.6, label='(a) region |Δlon| ≤ 7.5°, |β| ≤ 30°'),
           plt.Line2D([], [], color=INK2, lw=0.8, ls='--', label='(a) μ = cos 65°'),
           Patch(fc=OPEN, ec=C['nn'], lw=0.7, label='(b) open-field footpoints'),
           plt.Line2D([], [], color=LINE, lw=1.3, marker='o', ms=3.5, mec=INK, mew=0.5,
                      label='(b) field lines traced down from the source surface (every 4th)'),
           plt.Line2D([], [], color=EARTH, lw=1.6, label='(b, c) sub-Earth longitude'),
           plt.Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, label='(c) closed line: nearest open value')]
fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=10)
save(fig, 'fig_magnetic_inputs')
