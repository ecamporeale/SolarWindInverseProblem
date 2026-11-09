"""Part I figure `fig_aia_features` (manuscript label fig:features; Figure 2 in the 2026-10 draft): what each of the six AIA 193 Å inputs of B0 measures, on one frame.

(a) The AIA frame, log10(I / disk median), with the two dark levels used by the six features: I < 0.30 × median
    (channel 0) and 0.30-0.35 × median (counted by the f = 0.35 features); the outer strip |Δlon| ≤ 20° and the
    ecliptic.
(b)-(f) One panel per channel of Table tab:channels (channels 2 and 5, which differ only in strip width, share panel f), in the (Δlon, β) cells (0.5°) in which the features are computed.
    Cells are shown dark when most of their pixels are below the threshold; the channel's region is outlined and the
    cells it counts are drawn in full colour, the rest faded. The title gives the feature's value for this frame, from
    pipeline/p1_data/build_pixel_features.aia_frame (the code that built the training inputs).
    Channel 3 colours the dark pixels by their mean log10(I / median) per cell; channel 4 shows the coronal-hole mask
    (connected cells with dark fraction ≥ 0.5 at f = 0.35, ≥ 20 cells) and the distance from (β 0°, Δlon +10°) to its
    nearest edge (positive inside a hole).
The limb cut (μ ≥ cos 78°) never binds within |Δlon| ≤ 20°, |β| ≤ 70°, so it is not drawn.

  python reporting/part1/figures/fig_aia_features.py [--fits <AIA 193 level-1.5 FITS>]
Output: figures/part1/fig_aia_features.png/.pdf
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage
from sunpy.map import Map

import build_pixel_features as BPF
from _style import INK, INK2, save
from data_paths import SOLAR

ap = argparse.ArgumentParser()
ap.add_argument('--fits', default=str(SOLAR / 'SDO/AIA/LVL1.5/hourly/2017/193/AIA20170103_0400_0193.fits'))
args = ap.parse_args()

F0, F1 = 0.30, BPF.F_REF                                              # the two thresholds of the six features
DARK0, DARK1, DISK, FADE = '#0d366b', '#6da7ec', '#e9e8e4', 0.22      # I < 0.30 med; 0.30-0.35 med; on-disk cells
C_DIST, HOLE = '#eb6834', '#b3b0a8'                                   # channel 4: point and distance; hole mask

feat = BPF.aia_frame(args.fits, None)                                 # degradation factor cancels in I / median
m = Map(args.fits)
img = np.asarray(m.data, np.float64) / float(m.meta['exptime'])
cx, cy, r_pix, e1, e2, nn = BPF.ecliptic_basis(m)
Z, rho2, ri, ci, ingrid = BPF.pixel_coords(img.shape, cx, cy, r_pix, e1, e2, nn)
med = feat['disk_median']
use = ingrid & np.isfinite(img) & (Z >= BPF.MU_MIN_AIA)
idx = (ri * BPF.NC + ci)[use]
I = img[use]
logr = np.log10(np.clip(I, 1e-3 * med, None) / med)
n = BPF.cellsum(idx)
d0, d1 = BPF.cellsum(idx, (I < F0 * med).astype(float)), BPF.cellsum(idx, (I < F1 * med).astype(float))
sdl = BPF.cellsum(idx, np.where(I < F1 * med, logr, 0.0))
with np.errstate(invalid='ignore', divide='ignore'):
    f0, f1 = d0 / np.maximum(n, 1), d1 / np.maximum(n, 1)
    mean_dark_log = np.where(d1 > 0, sdl / np.maximum(d1, 1), np.nan)
ondisk = n > 0
level = np.where(~ondisk, np.nan, np.where(f0 >= 0.5, 2, np.where(f1 >= 0.5, 1, 0)))   # 0 bright, 1 0.30-0.35, 2 < 0.30
holes = BPF.hole_mask(ondisk & (f1 >= 0.5))
L, B = np.meshgrid(BPF.DLONS, BPF.LATS)

CH = [  # channel, name, definition, feature key, value format, region (half-width, |β| max), levels counted
    (0, 'dark area', 'f 0.30, ±20°', 'A_area f=0.3 w=20.0 off=+0', '{:.1f} %', (20.0, 70), (2,)),
    (1, 'ecliptic band', 'f 0.35, ±7.5°, |β| ≤ 15°', 'B_eclband |lat|<=15', '{:.1f} %', (7.5, 15), (1, 2)),
    (3, 'darkness', 'f 0.35, ±7.5°, |β| ≤ 30°', 'D_mean log(I/med) of dark pixels, band |lat|<=30', '{:.2f} dex',
     (7.5, 30), (1, 2)),
    (4, 'boundary distance', 'at β 0°, Δlon +10°', 'B_dchb signed dist off=+10', '{:+.1f}°', None, ()),
    (5, 'dark area, f 0.35', '±7.5° (5, ESWF, solid)', 'A_area f=0.35 w=7.5 off=+0', '{:.1f} %', (7.5, 70), (1, 2)),
]
CH2 = ('±12.5° (2, dashed)', 'A_area f=0.35 w=12.5 off=+0', (12.5, 70))   # drawn in channel 5's panel


def to_pix(beta, dlon):
    """(β, Δlon) in degrees -> image pixel (x, y), front hemisphere."""
    b, l = np.deg2rad(beta), np.deg2rad(dlon)
    v = np.outer(np.cos(b) * np.cos(l), e1) + np.outer(np.cos(b) * np.sin(l), e2) + np.outer(np.sin(b), nn)
    return cx + r_pix * v[:, 0], cy + r_pix * v[:, 1]


def draw_cells(a, region, counted):
    """Grey disk, the two dark levels; inside `region` the counted levels in full colour, everything else faded."""
    a.pcolormesh(BPF.DLONS, BPF.LATS, np.where(ondisk, 1.0, np.nan), cmap=ListedColormap([DISK]), shading='nearest',
                 rasterized=True)
    inside = np.ones_like(ondisk) if region is None else (np.abs(L) <= region[0]) & (np.abs(B) <= region[1])
    for lev, col in ((1, DARK1), (2, DARK0)):
        on = level == lev
        full = on & inside & (lev in counted)
        a.pcolormesh(BPF.DLONS, BPF.LATS, np.where(on & ~full, 1.0, np.nan), cmap=ListedColormap([col]), alpha=FADE,
                     shading='nearest', rasterized=True)
        a.pcolormesh(BPF.DLONS, BPF.LATS, np.where(full, 1.0, np.nan), cmap=ListedColormap([col]), shading='nearest',
                     rasterized=True)
    if region is not None:
        a.add_patch(Rectangle((-region[0], -region[1]), 2 * region[0], 2 * region[1], fill=False, ec=INK, lw=1.3))


fig = plt.figure(figsize=(14.0, 5.8), layout='constrained')
gs = fig.add_gridspec(1, 6, width_ratios=[2.6] + [1] * 5)

# (a) the disk
a = fig.add_subplot(gs[0])
s = 2
lr = np.log10(np.clip(img / med, 1e-3, None))[::s, ::s]
ext = (-0.5, img.shape[1] - 0.5, -0.5, img.shape[0] - 0.5)
on_disk = rho2[::s, ::s] < 1.0
a.imshow(np.where(on_disk, lr, np.nan), origin='lower', cmap='gray', vmin=-1.3, vmax=0.7, extent=ext)
lev_pix = np.where(img < F0 * med, 2.0, np.where(img < F1 * med, 1.0, np.nan))[::s, ::s]
a.imshow(np.where(on_disk & (lev_pix == 1), 1.0, np.nan), origin='lower', cmap=ListedColormap([DARK1]), alpha=0.75, extent=ext)
a.imshow(np.where(on_disk & (lev_pix == 2), 1.0, np.nan), origin='lower', cmap=ListedColormap([DARK0]), alpha=0.85, extent=ext)
bb = np.linspace(-70, 70, 200)
for sg in (-1, 1):
    x, y = to_pix(bb, np.full_like(bb, sg * 20.0))
    a.plot(x, y, color='white', lw=1.4, ls='--')
ll = np.linspace(-89, 89, 200)
x, y = to_pix(np.zeros_like(ll), ll)
a.plot(x, y, color='white', lw=0.9, ls=':')
a.set_xlim(cx - 1.04 * r_pix, cx + 1.04 * r_pix); a.set_ylim(cy - 1.04 * r_pix, cy + 1.04 * r_pix)
a.set_aspect('equal'); a.axis('off')
a.set_title(f'(a) AIA 193 Å, {m.date.datetime:%Y-%m-%d %H:%M} UT', fontsize=12)

# (b)-(g) one panel per channel
for k, (ch, name, defn, key, fmt, region, counted) in enumerate(CH):
    a = fig.add_subplot(gs[k + 1])
    a.grid(False)
    if ch == 3:
        draw_cells(a, region, ())
        inside = (np.abs(L) <= region[0]) & (np.abs(B) <= region[1]) & (level >= 1)
        pc = a.pcolormesh(BPF.DLONS, BPF.LATS, np.where(inside, mean_dark_log, np.nan), cmap='cividis',
                          vmin=-1.3, vmax=np.log10(F1), shading='nearest', rasterized=True)
        cb = fig.colorbar(pc, ax=a, location='bottom', shrink=0.9, pad=0.02, aspect=12)
        cb.set_label('log₁₀(I/med)', fontsize=9.5); cb.ax.tick_params(labelsize=8.5)
    elif ch == 4:
        i0, j0 = BPF.I_LAT0, int(np.argmin(np.abs(BPF.DLONS - 10.0)))
        inside_hole = bool(holes[i0, j0])
        _, (ii, jj) = ndimage.distance_transform_edt(holes if inside_hole else ~holes, sampling=BPF.STEP,
                                                     return_indices=True)
        seg = ([10.0, BPF.DLONS[jj[i0, j0]]], [0.0, BPF.LATS[ii[i0, j0]]])
        zoom = a.inset_axes([0.42, 0.66, 0.56, 0.32])                # empty upper part of the panel
        for ax_, ms in ((a, 4.5), (zoom, 6.5)):
            ax_.pcolormesh(BPF.DLONS, BPF.LATS, np.where(ondisk, 1.0, np.nan), cmap=ListedColormap([DISK]),
                           shading='nearest', rasterized=True)
            ax_.pcolormesh(BPF.DLONS, BPF.LATS, np.where(holes, 1.0, np.nan), cmap=ListedColormap([HOLE]),
                           alpha=1.0, shading='nearest', rasterized=True)
            ax_.contour(BPF.DLONS, BPF.LATS, holes.astype(float), levels=[0.5], colors=INK, linewidths=0.8)
            ax_.plot(*seg, color=C_DIST, lw=2.2, solid_capstyle='butt')
            ax_.plot(10.0, 0.0, 'o', ms=ms, mfc='white', mec=C_DIST, mew=1.6, zorder=5)
        zoom.set_xlim(4, 16); zoom.set_ylim(-4, 4); zoom.set_aspect('equal'); zoom.grid(False)
        zoom.set_xticks([]); zoom.set_yticks([])
        for sp in zoom.spines.values():
            sp.set_visible(True); sp.set_color(INK2)
        a.indicate_inset_zoom(zoom, edgecolor=INK2)
    elif ch == 5:                                                # channels 2 and 5 differ only in the strip width
        draw_cells(a, CH2[2], counted)
        a.patches[-1].set_linestyle('--')
        a.add_patch(Rectangle((-region[0], -region[1]), 2 * region[0], 2 * region[1], fill=False, ec=INK, lw=1.3))
    else:
        draw_cells(a, region, counted)
    val = feat[key]
    if ch == 5:
        title = f'(f) 2, 5: {name}\n{defn}: {fmt.format(val)}\n{CH2[0]}: {fmt.format(feat[CH2[1]])}'
    else:
        title = f'({"bcdef"[k]}) {ch}: {name}\n{defn}\n' + fmt.format(val).replace('-', '−')
    a.set_title(title, fontsize=9.5, loc='left')
    a.set_xlim(-25, 25); a.set_ylim(-70, 70); a.set_aspect('equal')
    a.set_xticks([-20, 0, 20]); a.set_yticks([-60, -30, 0, 30, 60])
    a.set_xlabel('Δlon (°)', fontsize=10.5)
    if k == 0:
        a.set_ylabel('ecliptic latitude β (°)')
    else:
        a.set_yticklabels([])

handles = [Patch(color=DARK0, label='I < 0.30 × disk median'), Patch(color=DARK1, label='0.30–0.35 × disk median'),
           Patch(color=DISK, label='other on-disk cells'),
           Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1.3, label='feature region (counted cells in full colour)'),
           Patch(fc=HOLE, ec=INK, lw=0.8, label='coronal-hole mask (channel 4)'),
           plt.Line2D([], [], color='0.5', lw=1.4, ls='--', label='|Δlon| = 20° (panel a)')]
fig.legend(handles=handles, loc='outside lower center', ncol=6, fontsize=10)
save(fig, 'fig_aia_features')
