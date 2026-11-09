"""Is the centre-to-limb effect a simple rescaling? (PAPER_2.md Fig. 11 follow-up)

For every Carrington column and every disk passage (contiguous visible run), the reference value is
the feature at the column's own central-meridian crossing (mean over hours with |dlon_c| <= 2 deg).
Every other hour of the passage is paired with that reference. Per 5-deg bin of dlon_c and channel:
  r            paired Pearson correlation (information retained; invariant to any affine rescaling)
  slope0, R2_0 fit y = b*x through the origin (pure multiplicative rescaling)
  a, b, R2     fit y = a + b*x (affine rescaling)
  zero_frac    fraction of zero values at dlon_c where the reference is non-zero (features lost)
Time lag to the reference is |dlon_c| / 13.2 deg per day, the same east and west, so east/west
symmetry separates geometry from solar evolution.
Outputs: data/nn1_dataset/offset_mapping_test.csv, figures/nn1_dataset_fig12_offset_mapping.png
"""
import json

import numpy as np
import pandas as pd

D = 'data/nn1_dataset'
F = np.load(f'{D}/features.npy', mmap_mode='r')
MA = np.load(f'{D}/mask_aia.npy', mmap_mode='r')
MH = np.load(f'{D}/mask_hmi.npy', mmap_mode='r')
DC = np.load(f'{D}/dlon_center.npy', mmap_mode='r')
names = json.load(open(f'{D}/meta.json'))['features']
EDGES = np.arange(-60, 60.01, 5.0)
NB = len(EDGES) - 1
REF_HALF = 2.0
S = {k: np.zeros((8, NB)) for k in ('n', 'x', 'y', 'xx', 'yy', 'xy', 'zero', 'nz_ref')}

for j in range(F.shape[1]):
    dc = np.asarray(DC[j], np.float64)
    vis = np.isfinite(dc)
    if not vis.any():
        continue
    idx = np.flatnonzero(vis)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 24) + 1)          # passages (gaps of a day allowed)
    for c in range(8):
        f = np.asarray(F[c, j], np.float64)
        m = np.asarray((MA if c < 6 else MH)[j], bool)
        for run in runs:
            run = run[m[run]]
            if run.size < 24:
                continue
            ref_sel = run[np.abs(dc[run]) <= REF_HALF]
            if ref_sel.size == 0:
                continue
            xr = f[ref_sel].mean()
            d, y = dc[run], f[run]
            b = np.digitize(d, EDGES) - 1
            ok = (b >= 0) & (b < NB)
            b, y = b[ok], y[ok]
            x = np.full(y.shape, xr)
            for key, val in (('n', np.ones_like(y)), ('x', x), ('y', y), ('xx', x * x), ('yy', y * y), ('xy', x * y)):
                S[key][c] += np.bincount(b, weights=val, minlength=NB)
            if xr > 0 and c in (0, 1, 2, 5):
                S['nz_ref'][c] += np.bincount(b, minlength=NB)
                S['zero'][c] += np.bincount(b, weights=(y == 0).astype(float), minlength=NB)

mid = 0.5 * (EDGES[:-1] + EDGES[1:])
rows = []
for c in range(8):
    n, sx, sy, sxx, syy, sxy = (S[k][c] for k in ('n', 'x', 'y', 'xx', 'yy', 'xy'))
    cov, vx, vy = sxy / n - sx * sy / n ** 2, sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    r = cov / np.sqrt(vx * vy)
    b = cov / vx
    a = sy / n - b * sx / n
    b0 = sxy / sxx
    sse0 = syy - 2 * b0 * sxy + b0 ** 2 * sxx                            # residuals of y = b0 x
    R2_0 = 1 - sse0 / (n * vy)
    for i in range(NB):
        rows.append(dict(ch=c, channel=names[c], dlon_mid=mid[i], n=int(n[i]), r=r[i], slope0=b0[i], R2_0=R2_0[i],
                         a=a[i], b=b[i], R2=r[i] ** 2,
                         zero_frac=(S['zero'][c][i] / S['nz_ref'][c][i]) if S['nz_ref'][c][i] > 0 else np.nan))
res = pd.DataFrame(rows)
res.to_csv(f'{D}/offset_mapping_test.csv', index=False)

pd.set_option('display.width', 220)
SHORT = ['dark area f=0.30 +-20', 'band +-15', 'dark area f=0.35 +-12.5', 'hole darkness', 'hole-boundary dist',
         'ESWF area', 'HMI weak-field frac', 'HMI mean |B_r|']
print('Paired correlation r with the central-meridian value, by |dlon_c| (east / west):')
hdr = '  ' + ' '.join(f'{v:>12}' for v in ['|d|=5', '15', '25', '35', '45', '55'])
print(f'{"channel":24s}' + hdr)
for c in range(8):
    s = res[res.ch == c].set_index('dlon_mid')
    cells = [f"{s.r.get(-d, np.nan):.2f}/{s.r.get(d, np.nan):.2f}" for d in (2.5, 12.5, 22.5, 32.5, 42.5, 52.5)]
    print(f'{SHORT[c]:24s}  ' + ' '.join(f'{x:>12}' for x in cells))
print('\nMultiplicative vs affine fit (R2 of y = b x  vs  y = a + b x), at |dlon_c| ~ 42.5 (mean of east and west):')
for c in range(8):
    s = res[(res.ch == c) & (res.dlon_mid.abs() == 42.5)]
    print(f'  {SHORT[c]:24s} R2_mult {s.R2_0.mean():6.3f}   R2_affine {s.R2.mean():6.3f}   slope0 {s.slope0.mean():6.2f}   a {s.a.mean():8.2f}  b {s.b.mean():5.2f}')
print('\nFraction of zero values where the central-meridian reference is non-zero (area channels):')
for c in (0, 1, 2, 5):
    s = res[res.ch == c].set_index('dlon_mid')
    print(f'  {SHORT[c]:24s} ' + '  '.join(f'|d|={d:4.1f}: {0.5*(s.zero_frac[-d]+s.zero_frac[d]):.2f}' for d in (2.5, 22.5, 42.5, 52.5)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID, C1 = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0', '#2a78d6'
fig, axs = plt.subplots(2, 4, figsize=(12, 5.6), dpi=150, facecolor=SURF)
for c, ax in enumerate(axs.ravel()):
    s = res[res.ch == c]
    ax.set_facecolor(SURF)
    ax.plot(s.dlon_mid, s.r, color=C1, lw=2, marker='o', ms=4, mec=SURF, mew=1)
    ax.set_ylim(0, 1.02)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_title(f'ch{c}: {SHORT[c]}', fontsize=8, color=INK, loc='left')
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis='y', color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if c % 4 == 0:
        ax.set_ylabel('r with central-meridian value', fontsize=7.5, color=INK2)
    if c >= 4:
        ax.set_xlabel('column offset from central meridian (°)', fontsize=7.5, color=INK2)
fig.suptitle('Information retained off the central meridian: paired correlation with the same longitude at its meridian crossing',
             fontsize=9.5, color=INK, x=0.01, ha='left')
fig.tight_layout()
fig.savefig('figures/nn1_dataset_fig12_offset_mapping.png', facecolor=SURF)
print('\nsaved figures/nn1_dataset_fig12_offset_mapping.png')
