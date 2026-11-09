"""NN1 input dataset: the 8 selected features (PAPER_1.md section 4.5, set 2a) per Carrington
longitude and hour, pixel level, development window only.

For every hour, each Carrington-longitude column phi (1 deg, 0..359) is located on the disk at
the ecliptic: dlon_c(phi) = ecliptic longitude offset from the sub-observer point of the point
(ecliptic lat 0, Carrington lon phi). Every feature is computed exactly as in the ranking, but
centred on dlon_c(phi) (offset 0): strips are |dlon - dlon_c| <= w, bands are |lat| <= k.
Columns with |dlon_c| <= VIS_MAX are visible (mask 1); others, and every column of an hour without
a usable frame, get mask 0 and value 0 (the pipeline treats them as gradient-free constants).

Outputs (data/nn1_dataset/):
  features.npy     float32 [8, 360, Nt]   channel order = FEATURES
  mask_aia.npy     uint8   [360, Nt]      channels 0-5 valid
  mask_hmi.npy     uint8   [360, Nt]      channels 6-7 valid
  dlon_center.npy  float16 [360, Nt]      column centre offset from central meridian (deg; NaN if not on disk)
  times.csv        hourly time axis, AIA/HMI frame used, status
  missing_hours.csv  hours without usable AIA and/or HMI, with reason
  meta.json        feature definitions and parameters
"""
import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import folds
import build_pixel_features as BP

OUT = Path('data/nn1_dataset')
DECONV = False             # --deconvolve: PSF deconvolution (BID) of the raw frame first (experiment E2)
ALLOW_UNCAL = False        # --heldout: frames after the last AIA calibration epoch (features are ratios to the frame median)
LATS = np.arange(-70.0, 70.0 + 0.25, 0.5)
DLONS = np.arange(-90.0, 90.0 + 0.25, 0.5)
NL, NC = len(LATS), len(DLONS)
STEP = 0.5
I_LAT0 = int(np.argmin(np.abs(LATS)))
NPHI = 360
PHI = np.arange(NPHI, dtype=float)
VIS_MAX = 60.0
MIN_HOLE_CELLS = 20
FEATURES = [
    'A: area f=0.30, w=20',
    'B: ecliptic band |lat|<=15 (strip w=7.5)',
    'A: area f=0.35, w=12.5',
    'D: mean log10(I/med) of dark pixels (f=0.35), |lat|<=30, w=7.5',
    'B: hole-boundary signed distance at the column ecliptic point (deg)',
    'A: ESWF area f=0.35, w=7.5',
    'H: weak-field fraction |B_r|<10 G, |lat|<=30, w=7.5',
    'H: mean |B_r| (G), |lat|<=30, w=7.5',
]


def pixel_cells(shape, m):
    cx, cy, r_pix, e1, e2, nn = BP.ecliptic_basis(m)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    X, Y = (xx - cx) / r_pix, (yy - cy) / r_pix
    rho2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1 - rho2, 0, None))
    beta = np.rad2deg(np.arcsin(np.clip(X * nn[0] + Y * nn[1] + Z * nn[2], -1, 1)))
    dlon = np.rad2deg(np.arctan2(X * e2[0] + Y * e2[1] + Z * e2[2], X * e1[0] + Y * e1[1] + Z * e1[2]))
    ri = np.rint((beta - LATS[0]) / STEP).astype(np.int64)
    ci = np.rint((dlon - DLONS[0]) / STEP).astype(np.int64)
    ingrid = (rho2 < 1) & (ri >= 0) & (ri < NL) & (ci >= 0) & (ci < NC)
    return Z, rho2, ri, ci, ingrid


def column_centres(m):
    """dlon_c for every Carrington longitude (NaN if not within +-VIS_MAX of central meridian)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic
    from sunpy.coordinates import frames
    ecl = HeliocentricMeanEcliptic(obstime=m.date)
    lam0 = m.observer_coordinate.transform_to(ecl).lon.to_value(u.deg)
    d = np.arange(-VIS_MAX - 2, VIS_MAX + 2.01, 0.5)
    pts = SkyCoord((lam0 + d) * u.deg, np.zeros_like(d) * u.deg, 1.0 * u.solRad, frame=ecl)
    carr = pts.transform_to(frames.HeliographicCarrington(observer=m.observer_coordinate, obstime=m.date)).lon.to_value(u.deg)
    carr = np.unwrap(np.deg2rad(carr))
    carr = np.rad2deg(carr)                           # monotonic in d
    if carr[-1] < carr[0]:
        d, carr = d[::-1], carr[::-1]
    out = np.full(NPHI, np.nan)
    base = carr[0]
    for k in (-1, 0, 1):
        targets = PHI + 360.0 * (np.floor(base / 360.0) + k)
        inside = (targets >= carr[0]) & (targets <= carr[-1])
        out[inside] = np.interp(targets[inside], carr, d)
    out[np.abs(out) > VIS_MAX] = np.nan
    return out


def window_sum(A, centre_cols, half_cols):
    """sum over columns [c-h, c+h] of A (rows already summed or 1-D per column)."""
    cs = np.concatenate([[0.0], np.cumsum(A)])
    lo = np.clip(centre_cols - half_cols, 0, len(A) - 1)
    hi = np.clip(centre_cols + half_cols, 0, len(A) - 1)
    return cs[hi + 1] - cs[lo]


def aia_columns(path, table):
    from sunpy.map import Map
    from preprocess_solar_image import apply_aia193_degradation_correction, set_aia_correction_table
    set_aia_correction_table(table)
    m = Map(path)
    data = np.asarray(m.data, np.float64)
    if DECONV:                                  # on raw DN, before exposure / degradation scaling (PROSWIN order)
        from psf_deconvolution import deconvolve
        data = deconvolve(data)
    img = apply_aia193_degradation_correction(data / float(m.meta['exptime']), m.date,
                                              allow_uncalibrated=ALLOW_UNCAL).astype(np.float64)
    Z, rho2, ri, ci, ingrid = pixel_cells(img.shape, m)
    disk = (rho2 <= 0.98 ** 2) & np.isfinite(img)
    med = float(np.median(img[disk]))
    if not med > 0:
        raise ValueError('non-positive disk median')
    use = ingrid & np.isfinite(img) & (Z >= BP.MU_MIN_AIA)
    r_, c_ = ri[use], ci[use]
    I = img[use]
    idx = r_ * NC + c_
    cell = lambda w=None: np.bincount(idx, weights=w, minlength=NL * NC).reshape(NL, NC)
    n = cell()
    d30, d35 = cell((I < 0.30 * med).astype(float)), cell((I < 0.35 * med).astype(float))
    logr = np.log10(np.clip(I, 1e-3 * med, None) / med)
    slog35 = cell(np.where(I < 0.35 * med, logr, 0.0))
    band15, band30 = np.abs(LATS) <= 15, np.abs(LATS) <= 30
    centres = column_centres(m)
    vis = np.isfinite(centres)
    cc = np.rint((np.where(vis, centres, 0) - DLONS[0]) / STEP).astype(int)
    h = lambda w: int(round(w / STEP))
    F = np.zeros((6, NPHI), np.float32)
    nall, d30all, d35all = n.sum(0), d30.sum(0), d35.sum(0)
    F[0] = 100 * window_sum(d30all, cc, h(20)) / np.maximum(window_sum(nall, cc, h(20)), 1)
    F[1] = 100 * window_sum(d35[band15].sum(0), cc, h(7.5)) / np.maximum(window_sum(n[band15].sum(0), cc, h(7.5)), 1)
    F[2] = 100 * window_sum(d35all, cc, h(12.5)) / np.maximum(window_sum(nall, cc, h(12.5)), 1)
    nd30b = window_sum(d35[band30].sum(0), cc, h(7.5))
    F[3] = np.where(nd30b > 0, window_sum(slog35[band30].sum(0), cc, h(7.5)) / np.maximum(nd30b, 1), np.log10(0.35))
    with np.errstate(invalid='ignore', divide='ignore'):
        holes = BP.hole_mask((n > 0) & (d35 / np.maximum(n, 1) >= 0.5))
    if holes.any():
        d_in = ndimage.distance_transform_edt(holes, sampling=STEP)
        d_out = ndimage.distance_transform_edt(~holes, sampling=STEP)
        F[4] = np.where(holes[I_LAT0, cc], d_in[I_LAT0, cc], -d_out[I_LAT0, cc])
    else:
        F[4] = -90.0
    F[5] = 100 * window_sum(d35all, cc, h(7.5)) / np.maximum(window_sum(nall, cc, h(7.5)), 1)
    F[:, ~vis] = 0.0
    return F, vis, centres


def hmi_columns(path):
    from sunpy.map import Map
    m = Map(path)
    Blos = np.asarray(m.data, np.float64)
    Z, rho2, ri, ci, ingrid = pixel_cells(Blos.shape, m)
    use = ingrid & np.isfinite(Blos) & (Z >= BP.MU_MIN_HMI)
    idx = (ri * NC + ci)[use]
    aB = np.abs(Blos[use] / Z[use])
    cell = lambda w=None: np.bincount(idx, weights=w, minlength=NL * NC).reshape(NL, NC)
    n, sA, w10 = cell(), cell(aB), cell((aB < 10).astype(float))
    band30 = np.abs(LATS) <= 30
    centres = column_centres(m)
    vis = np.isfinite(centres)
    cc = np.rint((np.where(vis, centres, 0) - DLONS[0]) / STEP).astype(int)
    nb = window_sum(n[band30].sum(0), cc, 15)
    F = np.zeros((2, NPHI), np.float32)
    F[0] = window_sum(w10[band30].sum(0), cc, 15) / np.maximum(nb, 1)
    F[1] = window_sum(sA[band30].sum(0), cc, 15) / np.maximum(nb, 1)
    vis &= nb >= 50
    F[:, ~vis] = 0.0
    return F, vis


_G = {}


def _init(nt, table, out=None, allow_uncal=False, vis_max=None, deconv=False):
    global OUT, ALLOW_UNCAL, VIS_MAX, DECONV
    if out is not None:
        OUT = Path(out)
    ALLOW_UNCAL, DECONV = allow_uncal, deconv
    if vis_max is not None:
        VIS_MAX = float(vis_max)
    _G['F'] = np.load(OUT / 'features.npy', mmap_mode='r+')
    _G['ma'] = np.load(OUT / 'mask_aia.npy', mmap_mode='r+')
    _G['mh'] = np.load(OUT / 'mask_hmi.npy', mmap_mode='r+')
    _G['dc'] = np.load(OUT / 'dlon_center.npy', mmap_mode='r+')
    _G['table'] = table


def _hour(args):
    k, aia_path, hmi_path = args
    st = {'aia': 'no usable AIA frame (missing or QUALITY != 0)', 'hmi': 'no HMI file'}
    if isinstance(aia_path, str):
        try:
            F, vis, centres = aia_columns(aia_path, _G['table'])
            _G['F'][0:6, :, k] = F
            _G['ma'][:, k] = vis
            _G['dc'][:, k] = centres
            st['aia'] = 'ok'
        except Exception as exc:
            st['aia'] = f'AIA error: {type(exc).__name__}: {exc}'[:200]
    if isinstance(hmi_path, str):
        try:
            F, vis = hmi_columns(hmi_path)
            _G['F'][6:8, :, k] = F
            _G['mh'][:, k] = vis
            st['hmi'] = 'ok'
        except Exception as exc:
            st['hmi'] = f'HMI error: {type(exc).__name__}: {exc}'[:200]
    return k, st['aia'], st['hmi']


def main():
    global OUT, ALLOW_UNCAL, VIS_MAX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--heldout', action='store_true', help='build the held-out dataset (frames from the development table '
                    'before 2020-06-01 12:00 and from the held-out manifest after; no HMI; uncalibrated frames allowed)')
    ap.add_argument('--start', default='2020-05-01 00:00'); ap.add_argument('--end', default='2025-07-31 23:00')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--vis-max', type=float, default=None, help='visibility cut |dlon_c| (deg); default 60')
    ap.add_argument('--aia-only', action='store_true', help='skip HMI (masks 0)')
    ap.add_argument('--deconvolve', action='store_true', help='PSF-deconvolve every AIA frame first (BID, '
                    'psf_deconvolution.py; experiment E2)')
    ap.add_argument('--overwrite', action='store_true', help='allow replacing an existing dataset in the output directory')
    ap.add_argument('--hours-file', default=None, help='debug/control: CSV with a "time" column; only these hours')
    a = ap.parse_args()
    if a.heldout:
        OUT, ALLOW_UNCAL = Path(a.out_dir or 'data/nn1_dataset_heldout'), True
        hours = pd.date_range(pd.Timestamp(a.start), pd.Timestamp(a.end), freq='1h')
    else:
        hours = pd.date_range(folds.DEV_START, folds.DEV_END, freq='1h')
    if a.vis_max is not None:
        VIS_MAX = float(a.vis_max)
    if a.out_dir:                              # 2026-10-15: honoured in every mode (was: only with --vis-max)
        OUT = Path(a.out_dir)
    if (OUT / 'features.npy').exists() and not a.overwrite:
        raise SystemExit(f'{OUT} already holds a dataset; pass --overwrite to replace it')
    OUT.mkdir(parents=True, exist_ok=True)
    if a.limit:
        hours = hours[:a.limit]
    if a.hours_file:
        hours = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(a.hours_file)['time']))
    nt = len(hours)
    q = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
    from aia_frames import usable
    q = q[usable(q)].copy()                    # QUALITY == 0 and no very short flare-mode exposure
    q['t'] = q.obstime.dt.round('h')
    aia = q.drop_duplicates('t').set_index('t').path.reindex(hours)
    if a.heldout:
        # after the development window: the held-out manifest (same frame rules incl. the flare-exposure replacement)
        hq = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_193_heldout.csv', parse_dates=['hour'])
        hq = hq[hq.status == 'ok'].set_index('hour').path
        late = hours > folds.DEV_END
        aia[late] = hq.reindex(hours[late]).to_numpy()
        hmi = pd.Series(np.nan, index=hours, dtype=object)               # HMI not used by NN1 (dropped)
    else:
        hp = pd.read_csv('data/ecliptic_map_cache/hmi_paths_hmi_dev.csv').path
        ht = pd.to_datetime(hp.str.extract(r'hmi\.M_720s\.(\d{8}_\d{6})')[0], format='%Y%m%d_%H%M%S')
        hmi = pd.Series(hp.values, index=ht).groupby(level=0).first().reindex(hours)
    if a.aia_only:
        hmi = pd.Series(np.nan, index=hours, dtype=object)
    print(f'{nt} hours; AIA frames {aia.notna().sum()}, HMI frames {hmi.notna().sum()}', flush=True)

    np.lib.format.open_memmap(OUT / 'features.npy', 'w+', np.float32, (8, NPHI, nt))[:] = 0
    np.lib.format.open_memmap(OUT / 'mask_aia.npy', 'w+', np.uint8, (NPHI, nt))[:] = 0
    np.lib.format.open_memmap(OUT / 'mask_hmi.npy', 'w+', np.uint8, (NPHI, nt))[:] = 0
    np.lib.format.open_memmap(OUT / 'dlon_center.npy', 'w+', np.float16, (NPHI, nt))[:] = np.nan
    from preprocess_solar_image import _get_aia_correction_table
    table = _get_aia_correction_table()
    tasks = [(k, aia.iloc[k], hmi.iloc[k]) for k in range(nt)]
    res, t0 = [], time.time()
    with concurrent.futures.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(nt, table, OUT, ALLOW_UNCAL, VIS_MAX, a.deconvolve)) as ex:
        for i, r in enumerate(ex.map(_hour, tasks, chunksize=8), 1):
            res.append(r)
            if i % 5000 == 0 or i == nt:
                el = time.time() - t0
                print(f'  {i}/{nt} {el/60:.1f} min, eta {(nt-i)*el/i/60:.0f} min', flush=True)
    st = pd.DataFrame(res, columns=['k', 'aia_status', 'hmi_status']).sort_values('k')
    tab = pd.DataFrame({'time': hours, 'aia_path': aia.values, 'hmi_path': hmi.values,
                        'aia_status': st.aia_status.values, 'hmi_status': st.hmi_status.values})
    tab.to_csv(OUT / 'times.csv', index=False)
    miss = tab[(tab.aia_status != 'ok') | (tab.hmi_status != 'ok')]
    miss.to_csv(OUT / 'missing_hours.csv', index=False)
    meta = dict(features=FEATURES, channel_mask={'0-5': 'mask_aia', '6-7': 'mask_hmi'}, n_phi=NPHI,
                carrington_longitude_deg='0..359 (1 deg)', time_axis='hourly, development window',
                dev_start=str(folds.DEV_START), dev_end=str(folds.DEV_END), window_start=str(hours[0]), window_end=str(hours[-1]), heldout=bool(a.heldout), visibility_max_dlon_deg=VIS_MAX,
                psf_deconvolution='BID (psf_deconvolution.py), aiapy PSF 193 rebinned to 1024' if a.deconvolve else 'none',
                dark_threshold='intensity < f x on-disk median of the frame', hmi='raw (no trend removal)',
                pixel_level=True, source='JSOC synoptic AIA-193 Level-1.5 (QUALITY==0), hmi.M_720s')
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'done: AIA ok {(tab.aia_status == "ok").sum()}/{nt}, HMI ok {(tab.hmi_status == "ok").sum()}/{nt}; '
          f'{len(miss)} hours with at least one channel group missing', flush=True)
    print(miss.aia_status.value_counts().head(), '\n', miss.hmi_status.value_counts().head(), flush=True)


if __name__ == '__main__':
    main()
