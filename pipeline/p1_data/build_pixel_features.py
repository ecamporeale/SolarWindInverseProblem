"""Pixel-level features for the definitive ranking (PAPER_1.md), AIA-193 FITS or HMI.

Per frame, every on-disk pixel gets its ecliptic latitude beta and longitude offset dlon from
the sub-observer point (heliocentric mean ecliptic frame, same convention as the map cache).
Pixels are then counted exactly into 0.5 deg cells (no point sampling, no interpolation);
every feature is a sum/count over pixels in a region. Weighting = pixel count (equal image area).

Geometry: the ecliptic basis (e1 toward lon offset 0 at beta 0, e2 toward +dlon, N toward
ecliptic north) is expressed in the image frame (x right, y up, z to observer, unit sphere)
by mapping three reference points through the map WCS (astropy), then applied analytically to
all pixels (orthographic approximation of the visible hemisphere).

Output: data/pixel_features_<instrument>.pkl -- one row per frame, column names identical to
the map-cache features (so the ranking code is unchanged), plus family D (AIA intensity within
the holes) and the old-NN1-input reference.
"""
import argparse
import concurrent.futures
import time

import numpy as np
import pandas as pd
from scipy import ndimage

import folds

LATS = np.arange(-70.0, 70.0 + 0.25, 0.5)
DLONS = np.arange(-75.0, 75.0 + 0.25, 0.5)
NL, NC = len(LATS), len(DLONS)
STEP = 0.5
I_LAT0 = int(np.argmin(np.abs(LATS)))
CENTRAL = np.abs(DLONS) <= 7.5
MU_MIN_AIA = np.cos(np.deg2rad(78.0))
MU_MIN_HMI = np.cos(np.deg2rad(65.0))
F_LIST = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
F_REF = 0.35
W_LIST_A = [2.5, 5.0, 7.5, 12.5, 20.0]
OFF_LIST_A = [-30, -20, -10, 0, 10, 20, 30]
K_LIST = [5, 10, 15, 20, 30, 45, 70]
DCHB_OFFS = [-20, -10, 0, 10, 20]
PROF_EDGES = np.arange(-70, 71, 5)
PROF_BIN = np.digitize(LATS, PROF_EDGES) - 1
N_PROF = len(PROF_EDGES) - 1
MIN_HOLE_CELLS = 20
SOFT_WIDTH_DEX = 0.05
W_LIST_H, K_LIST_H, WEAK_G = [7.5, 20.0], [10, 20, 30, 45, 70], [10.0, 20.0]
SIGMAS = [5.0, 10.0, 20.0]
PIL_OFFS = [-20, -10, 0, 10, 20]
L_MAX, SH_SUB = 2, 4


def ecliptic_basis(m):
    import astropy.units as u
    from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic
    ecl = HeliocentricMeanEcliptic(obstime=m.date)
    lam0 = m.observer_coordinate.transform_to(ecl).lon.to_value(u.deg)
    ref = SkyCoord([lam0, lam0 + 30.0, lam0] * u.deg, [0.0, 0.0, 30.0] * u.deg, 1.0 * u.solRad, frame=ecl)
    pix = m.world_to_pixel(ref)
    h = m.meta
    cx, cy = float(h['crpix1']) - 1, float(h['crpix2']) - 1
    r_pix = float(h['rsun_obs']) / float(h['cdelt1'])
    x = (pix.x.to_value(u.pix) - cx) / r_pix
    y = (pix.y.to_value(u.pix) - cy) / r_pix
    v = np.stack([x, y, np.sqrt(np.clip(1 - x ** 2 - y ** 2, 0, None))], 1)
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    e1 = v[0] / np.linalg.norm(v[0])
    e2 = (v[1] - c * e1); e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    nn = (v[2] - c * e1); nn -= (nn @ e1) * e1 + (nn @ e2) * e2; nn /= np.linalg.norm(nn)
    return cx, cy, r_pix, e1, e2, nn


def pixel_coords(shape, cx, cy, r_pix, e1, e2, nn):
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


def cellsum(flat_idx, w=None):
    return np.bincount(flat_idx, weights=w, minlength=NL * NC).reshape(NL, NC)


def hole_mask(dark_cells):
    lab, nl = ndimage.label(dark_cells)
    if not nl:
        return np.zeros_like(dark_cells)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1))
    keep = np.zeros(nl + 1, bool)
    keep[1:] = sizes >= MIN_HOLE_CELLS
    return keep[lab]


MED_MODE = 'frame'          # 'frame' | 'clip90' | 'tmed12' (set by --median-mode)
TMED = {}                   # path -> trailing 12-h median of disk medians (tmed12)


def aia_frame(path, table):
    from sunpy.map import Map
    from preprocess_solar_image import apply_aia193_degradation_correction, set_aia_correction_table
    m = Map(path)
    img = np.asarray(m.data, np.float64) / float(m.meta['exptime'])
    if table is not None:            # 193 A: degradation correction (a per-frame factor; cancels in every feature)
        set_aia_correction_table(table)
        img = apply_aia193_degradation_correction(img, m.date).astype(np.float64)
    cx, cy, r_pix, e1, e2, nn = ecliptic_basis(m)
    Z, rho2, ri, ci, ingrid = pixel_coords(img.shape, cx, cy, r_pix, e1, e2, nn)
    disk = (rho2 <= 0.98 ** 2) & np.isfinite(img)
    frame_med = float(np.median(img[disk]))
    if MED_MODE == 'clip90':                 # median after masking the brightest 10 % of disk pixels
        med = float(np.quantile(img[disk], 0.45))
    elif MED_MODE == 'tmed12':               # trailing 12-h median of the frame medians (causal)
        med = float(TMED[path])
    else:
        med = frame_med
    if not med > 0:
        raise ValueError('non-positive disk median')
    use = ingrid & np.isfinite(img) & (Z >= MU_MIN_AIA)
    idx = (ri * NC + ci)[use]
    I = img[use]
    logr = np.log10(np.clip(I, 1e-3 * med, None) / med)
    n = cellsum(idx)
    out = {'obstime': m.date.datetime, 'disk_median': med, 'frame_median': frame_med}

    # family A: dark area in full-height strips (all rows), from column counts
    ncol = n.sum(0)
    for f in F_LIST:
        dcol = cellsum(idx, (I < f * med).astype(float)).sum(0)
        for w in W_LIST_A:
            for off in OFF_LIST_A:
                cols = np.abs(DLONS - off) <= w
                out[f'A_area f={f} w={w} off={off:+d}'] = 100 * dcol[cols].sum() / max(ncol[cols].sum(), 1)
    dark = I < F_REF * med
    nd = cellsum(idx, dark.astype(float))
    # family B: ecliptic band (central strip), reach, hole-boundary distance
    nrow, ndrow = n[:, CENTRAL].sum(1), nd[:, CENTRAL].sum(1)
    for k in K_LIST:
        rows = np.abs(LATS) <= k
        out[f'B_eclband |lat|<={k}'] = 100 * ndrow[rows].sum() / max(nrow[rows].sum(), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        holes = hole_mask((n > 0) & (nd / np.maximum(n, 1) >= 0.5))
    strip = holes[:, CENTRAL]
    out['B_reach (min |lat| of hole in strip)'] = np.abs(LATS[strip.any(1)]).min() if strip.any() else 90.0
    if holes.any():
        d_in = ndimage.distance_transform_edt(holes, sampling=STEP)
        d_out = ndimage.distance_transform_edt(~holes, sampling=STEP)
    for o in DCHB_OFFS:
        c = int(np.argmin(np.abs(DLONS - o)))
        out[f'B_dchb signed dist off={o:+d}'] = (d_in[I_LAT0, c] if holes[I_LAT0, c] else -d_out[I_LAT0, c]) if holes.any() else -90.0
    # family C: latitude profiles of the central strip
    slog = cellsum(idx, logr)[:, CENTRAL].sum(1)
    for b in range(N_PROF):
        rows = PROF_BIN == b
        nb = nrow[rows].sum()
        lab = f'lat[{PROF_EDGES[b]:+d},{PROF_EDGES[b+1]:+d})'
        out[f'C_dark {lab}'] = ndrow[rows].sum() / nb if nb else np.nan
        out[f'C_logI {lab}'] = slog[rows].sum() / nb if nb else np.nan
    # family D: intensity within the holes (dark tail), new
    cen_pix = CENTRAL[ci[use]]
    lat_pix = np.abs(LATS[ri[use]])
    thr = np.log10(F_REF)
    for reg, msk in [('band |lat|<=30', cen_pix & (lat_pix <= 30)), ('strip', cen_pix)]:
        lr = logr[msk]
        dk = lr < thr
        out[f'D_mean log(I/med) of dark pixels, {reg}'] = lr[dk].mean() if dk.any() else thr
        out[f'D_10th pct log(I/med), {reg}'] = np.percentile(lr, 10) if lr.size else np.nan
        out[f'D_integrated darkness, {reg}'] = np.clip(thr - lr, 0, None).sum() / max(lr.size, 1)
        out[f'D_soft-threshold area, {reg}'] = 100 * np.mean(1 / (1 + np.exp((lr - thr) / SOFT_WIDTH_DEX))) if lr.size else np.nan
    lr = logr[cen_pix & (lat_pix <= 30)]
    out['D_5th pct log(I/med), band |lat|<=30'] = np.percentile(lr, 5) if lr.size else np.nan
    # old NN1 input: mean intensity of the ecliptic trace (|lat|<=3) in the central strip
    tr = cen_pix & (lat_pix <= 3)
    out['OLD NN1 input: AIA ecliptic-trace intensity'] = I[tr].mean() if tr.any() else np.nan
    return out


def pil_dist(Bs, valid):
    pos, neg = valid & (Bs > 0), valid & (Bs < 0)
    d_neg = ndimage.distance_transform_edt(~neg, sampling=STEP) if neg.any() else np.full(Bs.shape, 90.0)
    d_pos = ndimage.distance_transform_edt(~pos, sampling=STEP) if pos.any() else np.full(Bs.shape, 90.0)
    res = []
    for o in PIL_OFFS:
        c = int(np.argmin(np.abs(DLONS - o)))
        res.append(np.nan if not valid[I_LAT0, c] else min(d_neg[I_LAT0, c] if pos[I_LAT0, c] else d_pos[I_LAT0, c], 90.0))
    return res


def hmi_frame(path, _table):
    from sunpy.map import Map
    from global_field_sh import fit_sh_coefficients, evaluate_sh_field, sh_index_list
    m = Map(path)
    Blos = np.asarray(m.data, np.float64)
    cx, cy, r_pix, e1, e2, nn = ecliptic_basis(m)
    Z, rho2, ri, ci, ingrid = pixel_coords(Blos.shape, cx, cy, r_pix, e1, e2, nn)
    use = ingrid & np.isfinite(Blos) & (Z >= MU_MIN_HMI)
    idx = (ri * NC + ci)[use]
    Br = Blos[use] / Z[use]
    aB = np.abs(Br)
    n, sB, sA = cellsum(idx), cellsum(idx, Br), cellsum(idx, aB)
    w10, w20 = cellsum(idx, (aB < 10).astype(float)), cellsum(idx, (aB < 20).astype(float))
    smu = cellsum(idx, Z[use])
    out = {'obstime': m.date.datetime}
    for w in W_LIST_H:
        cols = np.abs(DLONS) <= w
        for k in K_LIST_H:
            rows = np.abs(LATS) <= k
            N_ = n[np.ix_(rows, cols)].sum()
            if N_ < 50:
                continue
            SB, SA = sB[np.ix_(rows, cols)].sum(), sA[np.ix_(rows, cols)].sum()
            out[f'H1_meanabs w={w} k={k}'] = SA / N_
            out[f'H1_absnet w={w} k={k}'] = abs(SB) / N_
            out[f'H1_imbal w={w} k={k}'] = abs(SB) / max(SA, 1e-6)
            out[f'H1_weakfrac<10G w={w} k={k}'] = w10[np.ix_(rows, cols)].sum() / N_
            out[f'H1_weakfrac<20G w={w} k={k}'] = w20[np.ix_(rows, cols)].sum() / N_
    for sdeg in SIGMAS:
        s = sdeg / STEP
        Gn = ndimage.gaussian_filter(n, s)
        Bs = ndimage.gaussian_filter(sB, s) / np.maximum(Gn, 1e-9)
        sv = ndimage.gaussian_filter((n > 0).astype(float), s) > 0.3
        for o, dd in zip(PIL_OFFS, pil_dist(Bs, sv)):
            out[f'H2_pildist sigma={int(sdeg)} off={o:+d}'] = dd
    nrow, srow, arow = n[:, CENTRAL].sum(1), sB[:, CENTRAL].sum(1), sA[:, CENTRAL].sum(1)
    for b in range(N_PROF):
        rows = PROF_BIN == b
        nb = nrow[rows].sum()
        if nb >= 20:
            lab = f'lat[{PROF_EDGES[b]:+d},{PROF_EDGES[b+1]:+d})'
            sb_, sa_ = srow[rows].sum(), arow[rows].sum()
            out[f'H3_meanabs {lab}'] = sa_ / nb
            out[f'H3_absnet {lab}'] = abs(sb_) / nb
            out[f'H3_imbal {lab}'] = abs(sb_) / max(sa_, 1e-6)
    B_, D_ = np.meshgrid(np.deg2rad(LATS), np.deg2rad(DLONS), indexing='ij')
    sub = np.zeros((NL, NC), bool); sub[::SH_SUB, ::SH_SUB] = True
    sel = (n > 0) & sub
    if sel.sum() > 200:
        cellB = sB[sel] / n[sel]
        fit = fit_sh_coefficients(cellB, (np.pi / 2 - B_)[sel], np.mod(D_, 2 * np.pi)[sel], L_MAX,
                                  weights=smu[sel] / n[sel], ridge_lambda=1e-3)
        cdict = dict(zip(sh_index_list(L_MAX), fit['coeffs']))
        out['H4_dipole_tilt_deg'] = np.rad2deg(np.arctan2(np.hypot(cdict[(1, -1)], cdict[(1, 1)]), abs(cdict[(1, 0)])))
        Bl = evaluate_sh_field(fit['coeffs'], L_MAX, np.pi / 2 - B_, np.mod(D_, 2 * np.pi))
        out['H4_|Blowl| subEarth'] = abs(Bl[I_LAT0, int(np.argmin(np.abs(DLONS)))])
        for o, dd in zip(PIL_OFFS, pil_dist(Bl, n > 0)):
            out[f'H4_pildist_lowl off={o:+d}'] = dd
    cen_pix = CENTRAL[ci[use]]
    lat_pix = np.abs(LATS[ri[use]])
    tr = cen_pix & (lat_pix <= 3)
    out['OLD NN1 input: HMI ecliptic-trace |B_los|'] = np.abs(Blos[use][tr]).mean() if tr.any() else np.nan
    return out


def _run(args):
    inst, path, table = args[:3]
    if len(args) > 3:
        global MED_MODE
        MED_MODE = args[3]
        if args[3] == 'tmed12':
            TMED[path] = args[4]
    try:
        r = (aia_frame if inst == 'aia' else hmi_frame)(path, table)
        r['ok'] = True
    except Exception as exc:
        r = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    r['path'] = path
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', choices=['aia', 'hmi'], required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--every', type=int, default=1, help='take every k-th frame (testing)')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', default=None)
    ap.add_argument('--median-mode', choices=['frame', 'clip90', 'tmed12'], default='frame',
                    help="AIA normalising median: per frame (default), after masking the brightest 10 %% of disk pixels, "
                         "or the trailing 12-h median of the frame medians (from data/pixel_features_aia.pkl)")
    ap.add_argument('--heldout', action='store_true', help='AIA 193 held-out frames (2020-06-01 12:00 .. 2025-07-31)')
    ap.add_argument('--channel', default='193', help="AIA channel; non-193 frames from the hourly manifest "
                    "data/preprocessed_window/aia_fits_hourly_quality_<ch>.csv, no degradation correction")
    a = ap.parse_args()
    if a.instrument == 'aia' and a.heldout:
        # held-out 193 frames (fetch_aia_hourly_channel.py --tag heldout); ratio features need no degradation table
        q = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_193_heldout.csv', parse_dates=['hour'])
        q = q[(q.status == 'ok') & (q.hour > folds.DEV_END) & (q.hour <= pd.Timestamp('2025-07-31 23:00'))]
        paths, table = q.sort_values('hour').path.tolist(), None
    elif a.instrument == 'aia' and a.channel != '193':
        q = pd.read_csv(f'data/preprocessed_window/aia_fits_hourly_quality_{a.channel}.csv', parse_dates=['hour'])
        q = q[(q.status == 'ok') & (q.hour <= folds.DEV_END) & (q.hour >= folds.DEV_START)]
        paths, table = q.sort_values('hour').path.tolist(), None
    elif a.instrument == 'aia':
        q = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
        from aia_frames import usable
        q = q[usable(q) & (q.obstime <= folds.DEV_END) & (q.obstime >= folds.DEV_START)]
        paths = q.sort_values('obstime').path.tolist()
        from preprocess_solar_image import _get_aia_correction_table
        table = _get_aia_correction_table()
    else:
        paths = pd.read_csv('data/ecliptic_map_cache/hmi_paths_hmi_dev.csv').path.tolist()
        table = None
    paths = paths[::a.every][:a.limit]
    extra = {}
    if a.median_mode == 'tmed12':
        base = pd.read_pickle('data/pixel_features_aia.pkl')[['path', 'obstime', 'disk_median']]
        base['obstime'] = pd.to_datetime(base.obstime)
        s = base.sort_values('obstime').set_index('obstime')['disk_median']
        tm = s.rolling('12h', min_periods=1).median()                      # window (t - 12 h, t], trailing
        extra = dict(zip(base.sort_values('obstime').path, tm.to_numpy()))
    out = a.out or (f'data/pixel_features_aia{a.channel}.pkl' if a.instrument == 'aia' and a.channel != '193'
                    else f'data/pixel_features_{a.instrument}.pkl')
    if a.heldout and a.out is None:
        out = 'data/pixel_features_aia_heldout.pkl'
    if a.median_mode != 'frame' and a.out is None:
        out = out.replace('.pkl', f'_{a.median_mode}.pkl')
    if a.median_mode == 'tmed12':
        paths = [p for p in paths if p in extra]
    print(f'{a.instrument}: {len(paths)} frames -> {out}', flush=True)
    rows, t0 = [], time.time()
    with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_run, [(a.instrument, p, table, a.median_mode, extra.get(p)) for p in paths], chunksize=4), 1):
            rows.append(r)
            if i % 5000 == 0 or i == len(paths):
                el = time.time() - t0
                print(f'  {i}/{len(paths)} ok={sum(x["ok"] for x in rows)} {el/60:.1f} min, eta {(len(paths)-i)*el/i/60:.0f} min', flush=True)
    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    print(f'done: {df.ok.sum()} ok, {len(bad)} failed', flush=True)
    for e in bad.get('error', pd.Series(dtype=str)).head(5):
        print('  ERR', e, flush=True)
    df[df.ok].drop(columns=['ok']).to_pickle(out)


if __name__ == '__main__':
    main()
