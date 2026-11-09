"""PFSS magnetic connectivity for the look-alike test (knn_degeneracy_test.py matching).

For every matched ACE hour (data/knn_degeneracy/inputs.npz: ACE time T, speed v) the source hour is
t_src = T - 193.5 Rs / v and the source longitude is the Carrington longitude of the central-meridian column at t_src
(as in the look-alike test); the source latitude is Earth's heliographic latitude B0 (ACE HC_lat).
Per source day: PFSS (sunkit-magex, rss = 2.5 Rsun, nrho = 30) from the FIRST GONG mrzqs map of that day
(causal: data up to 00:xx UT of the day), then for each hour of that day a field line traced down from the source
surface at (source lon, B0).
Quantities per hour:
  log10_fs    expansion factor of the traced line, f_s = (R/Rss)^2 Bph/Bss
  foot_lat    heliographic latitude of the photospheric footpoint (deg); |foot_lat| also used
  theta_b     great-circle distance from the footpoint to the nearest closed-field point (open/closed map traced
              from a 2-deg photospheric grid) (deg)
  theta_b_fine  same from a local 0.5-deg patch (+-5 deg) around the footpoint; 5 deg (capped) if no closed point in it
  v_wsa       WSA speed 285 + 625 / (1 + fs)^(2/9) * (1 - 0.8 exp(-(theta_b_fine / 2)^2))^3 km/s
  hcs_dist    great-circle distance from the source-surface point to the nearest polarity inversion of Br at the
              source surface (deg)
  polarity    sign of Br at the source-surface point
  foot_dlon   footpoint longitude minus source longitude (deg, wrapped)

Input map (--source; solver and extraction identical for all three):
  gong       the GONG mrzqs synoptic map as is (full sphere)
  gong_vis   the same map with everything outside the visible disk set to 0 (mu = cos(angle from disk centre)
             >= cos 65 deg at the map time, as the HMI features), then flux-balanced on the visible part
  gong_vis_poles  gong_vis plus the polar caps (|lat| >= 60 deg) of the same GONG map: only the far side
             outside the caps is removed
  gong_old27 the GONG map of the source day minus 27 days (first map of that day): near side ~27 d stale
  hmi_vis    the HMI hmi.M_720s magnetogram of the same day (first frame of the day), remapped to the GONG grid,
             B_r = B_los / mu, same mask and flux balance -> visible disk only, HMI instead of GONG

  python pfss_connectivity.py --test [--source gong_vis]   # one day, timing
  python pfss_connectivity.py --workers 48 [--source ...]  # all days -> data/pfss_connectivity/per_hour[_<source>].csv
"""
import argparse
import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from data_paths import SOLAR

warnings.filterwarnings('ignore')
OUT = Path('data/pfss_connectivity'); OUT.mkdir(parents=True, exist_ok=True)
GONG = str(SOLAR / 'GONG_MAG')
D_KM = (215.0 - 21.5) * 695700.0
RSS, NRHO, GRID_DEG = 2.5, 30, 2.0
FINE_DEG, FINE_HALF = 0.5, 5.0          # local open/closed patch around each footpoint for theta_b_fine
BETA = float(os.environ.get('KNN_ALIGN_BETA', '1'))    # source-time alignment, as in knn_backmap_test.py
ALIGN_TAG = '' if BETA == 1 else f'_beta{BETA:g}'


def build_hours():
    """Per matched ACE hour: source day, source Carrington lon (CM column at t_src), B0."""
    D = np.load('data/knn_degeneracy/inputs.npz', allow_pickle=True)
    T, v = pd.to_datetime(D['time']), D['v']
    DC = np.load('data/nn1_dataset/dlon_center.npy').astype(np.float32)
    t0 = pd.to_datetime(pd.read_csv('data/nn1_dataset/times.csv', usecols=['time'])['time']).iloc[0]
    if BETA == 1:
        src = T - pd.to_timedelta(D_KM / v, unit='s')
    else:
        src = T - pd.to_timedelta(96 * 3600 + BETA * (D_KM / v - 96 * 3600), unit='s')
    hs = np.rint((src - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    adc = np.where(np.isfinite(DC[:, hs]), np.abs(DC[:, hs]), np.inf)
    col = adc.argmin(0)
    o = pd.read_csv('data/preprocessed_window/multi_spacecraft_obs_20110101_20211231T235959.csv', parse_dates=['Time'])
    lat = o[o.Spacecraft_ID == 'ACE'].set_index('Time')['HC_lat'].resample('1h').mean().interpolate(limit=24)
    return pd.DataFrame({'row': np.arange(len(T)), 'time': T, 'src_time': src.round('h'), 'src_day': src.floor('D'),
                         'src_lon': col.astype(float), 'b0': lat.reindex(T).to_numpy()})


def first_map(day):
    files = sorted(glob.glob(f'{GONG}/{day:%Y}/{day:%Y%m%d}/mrzqs*.fits.gz'))
    return files[0] if files else None


MU_MIN = np.cos(np.deg2rad(65.0))
POLE_LAT = 60.0
HMI_PATHS = None


def first_hmi(day):
    global HMI_PATHS
    if HMI_PATHS is None:
        hp = pd.read_csv('data/ecliptic_map_cache/hmi_paths_hmi_dev.csv').path
        t = pd.to_datetime(hp.str.extract(r'hmi\.M_720s\.(\d{8}_\d{6})')[0], format='%Y%m%d_%H%M%S')
        HMI_PATHS = pd.Series(hp.values, index=t).sort_index()
    s = HMI_PATHS[(HMI_PATHS.index >= day) & (HMI_PATHS.index < day + pd.Timedelta(days=1))]
    return s.iloc[0] if len(s) else None


def visible_mu(m, t):
    """mu = cos(angle from disk centre) of every pixel of the Carrington map m, for an Earth observer at time t."""
    import astropy.units as u
    import sunpy.map
    from sunpy.coordinates import sun
    c = sunpy.map.all_coordinates_from_map(m)
    lat, lon = np.deg2rad(c.lat.to_value(u.deg)), np.deg2rad(c.lon.to_value(u.deg))
    b0, l0 = np.deg2rad(sun.B0(t).to_value(u.deg)), np.deg2rad(sun.L0(t).to_value(u.deg))
    return np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(b0) * np.cos(lon - l0)


def make_input(m, day, source):
    """Return (map, info) for the chosen source; m is the GONG map (sets the grid and time)."""
    import astropy.units as u
    import sunpy.map
    if source in ('gong', 'gong_old27'):
        return m, ''
    mu = visible_mu(m, m.date)
    vis = np.isfinite(mu) & (mu >= MU_MIN)
    if source == 'gong_vis_poles':
        lat = sunpy.map.all_coordinates_from_map(m).lat.to_value(u.deg)
        vis = vis | (np.abs(lat) >= POLE_LAT)
    if source in ('gong_vis', 'gong_vis_poles'):
        d = np.where(vis, np.asarray(m.data, float), 0.0)
        info = ''
    else:
        hp = first_hmi(day)
        if hp is None:
            raise FileNotFoundError('no HMI frame for the day')
        H = sunpy.map.Map(hp)
        H = H.superpixel([2, 2] * u.pix, func=np.nanmean)        # hmi_paths files are 1024 px -> 512
        Hr = H.reproject_to(m.wcs)
        mu_h = visible_mu(m, H.date)
        d = np.asarray(Hr.data, float) / np.where(mu_h > 0.05, mu_h, np.nan)
        d = np.where(vis & np.isfinite(d), d, 0.0)
        info = os.path.basename(hp)
    d[vis] -= d[vis].mean()                                  # flux balance on the visible part
    return sunpy.map.Map(d, m.meta), info


def gc_dist(lat1, lon1, lat2, lon2):
    """Great-circle distance (deg) between one point and arrays of points (all deg)."""
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def process_day(args):
    day, rows, source = args
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    import sunkit_magex.pfss as pfss
    f = first_map(day - pd.Timedelta(days=27) if source == 'gong_old27' else day)
    if f is None:
        return [dict(row=r, status='no_map') for r in rows['row']]
    try:
        m = sunpy.map.Map(f)
        m = m.resample([360, 180] * u.pix)
        m, src_info = make_input(m, day, source)
        pin = pfss.Input(m, NRHO, RSS)
        pout = pfss.pfss(pin)
        frame = pout.coordinate_frame
        tracer = pfss.tracing.PerformanceTracer()
        # open/closed map on a photospheric grid
        glat = np.arange(-89.0, 89.1, GRID_DEG); glon = np.arange(0.0, 360.0, GRID_DEG)
        LA, LO = np.meshgrid(glat, glon, indexing='ij')
        seeds = SkyCoord(LO.ravel() * u.deg, LA.ravel() * u.deg, 1.01 * u.R_sun, frame=frame)
        fl_grid = tracer.trace(seeds, pout)
        open_g = np.array([fl.is_open for fl in fl_grid]).reshape(LA.shape)
        cl_lat, cl_lon = LA[~open_g], LO[~open_g]
        # source-surface polarity inversion
        br_ss = pout.source_surface_br
        ss = np.asarray(br_ss.data)
        ss_lat = np.rad2deg(np.arcsin(np.linspace(-1, 1, ss.shape[0] + 1)[:-1] + 1.0 / ss.shape[0]))
        ss_lon = (np.arange(ss.shape[1]) + 0.5) * 360.0 / ss.shape[1]
        sgn = np.sign(ss)
        pil = (sgn != np.roll(sgn, 1, axis=1)) | (sgn != np.vstack([sgn[:1], sgn[:-1]]))
        pli, plj = np.nonzero(pil)
        pil_lat, pil_lon = ss_lat[pli], ss_lon[plj]
        # field lines from the source surface down
        s = SkyCoord(rows['src_lon'].to_numpy() * u.deg, rows['b0'].to_numpy() * u.deg, (RSS - 0.01) * u.R_sun, frame=frame)
        fls = tracer.trace(s, pout)
        efs = fls.expansion_factors
        # fine theta_b: local 0.5-deg patches around every open footpoint, traced in one batch
        off = np.arange(-FINE_HALF, FINE_HALF + 1e-9, FINE_DEG)
        OA, OB = np.meshgrid(off, off, indexing='ij')
        foot = []
        for fl in fls:
            if fl.is_open and len(fl.coords) > 1:
                fp = fl.solar_footpoint
                foot.append((float(fp.lat.to_value(u.deg)), float(fp.lon.to_value(u.deg)) % 360.0))
            else:
                foot.append(None)
        plat, plon, owner = [], [], []
        for i, fp in enumerate(foot):
            if fp is None:
                continue
            la = np.clip(fp[0] + OA.ravel(), -89.5, 89.5)
            lo = (fp[1] + OB.ravel() / max(np.cos(np.deg2rad(fp[0])), 0.1)) % 360.0
            plat.append(la); plon.append(lo); owner.append(np.full(la.size, i))
        tb_fine = np.full(len(foot), np.nan)
        if plat:
            plat, plon, owner = np.concatenate(plat), np.concatenate(plon), np.concatenate(owner)
            fpatch = tracer.trace(SkyCoord(plon * u.deg, plat * u.deg, 1.01 * u.R_sun, frame=frame), pout)
            closed = ~np.array([fl.is_open for fl in fpatch])
            for i in np.unique(owner):
                m = (owner == i) & closed
                tb_fine[i] = float(gc_dist(foot[i][0], foot[i][1], plat[m], plon[m]).min()) if m.any() else FINE_HALF
        out = []
        for i, (r, lon, b0) in enumerate(zip(rows['row'], rows['src_lon'], rows['b0'])):
            fl = fls[i]
            res = dict(row=r, status='ok', map=os.path.basename(f), src_file=src_info, polarity=float(fl.polarity) if fl.is_open else np.nan,
                       is_open=bool(fl.is_open))
            if fl.is_open and len(fl.coords) > 1:
                fp = fl.solar_footpoint
                flat, flon = float(fp.lat.to_value(u.deg)), float(fp.lon.to_value(u.deg)) % 360.0
                res.update(log10_fs=float(np.log10(efs[i])) if np.isfinite(efs[i]) and efs[i] > 0 else np.nan,
                           foot_lat=flat, foot_dlon=(flon - lon + 180.0) % 360.0 - 180.0,
                           theta_b=float(gc_dist(flat, flon, cl_lat, cl_lon).min()) if cl_lat.size else np.nan,
                           theta_b_fine=float(tb_fine[i]))
                if np.isfinite(efs[i]) and efs[i] > 0:
                    res['v_wsa'] = float(285.0 + 625.0 / (1.0 + efs[i]) ** (2.0 / 9.0)
                                         * (1.0 - 0.8 * np.exp(-(tb_fine[i] / 2.0) ** 2)) ** 3)
            res['hcs_dist'] = float(gc_dist(b0, lon, pil_lat, pil_lon).min()) if pil_lat.size else np.nan
            out.append(res)
        return out
    except Exception as e:                                   # keep going; report per day
        return [dict(row=r, status=f'error: {type(e).__name__}: {e}'[:200]) for r in rows['row']]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true')
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--source', choices=['gong', 'gong_old27', 'gong_vis', 'gong_vis_poles', 'hmi_vis'], default='gong')
    a = p.parse_args()
    H = build_hours()
    groups = [(d, g, a.source) for d, g in H.groupby('src_day')]
    print(f'{len(H)} hours over {len(groups)} source days', flush=True)
    if a.test:
        d, g, _ = groups[len(groups) // 2]
        t = time.time(); res = process_day((d, g, a.source)); dt = time.time() - t
        R = pd.DataFrame(res)
        print(f'[{a.source}] day {d:%Y-%m-%d}: {len(g)} hours in {dt:.1f} s'); print(R.drop(columns=['map', 'src_file'], errors='ignore').head(8).round(3).to_string())
        print(R.status.value_counts().to_string())
    else:
        from multiprocessing import Pool
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        t = time.time(); res = []
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_day, groups, chunksize=4)):
                res += r
                if (i + 1) % 200 == 0:
                    print(f'  {i + 1}/{len(groups)} days, {time.time() - t:.0f} s', flush=True)
        R = H.merge(pd.DataFrame(res), on='row', how='left')
        R.to_csv(OUT / (f'per_hour{ALIGN_TAG}.csv' if a.source == 'gong' else f'per_hour_{a.source}{ALIGN_TAG}.csv'), index=False)
        print(f'done in {time.time() - t:.0f} s'); print(R.status.str[:40].value_counts().to_string())
