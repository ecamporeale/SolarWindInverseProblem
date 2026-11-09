"""WSA 4.5 (GONG) speed profiles for the Part I comparison (2026-10-24), shared by the HUXt run and the run through our
solver so that both use identical WSA inputs. Run with the `huxt` environment.

For every downloaded CCMC iSWA WSA 4.5 map (download_wsa_iswa.py --product wsa45; 2010-12 .. 2020-06): the WSA speed
(1-AU-equivalent, as WSA delivers it) on the 128 HUXt Carrington longitudes ((i + 0.5)·2π/128) at ACE's heliographic
latitude at the map time (before 2011-01-01: the latitude one year later), read with HUXt's own reader (huxt_inputs.get_WSA_maps) and interpolated in latitude as in
part2_huxt_forecast.py. ACE positions: data/ace_dev/ACE_speed_lon_lat_hr.txt (development observation file).

  $PY_H pipeline/p1_model/part1_wsa_profiles.py [--workers 32]
Output: runs/part1_wsa/wsa45_profiles.npz  map_time [M], lon [128] (rad), v1au [M, 128] (km/s), path [M]
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from data_paths import SOLAR

DIR = str(SOLAR / 'WSA4.5_iSWA/GONG')
ACE = 'data/ace_dev/ACE_speed_lon_lat_hr.txt'
NLON = 128
LON = (np.arange(NLON) + 0.5) * 2 * np.pi / NLON
_LAT = {}


def _init():
    a = pd.read_csv(ACE, parse_dates=['Time']).set_index('Time')['HC_lat'].astype(float)
    _LAT['s'] = a.interpolate(limit_area='inside')


def profile(args):
    f, t = args
    from huxt import huxt_inputs as Hin
    try:
        s = _LAT['s']
        # before the ACE file starts (2011-01-01; maps from 2010-12 feed the first chunk's warm-up): ACE latitude one year
        # later, Earth's heliographic latitude being periodic in the year to well under a degree
        lat = s.asof(t) if t >= s.first_valid_index() else s.asof(t + pd.DateOffset(years=1))
        v, vlon, vlat, *_ = Hin.get_WSA_maps(f)
        prof = np.array([np.interp(np.deg2rad(lat), vlat.value, v[:, i].value) for i in range(vlon.size)])
        return np.interp(LON, vlon.value, prof, period=2 * np.pi).astype(np.float32)
    except Exception:
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(f'{DIR}/*/*/vel_*.fits') if os.path.getsize(f) > 0)
    t = pd.to_datetime([re.search(r'vel_(\d{12})', os.path.basename(f)).group(1) for f in files], format='%Y%m%d%H%M')
    with Pool(a.workers, initializer=_init) as pool:
        P = pool.map(profile, list(zip(files, t)), chunksize=64)
    ok = np.array([p is not None for p in P])
    os.makedirs('runs/part1_wsa', exist_ok=True)
    np.savez_compressed('runs/part1_wsa/wsa45_profiles.npz', map_time=t[ok].values, lon=LON,
                        v1au=np.stack([p for p in P if p is not None]), path=np.array(files)[ok])
    print(f'{ok.sum()} profiles of {len(files)} maps ({t.min()} .. {t.max()}); failed {(~ok).sum()}')
