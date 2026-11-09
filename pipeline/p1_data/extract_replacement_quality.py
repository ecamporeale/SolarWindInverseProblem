"""Append QUALITY / pixel-ESWF rows for the replacement frames (fetch_replacement_frames.py) to
data/preprocessed_window/aia_fits_hourly_quality_eswf.csv, using the same per-frame code as
extract_fits_quality_eswf.py. Frames already listed are skipped."""
import concurrent.futures

import pandas as pd

from extract_fits_quality_eswf import _process, OUT
from preprocess_solar_image import _get_aia_correction_table

m = pd.read_csv('data/aia_synoptic_replacements_manifest.csv')
q = pd.read_csv(OUT)
new = sorted(set(m.loc[m.status == 'ok', 'file']) - set(q.path))
print(f'{len(new)} replacement frames to add', flush=True)
table = _get_aia_correction_table()
with concurrent.futures.ProcessPoolExecutor(48) as ex:
    rows = list(ex.map(_process, [(p, table) for p in new], chunksize=8))
add = pd.DataFrame(rows)
out = pd.concat([q, add], ignore_index=True)
out['obstime'] = pd.to_datetime(out.obstime)
out.sort_values('obstime').to_csv(OUT, index=False)
print(f'added {len(add)}; QUALITY==0 among them: {(add.quality == 0).sum()}', flush=True)
