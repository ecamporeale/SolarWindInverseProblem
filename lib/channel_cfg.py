"""AIA channel switch for the pixel-level ranking pipeline (env AIA_CHANNEL, default 193).

193 keeps the original file names (pixel_features_aia.pkl, *_pixel.*); any other channel uses
pixel_features_aia<ch>.pkl, the hourly quality manifest aia_fits_hourly_quality_<ch>.csv and the suffix _pixel<ch>.
"""
import os

import pandas as pd

CHANNEL = os.environ.get('AIA_CHANNEL', '193')
VARIANT = os.environ.get('AIA_VARIANT', '')          # e.g. 'clip90', 'tmed12' (normalising-median variants)
IS_193 = CHANNEL == '193'
PIXEL_AIA = 'data/pixel_features_aia.pkl' if IS_193 else f'data/pixel_features_aia{CHANNEL}.pkl'
SUF_PIXEL = '_pixel' if IS_193 else f'_pixel{CHANNEL}'
if VARIANT:
    PIXEL_AIA = PIXEL_AIA.replace('.pkl', f'_{VARIANT}.pkl')
    SUF_PIXEL = f'{SUF_PIXEL}_{VARIANT}'
# Image-to-ACE alignment of the ranking rows (env RANK_ALIGN): 'lag96' = ACE v(t + 4 d) for every image t (default);
# 'backmap' = every ACE hour T paired with the frame at its ballistic source time (as in the look-alike analysis)
ALIGN = os.environ.get('RANK_ALIGN', 'lag96')
if ALIGN != 'lag96':
    SUF_PIXEL = f'{SUF_PIXEL}_{ALIGN}'


def good_hours():
    """Set of hourly timestamps (rounded) with a QUALITY == 0 frame for the channel."""
    if IS_193:
        from aia_frames import usable
        q = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
        return set(q.loc[usable(q), 'obstime'].dt.round('h'))
    q = pd.read_csv(f'data/preprocessed_window/aia_fits_hourly_quality_{CHANNEL}.csv', parse_dates=['hour'])
    return set(q.loc[q.status == 'ok', 'hour'])
