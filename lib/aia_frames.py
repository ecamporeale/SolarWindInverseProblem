"""Usability rule for hourly AIA frames, applied everywhere frames are selected (dataset build, pixel features, ranking).

A frame is usable when QUALITY == 0 and EXPTIME >= MIN_EXPTIME. Very short flare-mode exposures (automatic exposure
control during large flares, down to 5 ms) leave the quiet Sun at ~1 DN, so dark-area features explode
(e.g. 2017-09-06 12:00: 27 % dark area vs 5 %); such hours use a replacement frame (fix_short_exposure_frames.py).
"""
MIN_EXPTIME = 0.1        # s
NORMAL_EXPTIME = 1.0     # s, required for a replacement frame


def usable(q):
    """Boolean mask over a quality table with columns quality, exptime."""
    return (q.quality == 0) & (q.exptime >= MIN_EXPTIME)
