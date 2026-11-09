#!/usr/bin/env python3
"""
Preprocess local AIA full-disk images into Carrington equatorial intensity slices.

This script reads image files from a local directory, projects each image
onto Carrington longitude/latitude, masks the non-visible hemisphere for the
current observer, extracts a 1D cut along the solar equator/ecliptic, and
and saves the resulting time series for later FNO boundary mapping.

Output file format: 
  - times: (Nt,) ISO time strings
  - image_slices: (Nphi, Nt) intensity values along Carrington longitude
  - carrington_longitude: (Nphi,) longitude values in degrees
    - visibility_mask: (Nphi, Nt) frontside visibility mask (1=visible, 0=non-visible)

Example:
  python preprocess_image.py --input-dir data/aia_fits --output data/aia_equatorial_slices.npz
"""

import argparse
import concurrent.futures
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

_Map = None
_SkyCoord = None
_frames = None
_u = None
_map_coordinates = None
_HeliocentricMeanEcliptic = None

_aiapy_degradation = None
_aiapy_get_correction_table = None
_aiapy_correction_table = None  # fetched once (network call), reused for every file
# Last measured AIA-193 calibration epoch in the JSOC correction table (VER_NUM 10,
# created 2020-11-02): the final segment starts 2020-06-01T12:00 with zero slope and
# runs to 2030, i.e. a flat placeholder, not a measurement.
_AIA193_CALIBRATION_CUTOFF = datetime(2020, 6, 1, 12, 0)

# Helioviewer jp2gen byte-scaling parameters for AIA 193 (verified against
# https://github.com/Helioviewer-Project/jp2gen idl/sdo/aia/hvs_version5_aia.pro,
# b.details[3] for measurement '193'):
#   A (DN) -> B = A / dataExptime=1.0 (no-op for this channel) -> C = clip(B, dataMin, dataMax)
#   -> D = BYTSCL(log10(C))  in [0, 255]
# So byte value D inverts to an approximate DN via a linear un-scaling in
# log10 space between log10(dataMin) and log10(dataMax). This is lossy
# (256 discrete levels across ~2.1 decades, ~1.9% per step) and clips any
# true DN outside [20, 2500] to that boundary -- confirmed on real data to
# affect a small fraction of on-disk pixels (0-2.6% at the floor across
# 2011-2022 test frames), not a dominant source of error.
_AIA193_JP2_DATA_MIN = 20.0
_AIA193_JP2_DATA_MAX = 2500.0


def _lazy_aiapy_imports():
    global _aiapy_degradation, _aiapy_get_correction_table
    if _aiapy_degradation is None:
        try:
            from aiapy.calibrate import degradation
            from aiapy.calibrate.util import get_correction_table
        except ImportError as exc:
            raise ImportError(
                "AIA degradation correction requires aiapy. Install with: pip install aiapy"
            ) from exc
        _aiapy_degradation = degradation
        _aiapy_get_correction_table = get_correction_table
    return _aiapy_degradation, _aiapy_get_correction_table


def set_aia_correction_table(table):
    """Pre-seeds this process's correction-table cache with an already-fetched table.

    Used to avoid every worker process in a ProcessPoolExecutor independently
    fetching the table over the network -- with 24 workers hitting JSOC at
    once, some connections were being refused, silently dropping files. The
    fix is to fetch once in the main process and pass the table to each
    worker call, which seeds it here before any per-file work runs.
    """
    global _aiapy_correction_table
    _aiapy_correction_table = table


def _get_aia_correction_table():
    """Fetches the JSOC degradation correction table once per process (network call) and caches it."""
    global _aiapy_correction_table
    if _aiapy_correction_table is None:
        _, get_correction_table = _lazy_aiapy_imports()
        _aiapy_correction_table = get_correction_table("jsoc")
    return _aiapy_correction_table


def invert_aia193_jp2_byte_scale(byte_data: np.ndarray) -> np.ndarray:
    """Recovers an approximate linear DN image from Helioviewer AIA-193 JP2 byte values."""
    log_min = np.log10(_AIA193_JP2_DATA_MIN)
    log_max = np.log10(_AIA193_JP2_DATA_MAX)
    frac = np.clip(np.asarray(byte_data, dtype=np.float64), 0.0, 255.0) / 255.0
    log_dn = log_min + frac * (log_max - log_min)
    return np.power(10.0, log_dn).astype(np.float32)


def apply_aia193_degradation_correction(dn_data: np.ndarray, obstime, allow_uncalibrated: bool = False) -> np.ndarray:
    """Divides an AIA-193 DN image by the aiapy/JSOC sensitivity factor at obstime.

    No extrapolation: after _AIA193_CALIBRATION_CUTOFF this raises, unless
    allow_uncalibrated=True, in which case the factor is frozen at the cutoff value
    (equal to the table's own flat placeholder) and the result must be treated as
    uncalibrated. Only scale-invariant features are meaningful on such frames.
    """
    from astropy.time import Time
    obstime = Time(obstime)
    obstime_dt = obstime.datetime
    if obstime_dt > _AIA193_CALIBRATION_CUTOFF:
        if not allow_uncalibrated:
            raise ValueError(f"{obstime_dt} is after the last AIA-193 calibration epoch "
                             f"{_AIA193_CALIBRATION_CUTOFF}; pass allow_uncalibrated=True to proceed")
        obstime = Time(_AIA193_CALIBRATION_CUTOFF)
    degradation, _ = _lazy_aiapy_imports()
    u = _lazy_imports()[3]
    factor = float(degradation(193 * u.angstrom, obstime, correction_table=_get_aia_correction_table()))
    return (dn_data / factor).astype(np.float32)


AIA_FILENAME_RE = re.compile(
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})__(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
)

HMI_FILENAME_RE = re.compile(
    r"hmi\.M_720s\."
    r"(?P<year>\d{4})"
    r"(?P<month>\d{2})"
    r"(?P<day>\d{2})_"
    r"(?P<hour>\d{2})"
    r"(?P<minute>\d{2})"
    r"(?P<second>\d{2})"
    r"_TAI\.fits$"
)


def _lazy_imports():
    global _Map, _SkyCoord, _frames, _u, _map_coordinates, _HeliocentricMeanEcliptic
    if _Map is None:
        try:
            import astropy.units as u
            from astropy.coordinates import HeliocentricMeanEcliptic, SkyCoord
            from sunpy.map import Map
            from sunpy.coordinates import frames
        except ImportError as exc:
            raise ImportError(
                "Failed to import SunPy map dependencies. "
                f"Original error: {exc}. "
                "Install required packages, e.g. pip install astropy sunpy matplotlib scipy "
                "or pip install 'sunpy[map]'."
            ) from exc

        try:
            from scipy.ndimage import map_coordinates
        except ImportError as exc:
            raise ImportError(
                "This script requires scipy. Install with: pip install scipy"
            ) from exc

        _u = u
        _Map = Map
        _SkyCoord = SkyCoord
        _frames = frames
        _map_coordinates = map_coordinates
        _HeliocentricMeanEcliptic = HeliocentricMeanEcliptic

    return _Map, _SkyCoord, _frames, _u, _map_coordinates, _HeliocentricMeanEcliptic


SUPPORTED_EXTENSIONS = {".fits", ".fts", ".fit", ".fz", ".jp2", ".j2k", ".jpg",".jpeg"}


def _progress_message(stage: str, count: int, total: int | None = None, extra: str = ""):
    if total is not None and total > 0:
        pct = 100.0 * count / total
        base = f"[{stage}] {count:,}/{total:,} ({pct:5.1f}%)"
    else:
        base = f"[{stage}] {count:,}"
    if extra:
        base = f"{base} | {extra}"
    print(base, flush=True)


def parse_image_filename_time(path, instrument):
    if instrument == "aia":
        match = AIA_FILENAME_RE.match(path.name)
    elif instrument == "hmi":
        match = HMI_FILENAME_RE.match(path.name)
    else:
        match = None
    
    if match is None:
        return None

    parts = {k: int(v) for k, v in match.groupdict().items()}
    return datetime(
        parts["year"],
        parts["month"],
        parts["day"],
        parts["hour"],
        parts["minute"],
        parts["second"],
    )


def thin_files_by_min_separation(
    files: list[Path],
    min_separation_minutes: float,
    verbose: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
    instrument: str = "aia",
) -> list[tuple[Path, str]]:
    if min_separation_minutes <= 0:
        selected = []
        for path in files:
            ts = parse_image_filename_time(path, instrument)
            if ts is not None:
                selected.append((path, ts.isoformat()))
        return selected

    tick_seconds = float(min_separation_minutes) * 60.0
    epoch = datetime(1970, 1, 1)

    parsed = [(path, parse_image_filename_time(path, instrument)) for path in files]
    known = [(path, ts) for path, ts in parsed if ts is not None]
    unknown_time = len(parsed) - len(known)
    if not known:
        _progress_message(
            "cadence",
            0,
            0,
            f"tick={min_separation_minutes:g} min, dropped_files=0, unmatched_ticks=0, unknown_time={unknown_time:,}",
        )
        return []

    def _ceil_to_tick(ts: datetime) -> datetime:
        offset = (ts - epoch).total_seconds()
        tick_idx = int(np.ceil(offset / tick_seconds))
        return epoch + timedelta(seconds=tick_idx * tick_seconds)

    def _floor_to_tick(ts: datetime) -> datetime:
        offset = (ts - epoch).total_seconds()
        tick_idx = int(np.floor(offset / tick_seconds))
        return epoch + timedelta(seconds=tick_idx * tick_seconds)

    first_source = min(ts for _, ts in known)
    last_source = max(ts for _, ts in known)
    tick_start = _ceil_to_tick(start if start is not None else first_source)
    tick_end = _floor_to_tick(end if end is not None else last_source)
    if tick_end < tick_start:
        return []

    best_by_tick: dict[datetime, tuple[float, datetime, Path]] = {}
    for path, ts in known:
        offset = (ts - epoch).total_seconds()
        tick_idx = int(np.floor(offset / tick_seconds + 0.5))
        tick_time = epoch + timedelta(seconds=tick_idx * tick_seconds)
        if tick_time < tick_start or tick_time > tick_end:
            continue

        distance_seconds = abs((ts - tick_time).total_seconds())
        current = best_by_tick.get(tick_time)
        if current is None or distance_seconds < current[0] or (
            np.isclose(distance_seconds, current[0]) and ts < current[1]
        ):
            best_by_tick[tick_time] = (distance_seconds, ts, path)

    selected_ticks = sorted(best_by_tick.keys())
    selected = [(best_by_tick[tick][2], tick.isoformat()) for tick in selected_ticks]
    total_ticks = int(round((tick_end - tick_start).total_seconds() / tick_seconds)) + 1
    dropped = max(0, len(known) - len(selected))

    _progress_message(
        "cadence",
        len(selected),
        total_ticks,
        f"tick={min_separation_minutes:g} min, dropped_files={dropped:,}, unmatched_ticks={max(total_ticks - len(selected), 0):,}, unknown_time={unknown_time:,}",
    )
    if verbose and unknown_time > 0:
        verbose_print(
            f"Cadence filter skipped {unknown_time} files with unparseable filename timestamps.",
            verbose,
        )
    return selected

def match_files_to_target_times(
    files,
    target_times,
    instrument,
    max_dt_minutes=15,
):
    """For each target time, finds the nearest candidate file within tolerance.

    Sorts candidates once and uses bisect per target instead of an O(N*M)
    nested scan (N target times x M candidate files) -- with N~60,000 and
    M~230,000 that nested version is ~14 billion comparisons in pure Python,
    which in practice never finished in reasonable time. After sorting by
    timestamp, the true nearest neighbor to a target is always the candidate
    immediately before or after the bisect insertion point, so this is exact,
    not an approximation.
    """
    import bisect

    parsed = []
    for path in files:
        ts = parse_image_filename_time(path, instrument)
        if ts is not None:
            parsed.append((ts, path))
    parsed.sort(key=lambda item: item[0])
    parsed_times = [ts for ts, _ in parsed]

    tol_seconds = max_dt_minutes * 60
    selected = []

    for target in target_times:
        target_dt = datetime.fromisoformat(str(target))
        idx = bisect.bisect_left(parsed_times, target_dt)

        best_path = None
        best_dt = None
        for cand_idx in (idx - 1, idx):
            if 0 <= cand_idx < len(parsed_times):
                dt = abs((parsed_times[cand_idx] - target_dt).total_seconds())
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_path = parsed[cand_idx][1]

        if best_path is not None and best_dt <= tol_seconds:
            selected.append((best_path, target_dt.isoformat()))

    return selected

def _candidate_date_dirs(input_dir: Path, start: datetime, end: datetime) -> list[Path]:
    candidate_dirs: list[Path] = []
    for year_dir in sorted(
        p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4
    ):
        year = int(year_dir.name)
        if year < start.year or year > end.year:
            continue

        month_dirs = sorted(
            p for p in year_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2
        )
        if not month_dirs:
            candidate_dirs.append(year_dir)
            continue

        for month_dir in month_dirs:
            month = int(month_dir.name)
            if year == start.year and month < start.month:
                continue
            if year == end.year and month > end.month:
                continue

            day_dirs = sorted(
                p for p in month_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2
            )
            if not day_dirs:
                candidate_dirs.append(month_dir)
                continue

            for day_dir in day_dirs:
                day = int(day_dir.name)
                if year == start.year and month == start.month and day < start.day:
                    continue
                if year == end.year and month == end.month and day > end.day:
                    continue
                candidate_dirs.append(day_dir)

    return candidate_dirs


def list_image_files(
    input_dir: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    progress_every: int = 50000,
    instrument: str = "aia",
):
    if start is not None and end is not None:
        candidate_dirs = _candidate_date_dirs(input_dir, start, end)
    elif start is not None:
        candidate_dirs = _candidate_date_dirs(input_dir, start, start.replace(month=12, day=31))
    elif end is not None:
        candidate_dirs = _candidate_date_dirs(input_dir, datetime(end.year, 1, 1), end)
    else:
        candidate_dirs = []

    if candidate_dirs:
        if len(candidate_dirs) == 1 and candidate_dirs[0] == input_dir:
            search_dirs = [input_dir]
        else:
            search_dirs = candidate_dirs
    else:
        search_dirs = [input_dir]

    files = []
    inspected = 0
    matched = 0
    t0 = time.monotonic()
    step = max(progress_every, 1)
    for directory in search_dirs:
        for root, _, names in os.walk(directory):
            for name in names:
                inspected += 1
                path = Path(root) / name
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    if inspected % step == 0:
                        elapsed = time.monotonic() - t0
                        _progress_message(
                            "scan",
                            inspected,
                            None,
                            f"matched={matched:,}, elapsed={elapsed:.1f}s",
                        )
                    continue

                file_time = parse_image_filename_time(path, instrument)
                if file_time is not None:
                    if start is not None and file_time < start:
                        continue
                    if end is not None and file_time > end:
                        continue

                files.append(path)
                matched += 1

                if inspected % step == 0:
                    elapsed = time.monotonic() - t0
                    _progress_message(
                        "scan",
                        inspected,
                        None,
                        f"matched={matched:,}, elapsed={elapsed:.1f}s",
                    )

    elapsed = time.monotonic() - t0
    _progress_message("scan", inspected, None, f"matched={matched:,}, elapsed={elapsed:.1f}s (done)")
    return sorted(files)


def parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: {value}. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
        ) from exc


def filter_image_files_by_time(
    files: list[Path],
    start: datetime | None,
    end: datetime | None,
    verbose: bool = False,
    progress_every: int = 2000,
) -> list[Path]:
    selected = []
    verbose_print("Loading image dependency modules...", verbose)

    Map, _, _, _, _, _ = _lazy_imports()
    step = max(progress_every, 1)
    for idx, path in enumerate(files, start=1):
        try:
            image_map = Map(path)
            obs_time = image_map.date.datetime
        except Exception as exc:
            verbose_print(f"Skipping unreadable image file {path.name}: {exc}", verbose)
            continue

        if start is not None and obs_time < start:
            continue
        if end is not None and obs_time > end:
            continue

        selected.append(path)

        if idx % step == 0:
            _progress_message("time-filter", idx, len(files), f"selected={len(selected):,}")

    _progress_message("time-filter", len(files), len(files), f"selected={len(selected):,} (done)")

    if verbose:
        verbose_print(f"Selected {len(selected)} image files after time filtering", verbose)
    return selected


def safe_load_image_map(path, verbose=False):
    Map, _, _, _, _, _ = _lazy_imports()

    try:
        return Map(path)

    except Exception as exc:
        verbose_print(
            f"Skipping image file {path.name}: "
            f"{type(exc).__name__}: {exc}",
            verbose,
        )
        return None
    


def _process_one_image_file(
    path_str: str,
    n_longitude: int,
    fill_value: float,
    instrument = "aia",
    correction_table = None,
):
    path = Path(path_str)
    if instrument == "aia" and correction_table is not None:
        set_aia_correction_table(correction_table)
    image_map = safe_load_image_map(path, verbose=False)
    if image_map is None:
        return None, None, f"corrupted/unreadable: {path.name}"

    try:
        _, intensity, visibility_mask = make_longitude_slice(
            image_map,
            n_longitude=n_longitude,
            fill_value=fill_value, instrument = instrument,
        )
    except Exception as exc:
        return None, None, f"projection failure: {path.name}: {exc}"

    return intensity, visibility_mask, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess image files into Carrington equatorial intensity slices."
    )

    parser.add_argument(
    "--instrument",
    choices=["aia", "hmi"],
    required=True,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing local image files.",
    )
    parser.add_argument(
        "--output",
        default="data/image_equatorial_slices.npz",
        help="Output .npz file for preprocessed image slices.",
    )

    parser.add_argument(
        "--n-longitude",
        type=int,
        default=360,
        help="Number of Carrington longitude samples along the equator.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=np.nan,
        help="Value used for sample points outside the disk.",
    )
    parser.add_argument(
        "--start",
        type=parse_iso_datetime,
        default=None,
        help="Start of the time window to preprocess (ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument(
        "--end",
        type=parse_iso_datetime,
        default=None,
        help="End of the time window to preprocess (ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during image preprocessing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N scanned/processed files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for image frame processing.",
    )
    parser.add_argument(
        "--min-separation-minutes",
        type=float,
        default=30.0,
        help="Minimum time separation between kept image frames (minutes).",
    )
    return parser.parse_args()


def verbose_print(message: str, verbose: bool = True):
    if verbose:
        print(message, flush=True)


def _is_effectively_blank_frame(data: np.ndarray, instrument = "aia") -> bool:
    """Heuristic detector for corrupted/near-empty image frames.

    Some JP2 files in the archive decode to almost-all-zero images with only a
    handful of non-zero pixels. These are not meaningful observations and
    should be treated as missing data rather than as genuine dark emission.
    """

    if instrument == "hmi":
        finite = np.isfinite(data)

        if finite.sum() == 0:
            return True

        vals = data[finite]

        return np.nanstd(vals) < 1e-6
    
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return True

    vals = arr[finite]
    nonzero_frac = float(np.count_nonzero(vals > 0.0)) / float(vals.size)
    mean_val = float(vals.mean())
    max_val = float(vals.max())

    # Normal image frames have broad non-zero coverage. Corrupted
    # JP2 decodes observed in this dataset have nonzero_frac ~ 4e-5 and mean
    # intensity ~ 5e-5 despite an isolated max pixel of 255.
    return (nonzero_frac < 1e-3 and mean_val < 1e-2) or max_val <= 0.0

def sample_one_lat(
    lat_deg,
    image_map,
    trace_lon,
    lon_grid,
    carrington_frame,
    ecliptic_frame_cls,
    fill_value,
    image_data,
    SkyCoord,
    frames,
    u,
    map_coordinates,
):

    sample_coords = SkyCoord(
        trace_lon * u.deg,
        np.full_like(trace_lon, lat_deg) * u.deg,
        1.0 * u.solRad,
        frame=ecliptic_frame_cls(obstime=image_map.date),
    )

    carrington_coords = sample_coords.transform_to(carrington_frame)
    sample_lon_carr_deg = np.mod(
        carrington_coords.lon.to_value(u.deg),
        360.0,
    )

    hgs_frame = frames.HeliographicStonyhurst(obstime=image_map.date)
    observer_hgs = image_map.observer_coordinate.transform_to(hgs_frame)
    sample_hgs = sample_coords.transform_to(hgs_frame)

    lon_rad = sample_hgs.lon.to_value(u.rad)
    lat_rad = sample_hgs.lat.to_value(u.rad)
    lon0_rad = observer_hgs.lon.to_value(u.rad)
    lat0_rad = observer_hgs.lat.to_value(u.rad)

    cos_central_angle = (
        np.sin(lat_rad) * np.sin(lat0_rad)
        + np.cos(lat_rad)
        * np.cos(lat0_rad)
        * np.cos(lon_rad - lon0_rad)
    )

    visible_mask = cos_central_angle >= 0.0

    try:
        pixel_coords = image_map.world_to_pixel(sample_coords)
    except Exception:
        try:
            target_frame = image_map.coordinate_frame
        except Exception:
            target_frame = frames.HeliographicStonyhurst(
                obstime=image_map.date
            )

        transformed = sample_coords.transform_to(target_frame)
        pixel_coords = image_map.world_to_pixel(transformed)

    x = pixel_coords.x.to_value(u.pix)
    y = pixel_coords.y.to_value(u.pix)

    ny, nx = image_map.data.shape

    in_bounds = (
        (x >= 0.0)
        & (x <= (nx - 1))
        & (y >= 0.0)
        & (y <= (ny - 1))
    )

    values = map_coordinates(
        image_data,
        [y, x],
        order=1,
        mode="constant",
        cval=fill_value,
        prefilter=False,
    ).astype(np.float32)

    sample_visibility = (
        visible_mask
        & in_bounds
        & np.isfinite(values)
    )

    values[~sample_visibility] = fill_value

    order = np.argsort(sample_lon_carr_deg)

    sample_lon_sorted = sample_lon_carr_deg[order]
    values_sorted = values[order]
    vis_sorted = sample_visibility[order]

    sample_lon_sorted, unique_idx = np.unique(
        sample_lon_sorted,
        return_index=True,
    )

    values_sorted = values_sorted[unique_idx]
    vis_sorted = vis_sorted[unique_idx]

    lon_ext = np.concatenate(
        [sample_lon_sorted, sample_lon_sorted[:1] + 360.0]
    )

    vis_ext = np.concatenate(
        [
            vis_sorted.astype(np.float32),
            vis_sorted[:1].astype(np.float32),
        ]
    )

    weighted_vals = (
        np.nan_to_num(
            values_sorted,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        * vis_sorted.astype(np.float32)
    )

    val_ext = np.concatenate(
        [weighted_vals, weighted_vals[:1]]
    )

    interp_vis = np.interp(
        lon_grid,
        lon_ext,
        vis_ext,
    ).astype(np.float32)

    interp_vals = np.interp(
        lon_grid,
        lon_ext,
        val_ext,
    ).astype(np.float32)

    return interp_vals, interp_vis

def make_longitude_slice(
    image_map,
    n_longitude,
    fill_value=np.nan,
    instrument = "aia",):
    """Extract a 1D intensity cut indexed by Carrington longitude.

    The sampled curve follows the true solar
    ecliptic at the map obstime, then gets re-indexed onto a uniform
    Carrington-longitude grid. This keeps the output shape fixed at
    ``(n_longitude,)`` while leaving backside / non-visible longitudes masked.
    """
    verbose_print("Loading image dependency modules...", False)

    if _is_effectively_blank_frame(image_map.data, instrument = instrument):
        lon_grid = np.linspace(0.0, 360.0, n_longitude, endpoint=False, dtype=np.float64)
        return (
            lon_grid,
            np.full(n_longitude, fill_value, dtype=np.float32),
            np.zeros(n_longitude, dtype=np.float32),
        )

    _, SkyCoord, frames, u, map_coordinates, HeliocentricMeanEcliptic = _lazy_imports()

    lon_grid = np.linspace(0.0, 360.0, n_longitude, endpoint=False, dtype=np.float64)

    try:
        observer = image_map.observer_coordinate
        carrington_frame = frames.HeliographicCarrington(
            observer=observer,
            obstime=image_map.date,
        )
    except Exception:
        observer = image_map.observer_coordinate
        carrington_frame = frames.HeliographicCarrington(obstime=image_map.date)

    ecliptic_frame_cls = HeliocentricMeanEcliptic
    if ecliptic_frame_cls is None:
        raise RuntimeError(
            "Astropy HeliocentricMeanEcliptic frame is unavailable in this environment"
        )

    dense_count = max(4 * int(n_longitude), 1440)
    trace_lon = np.linspace(
        0.0,
        360.0,
        dense_count,
        endpoint=False,
        dtype=np.float64,
    )

    if instrument == "aia":
        wavelnth = image_map.meta.get("wavelnth")
        if int(wavelnth) != 193:
            raise ValueError(
                f"AIA degradation correction is only calibrated for the 193 channel, "
                f"got wavelnth={wavelnth} from {getattr(image_map, 'name', '<unknown>')}"
            )
        dn_approx = invert_aia193_jp2_byte_scale(image_map.data)
        image_data = apply_aia193_degradation_correction(dn_approx, image_map.date)
    else:
        image_data = image_map.data.astype(np.float32)
    all_vals = []
    all_vis = []

    for lat_deg in range(-3, 4):
        vals, vis = sample_one_lat(
        lat_deg=lat_deg,
        image_map=image_map,
        trace_lon=trace_lon,
        lon_grid=lon_grid,
        carrington_frame=carrington_frame,
        ecliptic_frame_cls=ecliptic_frame_cls,
        fill_value=fill_value,
        image_data=image_data,
        SkyCoord=SkyCoord,
        frames=frames,
        u=u,
        map_coordinates=map_coordinates,
        )
        all_vals.append(vals)
        all_vis.append(vis)

    all_vals = np.stack(all_vals, axis=0)
    all_vis = np.stack(all_vis, axis=0)

    val_sum = np.sum(all_vals, axis=0)
    vis_sum = np.sum(all_vis, axis=0)

    out_visible = vis_sum > 0.0

    out_values = np.full(
        n_longitude,
        fill_value,
        dtype=np.float32,
    )


    
    if np.any(out_visible):
        out_values[out_visible] = (
        val_sum[out_visible]
        / np.maximum(vis_sum[out_visible], 1e-6)
        )

        visible_values = out_values[out_visible]

        if (
            (not np.any(np.isfinite(visible_values)))
            or (np.nanmax(visible_values) <= 0.0)
        ):
            out_values[:] = fill_value
            out_visible[:] = False

    return (
        lon_grid,
        out_values,
        out_visible.astype(np.float32),
    )
    



def preprocess_image_window(
    input_dir: Path,
    output_path: Path,
    n_longitude: int = 360,
    fill_value: float = np.nan,
    start: datetime | None = None,
    end: datetime | None = None,
    verbose: bool = False,
    progress_every: int = 2000,
    workers: int = 1,
    min_separation_minutes: float = 30.0,
    instrument: str = "aia",
    target_times = None,
) -> Path:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    verbose_print(f"Scanning AIA input directory {input_dir} for supported files...", verbose)
    fits_files = list_image_files(
        input_dir,
        start=start,
        end=end,
        progress_every=max(progress_every, 1000), instrument=instrument,
    )
    if not fits_files:
        raise FileNotFoundError(
            f"No {instrument.upper()} files found in {input_dir}. Expected extensions: {SUPPORTED_EXTENSIONS}"
        )

    if start or end:
        parsed_time = {path: parse_image_filename_time(path, instrument=instrument) for path in fits_files}
        unknown_time_paths = [path for path, ts in parsed_time.items() if ts is None]
        unknown_time_count = len(unknown_time_paths)

        if unknown_time_count == 0:
            # Fast path: filename timestamps already constrained to the requested window.
            _progress_message(
                "time-filter",
                len(fits_files),
                len(fits_files),
                "selected by filename timestamp (metadata read skipped)",
            )
            verbose_print(
                "Skipping metadata time-filter because all candidate filenames contain parseable timestamps.",
                verbose,
            )
        elif all(path.suffix.lower() in {".jp2", ".j2k"} for path in unknown_time_paths):
            # JP2/J2K time comes from filename convention in this pipeline; do not trigger expensive metadata reads.
            fits_files = [path for path, ts in parsed_time.items() if ts is not None]
            _progress_message(
                "time-filter",
                len(fits_files),
                len(fits_files),
                "selected by filename timestamp; dropped JP2/J2K files with unparseable names",
            )
            verbose_print(
                f"Dropped {unknown_time_count} JP2/J2K files with unparseable timestamps in filename.",
                verbose,
            )
        else:
            verbose_print(
                f"Filtering AIA files to window {start or '-inf'} -> {end or '+inf'} "
                f"using metadata for {unknown_time_count} filename-unparseable files",
                verbose,
            )
            fits_files = filter_image_files_by_time(
                fits_files,
                start=start,
                end=end,
                verbose=verbose,
                progress_every=max(progress_every // 5, 200),
            )

    if not fits_files:
        raise FileNotFoundError(
            f"No {instrument.upper()} files found in {input_dir} for requested window {start} to {end}"
        )

    if target_times is None:
        selected_files = thin_files_by_min_separation(
            fits_files,
            min_separation_minutes=min_separation_minutes,
            verbose=verbose,
            start=start,
            end=end,
            instrument=instrument,
        )
    else:
        selected_files = match_files_to_target_times(
            fits_files,
            target_times=target_times,
            instrument=instrument,
        )
 
    if not selected_files:
        raise FileNotFoundError(
            f"No {instrument.upper()} files remain after applying minimum frame separation filter"
        )

    fits_files = [path for path, _ in selected_files]
    selected_tick_times = [tick_time for _, tick_time in selected_files]

    print(f"Found {len(fits_files)} {instrument.upper()} files in {input_dir}")

    times = []
    slices = []
    visibility_masks = []
    lon_grid = None
    processed = 0
    process_started = time.monotonic()
    step = max(progress_every, 1)

    # Lon grid is invariant across frames.
    lon_grid = np.linspace(0.0, 360.0, n_longitude, endpoint=False)

    if workers > 1:
        if verbose:
            print(f"Using parallel {instrument.upper()} processing with workers={workers}", flush=True)

        # Fetch the degradation correction table ONCE here (network call) and
        # pass it to every worker, instead of each of `workers` worker
        # processes independently fetching it on first use -- with many
        # workers hitting JSOC simultaneously, some connections were being
        # refused, silently dropping files.
        correction_table = _get_aia_correction_table() if instrument == "aia" else None

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, (intensity, visibility_mask, error_msg) in enumerate(

                executor.map(
                    _process_one_image_file,
                    [str(p) for p in fits_files],
                    [n_longitude] * len(fits_files),
                    [fill_value] * len(fits_files),
                    [instrument] * len(fits_files),
                    [correction_table] * len(fits_files),
                    chunksize=max(1, min(64, len(fits_files) // max(workers * 8, 1)))
                ),
                start=1,
            ):
                if error_msg is not None:
                    verbose_print(f"Skipping {instrument.upper()} file: {error_msg}", verbose)
                    if (idx % step == 0) or (idx == len(fits_files)):
                        elapsed = time.monotonic() - process_started
                        _progress_message(
                            "process",
                            idx,
                            len(fits_files),
                            f"usable={processed:,}, elapsed={elapsed:.1f}s",
                        )
                    continue

                slices.append(intensity)
                visibility_masks.append(visibility_mask)
                times.append(selected_tick_times[idx - 1])
                processed += 1

                if (idx % step == 0) or (idx == len(fits_files)):
                    elapsed = time.monotonic() - process_started
                    _progress_message(
                        "process",
                        idx,
                        len(fits_files),
                        f"usable={processed:,}, elapsed={elapsed:.1f}s",
                    )
    else:
        for idx, fits_path in enumerate(fits_files, start=1):
            image_map = safe_load_image_map(fits_path, verbose=verbose)
            if image_map is None:
                continue

            try:
                _, intensity, visibility_mask = make_longitude_slice(
                    image_map,
                    n_longitude=n_longitude,
                    fill_value=fill_value, instrument=instrument,
                )
            except Exception as exc:
                verbose_print(
                    f"Skipping {instrument.upper()} file {fits_path.name} after projection failure: {exc}",
                    verbose,
                )
                continue

            slices.append(intensity)
            visibility_masks.append(visibility_mask)
            times.append(selected_tick_times[idx - 1])
            processed += 1

            if (idx % step == 0) or (idx == len(fits_files)):
                elapsed = time.monotonic() - process_started
                _progress_message(
                    "process",
                    idx,
                    len(fits_files),
                    f"usable={processed:,}, elapsed={elapsed:.1f}s",
                )

    if not slices:
        raise RuntimeError(f"No valid {instrument.upper()} frames could be decoded in the requested time window.")

    image_array = np.stack(slices, axis=1)
    visibility_array = np.stack(visibility_masks, axis=1).astype(np.float32)
    print(
        f"Processed {image_array.shape[1]} time frames, shape={image_array.shape} "
        f"(Nphi x Nt)"
    )

    slice_key = f"{instrument}_slices"

    np.savez(
        output_path,
        times=np.array(times, dtype="<U32"),
        carrington_longitude=lon_grid,
        visibility_mask=visibility_array,
        **{slice_key: image_array},
    )
    print(f"Saved preprocessed {instrument.upper()} slices to {output_path}")
    return output_path


def main():
    args = parse_args()
    print(
        f"Starting preprocess_image: {args.input_dir} -> {args.output} "
        f"(verbose={args.verbose})",
        flush=True,
    )
    preprocess_image_window(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        n_longitude=args.n_longitude,
        fill_value=args.fill_value,
        start=args.start,
        end=args.end,
        verbose=args.verbose,
        progress_every=args.progress_every,
        workers=max(1, args.workers),
        min_separation_minutes=max(0.0, args.min_separation_minutes),
        instrument=args.instrument,
    )


if __name__ == "__main__":
    main()
