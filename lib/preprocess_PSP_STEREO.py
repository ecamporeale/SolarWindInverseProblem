#!/usr/bin/env python3
"""
Preprocess PSP and STEREO-A observations for a requested time window.

This module can be called internally by preprocess_time_window.py or run directly
for debugging and data checks.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AU_IN_KM = 149597870.7

ALLOWED_SPACECRAFT = {"PSP", "STEREO_A"}
DEFAULT_PSP_RAW_DIR = Path("data/psp")
DEFAULT_STEREO_RAW_DIR = Path("data/stereo_a")
SUPPORTED_RAW_EXTENSIONS = {".cdf", ".csv", ".txt"}

_SPACECRAFT_NORMALIZATION = {
    "PSP": "PSP",
    "PARKER": "PSP",
    "PARKER_SOLAR_PROBE": "PSP",
    "STEREO_A": "STEREO_A",
    "STEREO-A": "STEREO_A",
    "STEREOA": "STEREO_A",
}


def _log(message: str, verbose: bool = False) -> None:
    if verbose:
        print(f"[preprocess_PSP_STEREO] {message}", flush=True)


def parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: {value}. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
        ) from exc


def normalize_spacecraft_ids(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        key = value.strip().upper().replace(" ", "_")
        spacecraft = _SPACECRAFT_NORMALIZATION.get(key, key)
        if spacecraft not in ALLOWED_SPACECRAFT:
            raise argparse.ArgumentTypeError(
                f"Unsupported spacecraft '{value}'. Supported values: {sorted(ALLOWED_SPACECRAFT)}"
            )
        if spacecraft not in normalized:
            normalized.append(spacecraft)
    return normalized


def _normalize_df_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        normalized[key] = col
    return normalized


def _find_column(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None) -> str | None:
    by_norm = _normalize_df_columns(df)
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "_", cand.strip().lower()).strip("_")
        if key in by_norm:
            return by_norm[key]

    if contains:
        for key, original in by_norm.items():
            if all(part in key for part in contains):
                return original
    return None


def _to_speed_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return np.sqrt(np.sum(np.square(arr[:, :3].astype(float)), axis=1))
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(float)
    return np.nanmean(arr.astype(float), axis=tuple(range(1, arr.ndim)))


def _to_1d_float(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(float)
    return np.nanmean(arr.astype(float), axis=tuple(range(1, arr.ndim)))


def _sanitize_speed(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    # Drop known CDF/table fill values and physically implausible speeds.
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e20] = np.nan
    arr[(arr <= 0.0) | (arr > 3000.0)] = np.nan
    return arr


def _sanitize_lon(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e20] = np.nan
    return arr % 360.0


def _sanitize_lat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e20] = np.nan
    arr[(arr < -90.0) | (arr > 90.0)] = np.nan
    return arr


def _sanitize_radius(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e20] = np.nan
    arr[(arr <= 0.0) | (arr > 10.0)] = np.nan
    return arr


def _convert_epoch(values: np.ndarray):
    try:
        import cdflib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading CDF files requires cdflib. Install with: pip install cdflib"
        ) from exc

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr)

    try:
        converted = cdflib.cdfepoch.to_datetime(arr)
    except Exception as exc:
        raise RuntimeError("Unable to convert CDF epoch variable to datetime") from exc
    return pd.to_datetime(converted)


def _load_cdf_dataframe(path: Path, spacecraft_id: str, verbose: bool = False) -> pd.DataFrame:
    try:
        import cdflib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading CDF files requires cdflib. Install with: pip install cdflib"
        ) from exc

    _log(f"Reading CDF raw file: {path.name}", verbose)
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    vars_available = list(getattr(info, "zVariables", [])) + list(getattr(info, "rVariables", []))
    var_map = {v.lower(): v for v in vars_available}

    def pick(candidates: list[str], contains: list[str] | None = None) -> str | None:
        for cand in candidates:
            if cand.lower() in var_map:
                return var_map[cand.lower()]
        if contains:
            for k, v in var_map.items():
                if all(part in k for part in contains):
                    return v
        return None

    time_var = pick(["Epoch", "EPOCH", "epoch", "time", "Time", "epoch_1hr"], contains=["epoch"])  # required
    speed_var = pick(
        [
            "vp_moment_RTN",
            "vp_fit_RTN",
            "Vp_RTN",
            "proton_bulk_speed_1hr",
            "proton_speed",
            "speed",
            "V",
        ],
        contains=["speed"],
    )
    lon_var = pick(["HC_lon", "hc_lon", "Carr_lon", "Carrington_lon", "HCI_lon"], contains=["lon"])
    lat_var = pick(["HC_lat", "hc_lat", "Carr_lat", "Carrington_lat", "HCI_lat"], contains=["lat"])
    rad_var = pick(
        ["HC_radius", "hc_radius", "radius", "R_AU", "r_au", "HCI_radius", "heliocentric_dist_1hr"],
        contains=["rad"],
    ) or pick([], contains=["dist"])
    # STEREO-A's "Derived_from_1D_Maxwellian" CDFs name the distance variable
    # heliocentric_dist_1hr -- "dist", not "rad" or "radius" -- so the contains=["rad"]
    # fallback alone never matched it, silently leaving HC_radius all-NaN for every
    # STEREO-A row read this way (only masked by 2018-2024 runs having a fallback
    # multi_spacecraft_obs.csv to merge_asof against; any window without one, e.g.
    # 2011-2017, dropped ~90% of otherwise-good rows for "missing coordinates").

    missing = [
        name
        for name, var in {
            "time": time_var,
            "speed": speed_var,
        }.items()
        if var is None
    ]
    if missing:
        raise RuntimeError(
            f"Missing required variables in {path.name}: {missing}. "
            f"Available CDF variables: {vars_available[:30]}"
        )

    time = _convert_epoch(cdf.varget(time_var))
    speed = _sanitize_speed(_to_speed_series(cdf.varget(speed_var)))
    if lon_var is not None:
        hc_lon = _sanitize_lon(_to_1d_float(cdf.varget(lon_var)))
    else:
        hc_lon = np.full(len(time), np.nan, dtype=float)

    if lat_var is not None:
        hc_lat = _sanitize_lat(_to_1d_float(cdf.varget(lat_var)))
    else:
        hc_lat = np.full(len(time), np.nan, dtype=float)

    if rad_var is not None:
        hc_radius_raw = _to_1d_float(cdf.varget(rad_var))
        # STEREO-A's heliocentric_dist_1hr (and similarly-named fallbacks) are in km,
        # not AU -- _sanitize_radius rejects anything > 10.0 as an AU-scale sanity
        # bound, so an unconverted km value (~1.5e8) is silently thrown away. Only
        # convert values that look like km (a genuine AU distance is always < 10);
        # already-AU-scale variables (e.g. HCI_radius, if ever picked) pass through.
        hc_radius_raw = np.where(np.abs(hc_radius_raw) > 10.0, hc_radius_raw / AU_IN_KM, hc_radius_raw)
        hc_radius = _sanitize_radius(hc_radius_raw)
    else:
        hc_radius = np.full(len(time), np.nan, dtype=float)

    n = min(len(time), len(speed), len(hc_lon), len(hc_lat), len(hc_radius))
    out = pd.DataFrame(
        {
            "Time": pd.to_datetime(time[:n], errors="coerce"),
            "Speed": pd.to_numeric(speed[:n], errors="coerce"),
            "HC_lon": pd.to_numeric(hc_lon[:n], errors="coerce"),
            "HC_lat": pd.to_numeric(hc_lat[:n], errors="coerce"),
            "HC_radius": pd.to_numeric(hc_radius[:n], errors="coerce"),
            "Spacecraft_ID": spacecraft_id,
        }
    )
    return out


def _load_table_dataframe(path: Path, spacecraft_id: str, verbose: bool = False) -> pd.DataFrame:
    _log(f"Reading table raw file: {path.name}", verbose)
    df = pd.read_csv(path)

    time_col = _find_column(df, ["Time", "Datetime", "timestamp"], contains=["time"])
    speed_col = _find_column(df, ["Speed", "V", "Vp", "proton_speed"], contains=["speed"])
    lon_col = _find_column(df, ["HC_lon", "Carrington_lon", "lon"], contains=["lon"])
    lat_col = _find_column(df, ["HC_lat", "Carrington_lat", "lat"], contains=["lat"])
    rad_col = _find_column(df, ["HC_radius", "radius", "r_au"], contains=["radius"])

    missing = [
        name
        for name, var in {
            "Time": time_col,
            "Speed": speed_col,
            "HC_lon": lon_col,
            "HC_lat": lat_col,
            "HC_radius": rad_col,
        }.items()
        if var is None
    ]
    if missing:
        raise RuntimeError(f"Missing required columns in {path.name}: {missing}")

    out = pd.DataFrame(
        {
            "Time": pd.to_datetime(df[time_col], errors="coerce"),
            "Speed": _sanitize_speed(pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=float)),
            "HC_lon": _sanitize_lon(pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)),
            "HC_lat": _sanitize_lat(pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)),
            "HC_radius": _sanitize_radius(pd.to_numeric(df[rad_col], errors="coerce").to_numpy(dtype=float)),
            "Spacecraft_ID": spacecraft_id,
        }
    )
    return out


def _extract_date_from_name(name: str) -> datetime | None:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if not match:
        return None
    y, m, d = map(int, match.groups())
    try:
        return datetime(y, m, d)
    except ValueError:
        return None


def _select_raw_files(raw_dir: Path, start: datetime, end: datetime, verbose: bool = False) -> list[Path]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        return []

    candidates = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_RAW_EXTENSIONS
    ]

    selected = []
    for path in candidates:
        file_dt = _extract_date_from_name(path.name)
        if file_dt is None:
            selected.append(path)
            continue
        if start.date() <= file_dt.date() <= end.date():
            selected.append(path)

    _log(f"Selected {len(selected)} raw files from {raw_dir}", verbose)
    return sorted(selected)


def _read_one_raw_file(path: Path, spacecraft_id: str, verbose: bool = False) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".cdf":
        return _load_cdf_dataframe(path, spacecraft_id=spacecraft_id, verbose=verbose)
    if suffix in {".csv", ".txt"}:
        return _load_table_dataframe(path, spacecraft_id=spacecraft_id, verbose=verbose)
    raise RuntimeError(f"Unsupported raw file extension: {path}")


def _load_fallback_multisat(input_path: Path, spacecraft_id: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Multi-spacecraft observation file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["Time"])
    if "Spacecraft_ID" not in df.columns and "spacecraft" not in df.columns:
        raise ValueError("Expected 'Spacecraft_ID' or 'spacecraft' column in the multi-spacecraft file")

    if "Spacecraft_ID" not in df.columns:
        df["Spacecraft_ID"] = df["spacecraft"].astype(str)

    df["Spacecraft_ID"] = (
        df["Spacecraft_ID"]
        .astype(str)
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    df["Spacecraft_ID"] = df["Spacecraft_ID"].replace(
        {
            "PARKER": "PSP",
            "PARKER_SOLAR_PROBE": "PSP",
            "STEREOA": "STEREO_A",
        }
    )

    df = df[df["Spacecraft_ID"] == spacecraft_id].copy()
    df["Speed"] = _sanitize_speed(pd.to_numeric(df["Speed"], errors="coerce").to_numpy(dtype=float))
    df["HC_lon"] = _sanitize_lon(pd.to_numeric(df["HC_lon"], errors="coerce").to_numpy(dtype=float))
    df["HC_lat"] = _sanitize_lat(pd.to_numeric(df["HC_lat"], errors="coerce").to_numpy(dtype=float))
    df["HC_radius"] = _sanitize_radius(pd.to_numeric(df["HC_radius"], errors="coerce").to_numpy(dtype=float))
    df = df.dropna(subset=["Time", "Speed", "HC_lon", "HC_lat", "HC_radius"])
    return df.sort_values("Time").reset_index(drop=True)


def _circular_mean_deg(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if valid.size == 0:
        return float("nan")
    radians = np.deg2rad(valid % 360.0)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    return float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0)


def _fill_missing_coords_from_fallback(
    mission_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    if mission_df.empty or fallback_df.empty:
        return mission_df

    left = mission_df.sort_values("Time").copy()
    right = fallback_df[["Time", "HC_lon", "HC_lat", "HC_radius"]].sort_values("Time").copy()
    merged = pd.merge_asof(
        left,
        right,
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
        suffixes=("", "_fb"),
    )

    for col in ["HC_lon", "HC_lat", "HC_radius"]:
        fb_col = f"{col}_fb"
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[fb_col] = pd.to_numeric(merged[fb_col], errors="coerce")
        merged[col] = merged[col].where(merged[col].notna(), merged[fb_col])
        merged.drop(columns=[fb_col], inplace=True)

    merged["HC_lon"] = merged["HC_lon"] % 360.0
    return merged


def _build_hourly_from_raw(
    spacecraft_id: str,
    raw_files: list[Path],
    start: datetime,
    end: datetime,
    verbose: bool = False,
) -> pd.DataFrame:
    if not raw_files:
        return pd.DataFrame(columns=["Time", "Speed", "HC_lon", "HC_lat", "HC_radius", "Spacecraft_ID"])

    frames = []
    for idx, path in enumerate(raw_files, start=1):
        _log(f"{spacecraft_id}: ingesting raw file {idx}/{len(raw_files)} -> {path.name}", verbose)
        try:
            frame = _read_one_raw_file(path, spacecraft_id=spacecraft_id, verbose=verbose)
        except Exception as exc:
            _log(f"{spacecraft_id}: skipping unreadable raw file {path.name}: {exc}", verbose)
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["Time", "Speed", "HC_lon", "HC_lat", "HC_radius", "Spacecraft_ID"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["Time", "Speed"]).copy()
    merged["Time"] = pd.to_datetime(merged["Time"])
    merged = merged[(merged["Time"] >= pd.Timestamp(start)) & (merged["Time"] <= pd.Timestamp(end))]

    if merged.empty:
        return pd.DataFrame(columns=["Time", "Speed", "HC_lon", "HC_lat", "HC_radius", "Spacecraft_ID"])

    merged = merged.sort_values("Time").set_index("Time")
    hourly_speed = pd.to_numeric(merged["Speed"], errors="coerce").resample("1h").mean()
    hourly_lon = merged["HC_lon"].resample("1h").apply(_circular_mean_deg)
    hourly_lat = pd.to_numeric(merged["HC_lat"], errors="coerce").resample("1h").mean()
    hourly_radius = pd.to_numeric(merged["HC_radius"], errors="coerce").resample("1h").mean()

    hourly = pd.concat(
        [hourly_speed.rename("Speed"), hourly_lon.rename("HC_lon"), hourly_lat.rename("HC_lat"), hourly_radius.rename("HC_radius")],
        axis=1,
    ).reset_index()
    hourly = hourly.dropna(subset=["Speed"])
    hourly["HC_lon"] = pd.to_numeric(hourly["HC_lon"], errors="coerce") % 360.0
    hourly["Spacecraft_ID"] = spacecraft_id
    return hourly


def preprocess_psp_stereo_window(
    output_path: Path,
    start: datetime,
    end: datetime,
    spacecraft_ids: list[str],
    psp_raw_dir: Path = DEFAULT_PSP_RAW_DIR,
    stereo_raw_dir: Path = DEFAULT_STEREO_RAW_DIR,
    fallback_input_path: Path | None = Path("data/multi_spacecraft_obs.csv"),
    verbose: bool = False,
) -> pd.DataFrame:
    selected_spacecraft = normalize_spacecraft_ids(spacecraft_ids)
    mission_frames = []

    for spacecraft_id in selected_spacecraft:
        raw_dir = psp_raw_dir if spacecraft_id == "PSP" else stereo_raw_dir
        raw_files = _select_raw_files(raw_dir, start=start, end=end, verbose=verbose)
        fallback_df = pd.DataFrame()
        if fallback_input_path is not None and fallback_input_path.exists():
            fallback_df = _load_fallback_multisat(fallback_input_path, spacecraft_id)

        mission_df = _build_hourly_from_raw(
            spacecraft_id=spacecraft_id,
            raw_files=raw_files,
            start=start,
            end=end,
            verbose=verbose,
        )

        if not mission_df.empty and not fallback_df.empty:
            mission_df = _fill_missing_coords_from_fallback(
                mission_df=mission_df,
                fallback_df=fallback_df,
                verbose=verbose,
            )

        before_drop = len(mission_df)
        if not mission_df.empty:
            mission_df = mission_df.dropna(subset=["Time", "Speed", "HC_lon", "HC_lat", "HC_radius"]).copy()
            mission_df = mission_df.sort_values("Time").reset_index(drop=True)
        dropped = before_drop - len(mission_df)
        if verbose and dropped > 0:
            _log(f"{spacecraft_id}: dropped {dropped} rows missing coordinates after fallback fill", verbose)

        if mission_df.empty and not fallback_df.empty:
            _log(
                f"{spacecraft_id}: no valid raw rows found, falling back to {fallback_input_path}",
                verbose,
            )
            mission_df = fallback_df[
                (fallback_df["Time"] >= pd.Timestamp(start)) & (fallback_df["Time"] <= pd.Timestamp(end))
            ].copy()

        if mission_df.empty:
            _log(f"{spacecraft_id}: no rows available in requested window", verbose)
        mission_frames.append(mission_df)

    filtered = pd.concat(mission_frames, ignore_index=True) if mission_frames else pd.DataFrame()
    if not filtered.empty:
        filtered = filtered.sort_values("Time").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    if verbose:
        print(f"Preprocessed PSP/STEREO observations -> {output_path}")
        print(f"  rows after hourly ingest: {len(filtered)}")
        if not filtered.empty:
            print(filtered["Spacecraft_ID"].value_counts().to_string())

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PSP and STEREO-A observations for a shared time window.",
    )
    parser.add_argument("--start", type=parse_iso_datetime, required=True)
    parser.add_argument("--end", type=parse_iso_datetime, required=True)
    parser.add_argument(
        "--spacecraft",
        nargs="+",
        default=["PSP", "STEREO_A"],
        help="Spacecraft IDs to include: PSP and/or STEREO_A",
    )
    parser.add_argument(
        "--psp-raw-dir",
        type=Path,
        default=DEFAULT_PSP_RAW_DIR,
        help="Directory containing raw PSP files (CDF/CSV/TXT).",
    )
    parser.add_argument(
        "--stereo-raw-dir",
        type=Path,
        default=DEFAULT_STEREO_RAW_DIR,
        help="Directory containing raw STEREO-A files (CDF/CSV/TXT).",
    )
    parser.add_argument(
        "--fallback-input",
        type=Path,
        default=Path("data/multi_spacecraft_obs.csv"),
        help="Optional fallback combined CSV if no valid raw rows are ingested.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output filtered CSV path.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be after or equal to --start")

    preprocess_psp_stereo_window(
        output_path=args.output,
        start=args.start,
        end=args.end,
        spacecraft_ids=args.spacecraft,
        psp_raw_dir=args.psp_raw_dir,
        stereo_raw_dir=args.stereo_raw_dir,
        fallback_input_path=args.fallback_input,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
