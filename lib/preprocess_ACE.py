#!/usr/bin/env python3
"""
Standalone ACE preprocessing for SW_inverse.

This script can:
1) Download ACE SWEPAM hourly speed text files.
2) Download ACE daily GSE position file.
3) Transform daily GSE positions to Heliographic Carrington coordinates.
4) Interpolate Carrington coordinates to hourly cadence (wrap-safe longitude).
5) Merge hourly coordinates with hourly speed and save ACE_speed_lon_lat_hr.txt.

Output columns:
- Time
- HC_lon
- HC_lat
- HC_radius
- Speed

Notes:
- Longitudes are stored in degrees [0, 360).
- Speed missing flags (<= -9999) are preserved in output.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from data_paths import ACE_1H

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires astropy. Install with: pip install astropy"
    ) from exc

try:
    from sunpy.coordinates import frames
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires sunpy for HeliographicCarrington transforms. "
        "Install with: pip install sunpy"
    ) from exc


ACE_MONTHLY_BASE = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/"
ACE_GSE_POS_URL = "https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt"
DEFAULT_ACE_MONTHLY_DIR = ACE_1H
DEFAULT_OUTPUT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # repository root/data


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def month_iter(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _log(msg: str) -> None:
    print(f"[preprocess_ACE] {msg}", flush=True)


def download_ace_swepam_monthly(
    out_dir: Path,
    start: datetime,
    end: datetime,
    include_prev_month: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if include_prev_month:
        if start.month == 1:
            start_dl = datetime(start.year - 1, 12, 1)
        else:
            start_dl = datetime(start.year, start.month - 1, 1)
    else:
        start_dl = datetime(start.year, start.month, 1)

    _log(f"Downloading ACE monthly SWEPAM files into {out_dir}")
    downloaded: list[Path] = []
    for year, month in month_iter(start_dl, end):
        tag = f"{year}{month:02d}"
        pattern = f"{tag}*_ace_swepam_1h.txt"

        _log(f"Fetching month {tag} ({pattern})")

        # Use wget in a controlled way and store files directly in out_dir.
        cmd = [
            "wget",
            "-q",
            "-nc",
            "-P",
            str(out_dir),
            "-A",
            pattern,
            "-r",
            "-l1",
            "--no-parent",
            "--no-directories",
            ACE_MONTHLY_BASE,
        ]
        _run_cmd(cmd)

    downloaded = sorted(out_dir.glob("*_ace_swepam_1h.txt"))
    if not downloaded:
        raise RuntimeError(f"No ACE monthly SWEPAM files found in {out_dir}")
    _log(f"Found {len(downloaded)} ACE monthly SWEPAM files")
    return downloaded


def download_ace_gse_position(out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0:
        _log(f"Using existing ACE GSE position file: {out_file}")
        return out_file

    _log(f"Downloading ACE GSE position file to {out_file}")
    cmd = ["wget", "-q", "-O", str(out_file), ACE_GSE_POS_URL]
    _run_cmd(cmd)
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError("Failed to download ACE_GSE_position.txt")
    _log("Downloaded ACE GSE position file successfully")
    return out_file


def parse_ace_swepam_file(file_path: Path) -> pd.DataFrame:
    # Historical format has 16 header lines. Keep parser defensive.
    raw = pd.read_csv(
        file_path,
        sep=r"\s+|\t+|\s+\t+|\t+\s+",
        engine="python",
        header=16,
    )

    if len(raw) <= 1:
        return pd.DataFrame(columns=["Time", "Speed"])

    raw = raw.iloc[1:, :].copy()
    if raw.shape[1] < 9:
        return pd.DataFrame(columns=["Time", "Speed"])

    # Use the first 10 expected columns when available.
    raw = raw.iloc[:, : min(10, raw.shape[1])]
    cols = ["YR", "MO", "DA", "HHMM", "Day", "Day.1", "S", "Density", "Speed", "Temperature"]
    raw.columns = cols[: raw.shape[1]]

    for c in ["YR", "MO", "DA", "HHMM", "Speed"]:
        if c not in raw.columns:
            raise ValueError(f"Expected column '{c}' in {file_path}")

    yr = pd.to_numeric(raw["YR"], errors="coerce")
    mo = pd.to_numeric(raw["MO"], errors="coerce")
    da = pd.to_numeric(raw["DA"], errors="coerce")
    hhmm = pd.to_numeric(raw["HHMM"], errors="coerce")

    # HHMM may contain values like 100, 2300; derive hour safely.
    hour = (hhmm // 100).astype("Int64")

    time_df = pd.DataFrame(
        {
            "year": yr.astype("Int64"),
            "month": mo.astype("Int64"),
            "day": da.astype("Int64"),
            "hour": hour,
        }
    )
    time = pd.to_datetime(time_df, errors="coerce")

    speed = pd.to_numeric(raw["Speed"], errors="coerce")
    out = pd.DataFrame({"Time": time, "Speed": speed})
    out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return out


def load_hourly_speed_from_monthlies(monthly_files: list[Path]) -> pd.DataFrame:
    _log(f"Parsing {len(monthly_files)} monthly ACE speed files")
    frames = [parse_ace_swepam_file(p) for p in monthly_files]
    hourly = pd.concat(frames, ignore_index=True)
    if hourly.empty:
        raise RuntimeError("No valid hourly speed rows parsed from monthly ACE files")

    hourly = hourly.drop_duplicates(subset="Time", keep="last").sort_values("Time").reset_index(drop=True)
    _log(f"Built hourly ACE speed table with {len(hourly)} rows")
    return hourly


def transform_daily_gse_to_carrington(gse_file: Path) -> pd.DataFrame:
    _log(f"Reading ACE GSE positions from {gse_file}")
    df = pd.read_csv(gse_file, sep="\t")

    required = ["Year", "DOY", "GSE_X(km)", "GSE_y(km)", "GSE_z(km)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {gse_file}: {missing}")

    hc_lon = np.full(len(df), np.nan, dtype=float)
    hc_lat = np.full(len(df), np.nan, dtype=float)
    hc_rad = np.full(len(df), np.nan, dtype=float)
    day_time = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(len(df)):
        if i % 365 == 0 or i == len(df) - 1:
            _log(f"Transforming daily ACE coordinates: {i + 1}/{len(df)}")
        year = int(float(df.loc[i, "Year"]))
        doy = int(float(df.loc[i, "DOY"]))
        obs_dt = datetime.strptime(f"{year} {doy}", "%Y %j")
        day_time[i] = np.datetime64(obs_dt)

        c = SkyCoord(
            x=float(df.loc[i, "GSE_X(km)"]) * u.km,
            y=float(df.loc[i, "GSE_y(km)"]) * u.km,
            z=float(df.loc[i, "GSE_z(km)"]) * u.km,
            representation_type="cartesian",
            obstime=obs_dt,
            frame=frames.GeocentricSolarEcliptic,
        )
        h = c.transform_to(frames.HeliographicCarrington(observer="sun"))
        hc_lon[i] = float(h.lon.to_value(u.deg)) % 360.0
        hc_lat[i] = float(h.lat.to_value(u.deg))
        hc_rad[i] = float(h.radius.to_value(u.AU))

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(day_time),
            "HC_lon": hc_lon,
            "HC_lat": hc_lat,
            "HC_radius": hc_rad,
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date", keep="last")
    _log(f"Built daily Carrington coordinate table with {len(out)} rows")
    return out.reset_index(drop=True)


def interpolate_daily_coords_to_hourly(
    daily_coords: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.DataFrame:
    _log(f"Interpolating daily Carrington coordinates to hourly cadence: {t_start} -> {t_end}")
    hourly_index = pd.date_range(start=t_start, end=t_end, freq="h")

    src = daily_coords.set_index("date").reindex(hourly_index)

    # Wrap-safe longitude interpolation:
    # unwrap in radians -> interpolate in time -> wrap back to [0, 360).
    lon_deg = src["HC_lon"].to_numpy(dtype=float)
    lon_valid = np.isfinite(lon_deg)
    lon_hourly = np.full(len(src), np.nan, dtype=float)

    if np.sum(lon_valid) >= 2:
        lon_rad = np.deg2rad(lon_deg[lon_valid])
        lon_unwrapped = np.unwrap(lon_rad)

        t_all = src.index.view("int64").astype(np.float64)
        t_valid = src.index[lon_valid].view("int64").astype(np.float64)

        lon_interp_unwrapped = np.interp(
            t_all,
            t_valid,
            lon_unwrapped,
            left=lon_unwrapped[0],
            right=lon_unwrapped[-1],
        )
        lon_hourly = (np.rad2deg(lon_interp_unwrapped) % 360.0).astype(float)
    elif np.sum(lon_valid) == 1:
        lon_hourly[:] = float(lon_deg[lon_valid][0]) % 360.0
    else:
        raise RuntimeError("No valid HC_lon values available for interpolation")

    # Non-angular columns: standard time interpolation.
    lat_series = src["HC_lat"].interpolate(method="time", limit_direction="both")
    rad_series = src["HC_radius"].interpolate(method="time", limit_direction="both")

    out = pd.DataFrame(
        {
            "Time": hourly_index,
            "HC_lon": lon_hourly,
            "HC_lat": lat_series.to_numpy(dtype=float),
            "HC_radius": rad_series.to_numpy(dtype=float),
        }
    )
    _log(f"Created hourly Carrington coordinate table with {len(out)} rows")
    return out


def build_ace_hourly_table(
    ace_data_dir: Path,
    data_dir: Path,
    start: datetime,
    end: datetime,
    download: bool,
) -> Path:
    ace_data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _log("Starting ACE preprocessing pipeline")
    _log(f"Requested date range: {start} -> {end}")
    _log(f"ACE monthly source dir: {ace_data_dir}")
    _log(f"Output data dir: {data_dir}")

    if download:
        _log("Download mode enabled")
        monthly_files = download_ace_swepam_monthly(
            out_dir=ace_data_dir,
            start=start,
            end=end,
            include_prev_month=True,
        )
    else:
        _log("Download mode disabled; using local monthly ACE files")
        monthly_files = sorted(ace_data_dir.glob("*_ace_swepam_1h.txt"))
        if not monthly_files:
            raise RuntimeError(
                f"No monthly ACE files in {ace_data_dir}. "
                f"Either provide files or run with --download-ace."
            )
        _log(f"Found {len(monthly_files)} local monthly ACE files")

    speed_hourly = load_hourly_speed_from_monthlies(monthly_files)

    gse_file = data_dir / "ACE_GSE_position.txt"
    if download:
        download_ace_gse_position(gse_file)
    elif not gse_file.exists():
        _log(f"Missing {gse_file}; downloading it from ACE archive...")
        download_ace_gse_position(gse_file)

    daily_coords = transform_daily_gse_to_carrington(gse_file)

    # Restrict to user date range and overlap with available speed.
    t0 = max(pd.Timestamp(start), pd.Timestamp(speed_hourly["Time"].iloc[0]))
    t1 = min(pd.Timestamp(end), pd.Timestamp(speed_hourly["Time"].iloc[-1]))
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between requested window and ACE speed data")

    _log(f"Overlapping time range: {t0} -> {t1}")

    coords_hourly = interpolate_daily_coords_to_hourly(daily_coords, t0, t1)

    speed_trim = speed_hourly[(speed_hourly["Time"] >= t0) & (speed_hourly["Time"] <= t1)].copy()
    speed_trim = speed_trim.drop_duplicates(subset="Time", keep="last").sort_values("Time")

    merged = coords_hourly.set_index("Time").join(speed_trim.set_index("Time"), how="left").reset_index()

    # Keep canonical column order.
    merged = merged[["Time", "HC_lon", "HC_lat", "HC_radius", "Speed"]]

    out_file = data_dir / "ACE_speed_lon_lat_hr.txt"
    _log("Merging hourly coordinates with hourly speed")
    merged.to_csv(out_file, index=False)

    _log(f"Saved: {out_file}")
    _log(f"Rows: {len(merged)}")
    _log(f"Time range: {merged['Time'].iloc[0]} to {merged['Time'].iloc[-1]}")
    _log(
        f"HC_lon range [deg]: {float(np.nanmin(merged['HC_lon'])):.3f} to {float(np.nanmax(merged['HC_lon'])):.3f}"
    )

    # Quick wrap sanity: report large hourly jumps.
    dlon = np.diff(merged["HC_lon"].to_numpy(dtype=float))
    bad = np.where(np.abs(dlon) > 20.0)[0]
    _log(f"Large HC_lon jumps >20 deg/hour: {int(len(bad))}")

    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ACE preprocessing for SW_inverse",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help=(
            "Output data directory "
            f"(default: {DEFAULT_OUTPUT_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--ace-monthly-dir",
        type=Path,
        default=DEFAULT_ACE_MONTHLY_DIR,
        help=(
            "Directory for monthly ACE SWEPAM text files "
            f"(default: {DEFAULT_ACE_MONTHLY_DIR})"
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2009-12-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-ace",
        action="store_true",
        help="Download ACE monthly speed files and ACE GSE position file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if end < start:
        raise ValueError("--end must be >= --start")

    build_ace_hourly_table(
        ace_data_dir=args.ace_monthly_dir,
        data_dir=args.data_dir,
        start=start,
        end=end,
        download=bool(args.download_ace),
    )


if __name__ == "__main__":
    main()
