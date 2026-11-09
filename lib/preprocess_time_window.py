#!/usr/bin/env python3
"""
Wrapper for windowed preprocessing of AIA, HMI, ACE, Parker Solar Probe, and STEREO-A data.

This script builds a common user-defined time window and only processes the
subset of data needed for that window instead of scanning all local archives.

Outputs:
  - AIA equatorial slice archive for the requested window
  - HMI magnetogram slice archive for the requested window
  - ACE hourly speed + Carrington position file for the requested window
  - Filtered multi-spacecraft observation CSV containing PSP and STEREO-A rows
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

import warnings
from erfa import ErfaWarning

warnings.filterwarnings(
    "ignore",
    category=ErfaWarning,
)

from preprocess_solar_image import preprocess_image_window
from preprocess_ACE import (
    DEFAULT_ACE_MONTHLY_DIR,
    DEFAULT_OUTPUT_DATA_DIR,
    build_ace_hourly_table,
)
from data_paths import SOLAR
from preprocess_PSP_STEREO import (
    DEFAULT_PSP_RAW_DIR,
    DEFAULT_STEREO_RAW_DIR,
    preprocess_psp_stereo_window,
)

ALLOWED_SPACECRAFT = {"ACE", "STEREO_A", "PSP"}
DEFAULT_SPACECRAFT = ["ACE"]
DEFAULT_CME_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "HELIO4CAST_ICMECAT_v23.csv"   # repository root/data

_CME_SPACECRAFT_MAP = {
    "WIND": "ACE",
    "PSP": "PSP",
    "STEREO_A": "STEREO_A",
    "STEREO-A": "STEREO_A",
    "SOLARORBITER": "SOLO",          # HELIO4CAST sc_insitu 'SolarOrbiter' (added 2026-10-22, Part II other spacecraft)
}


def resolve_existing_ace_output(ace_output_dir: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path
    else:
        path = ace_output_dir / "ACE_speed_lon_lat_hr.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Requested reuse of ACE preprocessing output, but file was not found: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Requested reuse of ACE preprocessing output, but path is not a file: {path}"
        )
    return path


def parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: {value}. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
        ) from exc


def clean_label(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S") if dt.time() != datetime.min.time() else dt.strftime("%Y%m%d")


def _normalize_spacecraft_list(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        spacecraft = value.strip().upper().replace("-", "_")
        if spacecraft == "STEREO":
            spacecraft = "STEREO_A"
        if spacecraft not in ALLOWED_SPACECRAFT:
            raise argparse.ArgumentTypeError(
                f"Unsupported spacecraft '{value}'. Supported values: {sorted(ALLOWED_SPACECRAFT)}"
            )
        if spacecraft not in normalized:
            normalized.append(spacecraft)
    return normalized


def _normalize_time_series(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values, errors="coerce", utc=True)
    return times.dt.tz_convert(None)


def load_cme_catalog(
    input_path: Path,
    spacecraft_ids: list[str],
    start: datetime,
    end: datetime,
    verbose: bool = False,
) -> pd.DataFrame:
    if input_path is None or not input_path.exists():
        if verbose:
            print(f"CME catalogue not found, skipping CME masking: {input_path}")
        return pd.DataFrame(columns=["Spacecraft_ID", "icme_start_time", "mo_end_time"])

    df = pd.read_csv(input_path)
    required_columns = {"sc_insitu", "icme_start_time", "mo_end_time"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CME catalogue missing required columns: {sorted(missing_columns)}"
        )

    work = df.loc[:, ["sc_insitu", "icme_start_time", "mo_end_time"]].copy()
    work["Spacecraft_ID"] = (
        work["sc_insitu"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .map(_CME_SPACECRAFT_MAP)
    )
    work["icme_start_time"] = _normalize_time_series(work["icme_start_time"])
    work["mo_end_time"] = _normalize_time_series(work["mo_end_time"])
    work = work.dropna(subset=["Spacecraft_ID", "icme_start_time", "mo_end_time"])

    selected_spacecraft = set(spacecraft_ids)
    window_start = pd.Timestamp(start)
    window_end = pd.Timestamp(end)
    work = work[
        work["Spacecraft_ID"].isin(selected_spacecraft)
        & (work["mo_end_time"] >= window_start)
        & (work["icme_start_time"] <= window_end)
    ].copy()

    if work.empty:
        if verbose:
            print("CME catalogue rows in window: 0")
        return pd.DataFrame(columns=["Spacecraft_ID", "icme_start_time", "mo_end_time"])

    work["icme_start_time"] = work["icme_start_time"].clip(lower=window_start, upper=window_end)
    work["mo_end_time"] = work["mo_end_time"].clip(lower=window_start, upper=window_end)
    work = work[work["mo_end_time"] >= work["icme_start_time"]]
    work = work.sort_values(["Spacecraft_ID", "icme_start_time", "mo_end_time"]).reset_index(drop=True)

    if verbose:
        counts = work["Spacecraft_ID"].value_counts().to_dict()
        print(f"CME catalogue rows in window: {len(work)} {counts}")

    return work[["Spacecraft_ID", "icme_start_time", "mo_end_time"]]


def apply_cme_speed_mask(
    df: pd.DataFrame,
    cme_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    if df.empty or cme_df.empty:
        return df

    out = df.copy()
    out["Time"] = _normalize_time_series(out["Time"])

    masked_count = 0
    for row in cme_df.itertuples(index=False):
        mask = (
            (out["Spacecraft_ID"] == row.Spacecraft_ID)
            & (out["Time"] >= row.icme_start_time)
            & (out["Time"] <= row.mo_end_time)
        )
        if not mask.any():
            continue
        masked_count += int(mask.sum())
        out.loc[mask, "Speed"] = np.nan

    if verbose:
        print(f"Applied CME speed mask to {masked_count} rows")

    return out


def keep_common_time_bins(
    df: pd.DataFrame,
    required_spacecraft: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    if df.empty or not required_spacecraft:
        return df

    work = df.copy()
    work["Time"] = pd.to_datetime(work["Time"], errors="coerce")
    work["Speed"] = pd.to_numeric(work["Speed"], errors="coerce")
    work = work.dropna(subset=["Time", "Spacecraft_ID"])
    work = work[np.isfinite(work["Speed"])]
    work["time_bin"] = work["Time"].dt.floor("h")

    required = set(required_spacecraft)
    counts = (
        work.groupby("time_bin")["Spacecraft_ID"]
        .apply(lambda s: len(set(s) & required))
        .rename("n_required")
    )
    good_bins = counts[counts >= len(required)].index
    filtered = work[work["time_bin"].isin(good_bins)].drop(columns=["time_bin"]).copy()

    if verbose:
        print(
            f"Common-time filter kept {len(good_bins)} hourly bins with all spacecraft "
            f"{sorted(required)}; rows {len(df)} -> {len(filtered)}"
        )

    return filtered.sort_values("Time").reset_index(drop=True)


def load_solo_near_ecliptic(
    input_path: Path,
    start: datetime,
    end: datetime,
    max_abs_lat_deg: float,
    verbose: bool = False,
) -> pd.DataFrame:
    if not input_path.exists():
        if verbose:
            print(f"SOLO source file not found, skipping SOLO merge: {input_path}")
        return pd.DataFrame(columns=["Time", "HC_lon", "HC_lat", "HC_radius", "Speed", "Spacecraft_ID"])

    df = pd.read_csv(input_path, parse_dates=["Time"])
    if "Spacecraft_ID" not in df.columns and "spacecraft" not in df.columns:
        return pd.DataFrame(columns=["Time", "HC_lon", "HC_lat", "HC_radius", "Speed", "Spacecraft_ID"])

    if "Spacecraft_ID" not in df.columns:
        df["Spacecraft_ID"] = df["spacecraft"].astype(str)

    sc = (
        df["Spacecraft_ID"]
        .astype(str)
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    solo_mask = sc.isin({"SOLO", "SOLAR_ORBITER", "SOLARORBITER"})
    out = df[solo_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=["Time", "HC_lon", "HC_lat", "HC_radius", "Speed", "Spacecraft_ID"])

    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out["Speed"] = pd.to_numeric(out["Speed"], errors="coerce")
    out["HC_lon"] = pd.to_numeric(out["HC_lon"], errors="coerce") % 360.0
    out["HC_lat"] = pd.to_numeric(out["HC_lat"], errors="coerce")
    out["HC_radius"] = pd.to_numeric(out["HC_radius"], errors="coerce")
    out["Spacecraft_ID"] = "SOLO"

    out = out[
        (out["Time"] >= pd.Timestamp(start))
        & (out["Time"] <= pd.Timestamp(end))
        & (out["Speed"] > 0.0)
        & (out["Speed"] <= 3000.0)
        & (out["HC_radius"] > 0.0)
        & (out["HC_radius"] <= 10.0)
        & (out["HC_lat"].abs() <= float(max_abs_lat_deg))
    ].copy()

    out = out.sort_values("Time").reset_index(drop=True)
    if verbose:
        print(
            f"SOLO near-ecliptic filter kept {len(out)} rows "
            f"(|HC_lat| <= {max_abs_lat_deg:g} deg)."
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess AIA, HMI,ACE, Parker, and STEREO-A data for a shared time window.",
    )
    parser.add_argument(
        "--start",
        type=parse_iso_datetime,
        required=True,
        help="Start of the window (ISO YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=parse_iso_datetime,
        required=True,
        help="End of the window (ISO YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--aia-input-dir",
        type=Path,
        default=SOLAR / "SDO/AIA/4096_jp2/193",
        help="Local AIA image folder to process.",
    )
    parser.add_argument(
        "--hmi-input-dir",
        type=Path,
        default=SOLAR / "SDO/HMI/fits",
        help="Local HMI image folder to process.",
    )
    parser.add_argument(
        "--aia-output",
        type=Path,
        default=None,
        help="Output AIA .npz file path.",
    )

    parser.add_argument(
        "--hmi-output",
        type=Path,
        default=None,
        help="Output HMI .npz file path.",
    )

    parser.add_argument(
        "--n-longitude",
        type=int,
        default=360,
        help="Number of Carrington longitude samples for AIA preprocessing.",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=0.0,
        help="Latitude (deg) used when --aia-slice-plane=carrington_equator.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=float("nan"),
        help="Fill value used outside the solar disk in AIA/HMI preprocessing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print AIA/HMI progress every N scanned/processed files.",
    )
    parser.add_argument(
        "--min-separation-minutes",
        type=float,
        default=30.0,
        help="Minimum time separation between kept AIA/HMI frames (minutes).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for AIA/HMI processing.",
    )
    parser.add_argument(
        "--spacecraft",
        nargs="+",
        default=DEFAULT_SPACECRAFT,
        help="Spacecraft to include on top of AIA/HMI: ACE, STEREO_A, PSP.",
    )
    parser.add_argument(
        "--ace-monthly-dir",
        type=Path,
        default=DEFAULT_ACE_MONTHLY_DIR,
        help="Directory containing local ACE monthly SWEPAM text files.",
    )
    parser.add_argument(
        "--ace-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory where ACE hourlies will be written.",
    )
    parser.add_argument(
        "--skip-ace-preprocessing",
        action="store_true",
        help="Reuse an existing ACE hourly output file instead of rebuilding it.",
    )
    parser.add_argument(
        "--ace-existing-output",
        type=Path,
        default=None,
        help=(
            "Explicit ACE hourly output file to reuse with --skip-ace-preprocessing. "
            "Defaults to --ace-output-dir/ACE_speed_lon_lat_hr.txt."
        ),
    )
    parser.add_argument(
        "--multisat-input",
        type=Path,
        default=Path("data/multi_spacecraft_obs.csv"),
        help="Fallback combined CSV used only if raw PSP/STEREO ingest yields no rows.",
    )
    parser.add_argument(
        "--include-solo-near-ecliptic",
        dest="include_solo_near_ecliptic",
        action="store_true",
        default=True,
        help="Include SOLO rows from --multisat-input when |HC_lat| is below threshold.",
    )
    parser.add_argument(
        "--no-include-solo-near-ecliptic",
        dest="include_solo_near_ecliptic",
        action="store_false",
        help="Disable SOLO near-ecliptic merge.",
    )
    parser.add_argument(
        "--solo-max-abs-lat",
        type=float,
        default=7.5,
        help="Maximum absolute HC_lat (deg) allowed when including SOLO.",
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
        "--multisat-output",
        type=Path,
        default=None,
        help="Output filtered multi-spacecraft observations CSV path.",
    )
    parser.add_argument(
        "--cme-catalog",
        type=Path,
        default=DEFAULT_CME_CATALOG_PATH,
        help="Path to HELIO4CAST ICME catalogue used to mask spacecraft speeds during CME intervals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed_window"),
        help="Directory for wrapper-produced output files.",
    )
    parser.add_argument(
        "--download-ace",
        action="store_true",
        help="Download ACE data if needed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--require-common-time-bins",
        action="store_true",
        help=(
            "Require hourly bins where all selected spacecraft are present. "
            "Default is disabled: keep each spacecraft whenever data is available."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be after or equal to --start")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_label = f"{clean_label(args.start)}_{clean_label(args.end)}"

    aia_output = args.aia_output or args.output_dir / f"aia_equatorial_slices_{window_label}.npz"
    hmi_output = args.hmi_output or args.output_dir / f"hmi_equatorial_slices_{window_label}.npz"
    multisat_output = args.multisat_output or args.output_dir / f"multi_spacecraft_obs_{window_label}.csv"

    if args.verbose:
        print(f"Window: {args.start} -> {args.end}")
        print(f"AIA input: {args.aia_input_dir}")
        print(f"HMI input: {args.hmi_input_dir}")
        print(f"ACE monthly input: {args.ace_monthly_dir}")
        if args.skip_ace_preprocessing:
            print("ACE preprocessing: skipped (reusing existing hourly output)")
            if args.ace_existing_output is not None:
                print(f"ACE existing output: {args.ace_existing_output}")
        print(f"PSP raw input dir: {args.psp_raw_dir}")
        print(f"STEREO-A raw input dir: {args.stereo_raw_dir}")
        print(f"Fallback combined input: {args.multisat_input}")
        print(
            f"SOLO near-ecliptic: enabled={args.include_solo_near_ecliptic}, "
            f"max |HC_lat|={args.solo_max_abs_lat:g} deg"
        )
        print(f"Output dir: {args.output_dir}")

    spacecraft = _normalize_spacecraft_list(args.spacecraft)
    if args.verbose:
        print(f"Selected spacecraft: {spacecraft}")

    cme_df = load_cme_catalog(
        input_path=args.cme_catalog,
        spacecraft_ids=spacecraft,
        start=args.start,
        end=args.end,
        verbose=args.verbose,
    )


    print("\nPreprocessing HMI...")

    hmi_path = preprocess_image_window(
        input_dir=args.hmi_input_dir,
        output_path=hmi_output,
            n_longitude=args.n_longitude,
        fill_value=args.fill_value,
        start=args.start,
        end=args.end,
        verbose=args.verbose,
        progress_every=args.progress_every,
        workers=max(1, args.workers),
        min_separation_minutes=max(0.0, args.min_separation_minutes),
        instrument="hmi",
    
    )

    hmi_npz = np.load(hmi_path)
    hmi_times = np.asarray(hmi_npz["times"])

    print("\nPreprocessing AIA...")

    aia_path = preprocess_image_window(
        input_dir=args.aia_input_dir,
        output_path=aia_output,
        n_longitude=args.n_longitude,
        fill_value=args.fill_value,
        start=args.start,
        end=args.end,
        verbose=args.verbose,
        progress_every=args.progress_every,
        workers=max(1, args.workers),
        min_separation_minutes=max(0.0, args.min_separation_minutes),
        instrument="aia",
        target_times=hmi_times,
    )


    aia_npz = np.load(aia_path)
    aia_times = np.asarray(aia_npz["times"])

    common_times = np.intersect1d(
        aia_times,
        hmi_times,
    )

    print(
        f"Keeping {len(common_times)} common timestamps "
        f"out of AIA={len(aia_times)}, HMI={len(hmi_times)}"
    )

    aia_keep = np.isin(aia_times, common_times)
    hmi_keep = np.isin(hmi_times, common_times)

    if aia_keep.sum() != len(common_times):
        raise RuntimeError("AIA/common time mismatch")

    if hmi_keep.sum() != len(common_times):
        raise RuntimeError("HMI/common time mismatch")

    np.savez(
        aia_path,
        times=aia_npz["times"][aia_keep],
        carrington_longitude=aia_npz["carrington_longitude"],
        visibility_mask=aia_npz["visibility_mask"][:, aia_keep],
        aia_slices=aia_npz["aia_slices"][:, aia_keep],
    )

    np.savez(
        hmi_path,
        times=hmi_npz["times"][hmi_keep],
        carrington_longitude=hmi_npz["carrington_longitude"],
        visibility_mask=hmi_npz["visibility_mask"][:, hmi_keep],
        hmi_slices=hmi_npz["hmi_slices"][:, hmi_keep],
    )

    aia_times_final = np.load(aia_path)["times"]
    hmi_times_final = np.load(hmi_path)["times"]

    if not np.array_equal(
        aia_times_final,
        hmi_times_final,
    ):
        raise RuntimeError(
            "AIA/HMI timestamps still differ after filtering"
        )

    print(
        f"AIA/HMI aligned successfully "
        f"with {len(aia_times_final)} common frames"
    )

    ace_path = None
    if "ACE" in spacecraft:
        if args.skip_ace_preprocessing:
            ace_path = resolve_existing_ace_output(args.ace_output_dir, args.ace_existing_output)
            print(f"\nReusing ACE hourly output: {ace_path}")
        else:
            print("\nPreprocessing ACE... ")
            ace_path = build_ace_hourly_table(
                ace_data_dir=args.ace_monthly_dir,
                data_dir=args.ace_output_dir,
                start=args.start,
                end=args.end,
                download=args.download_ace,
            )

    non_ace_spacecraft = [sc for sc in spacecraft if sc != "ACE"]
    solo_df = load_solo_near_ecliptic(
        input_path=args.multisat_input,
        start=args.start,
        end=args.end,
        max_abs_lat_deg=args.solo_max_abs_lat,
        verbose=args.verbose,
    ) if args.include_solo_near_ecliptic else pd.DataFrame()

    print("\nFiltering spacecraft observations... ")
    if "ACE" in spacecraft and non_ace_spacecraft:
        ace_df = pd.read_csv(ace_path, parse_dates=["Time"])
        ace_df["Spacecraft_ID"] = "ACE"
        ace_df = ace_df[(ace_df["Time"] >= pd.Timestamp(args.start)) & (ace_df["Time"] <= pd.Timestamp(args.end))]
        ace_df["Speed"] = pd.to_numeric(ace_df["Speed"], errors="coerce")
        ace_df = ace_df[(ace_df["Speed"] > 0.0) & (ace_df["Speed"] <= 3000.0)]

        other_df = preprocess_psp_stereo_window(
            output_path=args.output_dir / f"temp_multisat_{window_label}.csv",
            start=args.start,
            end=args.end,
            spacecraft_ids=non_ace_spacecraft,
            psp_raw_dir=args.psp_raw_dir,
            stereo_raw_dir=args.stereo_raw_dir,
            fallback_input_path=args.multisat_input,
            verbose=args.verbose,
        )
        combined_df = pd.concat([ace_df, other_df], ignore_index=True)
        combined_df = apply_cme_speed_mask(combined_df, cme_df, verbose=args.verbose)
        if args.require_common_time_bins:
            combined_df = keep_common_time_bins(combined_df, spacecraft, verbose=args.verbose)
        if not solo_df.empty:
            combined_df = pd.concat([combined_df, solo_df], ignore_index=True).sort_values("Time")
            if args.verbose:
                print(f"Merged SOLO rows in-window: +{len(solo_df)} rows")
        combined_df.to_csv(multisat_output, index=False)
        multisat_path = multisat_output
    elif "ACE" in spacecraft:
        # Only ACE selected; use ACE output as the observation file.
        ace_df = pd.read_csv(ace_path, parse_dates=["Time"])
        ace_df["Spacecraft_ID"] = "ACE"
        ace_df = ace_df[(ace_df["Time"] >= pd.Timestamp(args.start)) & (ace_df["Time"] <= pd.Timestamp(args.end))]
        ace_df["Speed"] = pd.to_numeric(ace_df["Speed"], errors="coerce")
        ace_df = ace_df[(ace_df["Speed"] > 0.0) & (ace_df["Speed"] <= 3000.0)]
        ace_df = apply_cme_speed_mask(ace_df, cme_df, verbose=args.verbose)
        if not solo_df.empty:
            ace_df = pd.concat([ace_df, solo_df], ignore_index=True).sort_values("Time")
            if args.verbose:
                print(f"Merged SOLO rows in-window: +{len(solo_df)} rows")
        ace_df.to_csv(multisat_output, index=False)
        multisat_path = multisat_output
    else:
        multisat_df = preprocess_psp_stereo_window(
            output_path=multisat_output,
            start=args.start,
            end=args.end,
            spacecraft_ids=non_ace_spacecraft,
            psp_raw_dir=args.psp_raw_dir,
            stereo_raw_dir=args.stereo_raw_dir,
            fallback_input_path=args.multisat_input,
            verbose=args.verbose,
        )
        multisat_df = apply_cme_speed_mask(multisat_df, cme_df, verbose=args.verbose)
        if args.require_common_time_bins:
            multisat_df = keep_common_time_bins(multisat_df, non_ace_spacecraft, verbose=args.verbose)
        if not solo_df.empty:
            multisat_df = pd.concat([multisat_df, solo_df], ignore_index=True).sort_values("Time")
            if args.verbose:
                print(f"Merged SOLO rows in-window: +{len(solo_df)} rows")
        multisat_df.to_csv(multisat_output, index=False)
        multisat_path = multisat_output

    print("\nSummary:")
    print(f"  AIA slices: {aia_path}")
    print(f"  HMI slices: {hmi_path}")
    if ace_path is not None:
        print(f"  ACE hourly output: {ace_path}")
    print(f"  Observation file: {multisat_path}")

    if args.verbose:
        df = pd.read_csv(multisat_path, parse_dates=["Time"])
        for spacecraft in sorted({"Spacecraft_ID", "spacecraft"} & set(df.columns)):
            counts = df[spacecraft].value_counts().to_dict()
            print(f"  {spacecraft} counts: {counts}")

    print("\nWindowed preprocessing complete.")


if __name__ == "__main__":
    main()
