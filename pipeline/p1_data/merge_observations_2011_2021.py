"""Build multi_spacecraft_obs_20110101_20211231T235959.csv for the new 2011-2021 window,
reusing the already-built ACE table (data/ace_2011_2021/ACE_speed_lon_lat_hr.txt) and the
already-cached STEREO-A raw CDFs (data/stereo_a/), instead of re-running preprocess_time_window.py
end to end (which would also needlessly redo the already-completed AIA/HMI extraction).

Replicates exactly the "ACE + non-ACE spacecraft present" branch of preprocess_time_window.py's
main() -- same functions, same CME-masking call, same column handling -- just skipping the
AIA/HMI section since those files already exist.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

from preprocess_time_window import (
    load_cme_catalog,
    apply_cme_speed_mask,
    load_solo_near_ecliptic,
    DEFAULT_CME_CATALOG_PATH,
)
from preprocess_PSP_STEREO import (
    DEFAULT_STEREO_RAW_DIR,
    preprocess_psp_stereo_window,
)

START = datetime(2011, 1, 1)
END = datetime(2021, 12, 31)
ACE_PATH = Path("data/ace_2011_2021/ACE_speed_lon_lat_hr.txt")
OUTPUT = Path("data/preprocessed_window/multi_spacecraft_obs_20110101_20211231T235959.csv")
SPACECRAFT = ["ACE", "STEREO_A"]

cme_df = load_cme_catalog(
    input_path=DEFAULT_CME_CATALOG_PATH, spacecraft_ids=SPACECRAFT, start=START, end=END, verbose=True,
)

ace_df = pd.read_csv(ACE_PATH, parse_dates=["Time"])
ace_df["Spacecraft_ID"] = "ACE"
ace_df = ace_df[(ace_df["Time"] >= pd.Timestamp(START)) & (ace_df["Time"] <= pd.Timestamp(END))]
ace_df["Speed"] = pd.to_numeric(ace_df["Speed"], errors="coerce")
ace_df = ace_df[(ace_df["Speed"] > 0.0) & (ace_df["Speed"] <= 3000.0)]
print(f"ACE rows in window: {len(ace_df)}")

other_df = preprocess_psp_stereo_window(
    output_path=Path("data/preprocessed_window/temp_multisat_20110101_20211231T235959.csv"),
    start=START, end=END, spacecraft_ids=["STEREO_A"],
    stereo_raw_dir=DEFAULT_STEREO_RAW_DIR,
    fallback_input_path=Path("data/multi_spacecraft_obs.csv"),
    verbose=True,
)
print(f"STEREO_A rows in window: {len(other_df)}")

combined_df = pd.concat([ace_df, other_df], ignore_index=True)
combined_df = apply_cme_speed_mask(combined_df, cme_df, verbose=True)

solo_df = load_solo_near_ecliptic(
    input_path=Path("data/multi_spacecraft_obs.csv"), start=START, end=END, max_abs_lat_deg=7.5, verbose=True,
)
if not solo_df.empty:
    combined_df = pd.concat([combined_df, solo_df], ignore_index=True).sort_values("Time")
    print(f"Merged SOLO rows in-window: +{len(solo_df)} rows")

combined_df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(combined_df)} rows to {OUTPUT}")
print(combined_df["Spacecraft_ID"].value_counts())
