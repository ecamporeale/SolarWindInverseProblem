import os

# Prevent MKL/OpenMP threading layer clash in spawned worker processes
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import warnings
warnings.filterwarnings(
    "ignore",
    message="resource_tracker:.*leaked semaphore",
    category=UserWarning,
)

from pathlib import Path

import pandas as pd
import torch

from config import Config
from inverse_solver import batch_inverse_solver


VALID_FORWARD_SCHEMES = {
    "fv",
    "rk4_fast",
    "rk4_time_upwind",
    "rk2_tvd_split",
}


# =============================================================================
# Configuration helpers
# =============================================================================

def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str):
    value = os.getenv(name)
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_environment_overrides(config: Config):
    """Apply optional runtime overrides from environment variables."""

    config.forward_scheme = os.getenv(
        "FORWARD_SCHEME",
        config.forward_scheme,
    )

    if os.getenv("INPUT_HOURS_LIMIT") is not None:
        config.input_hours_limit = int(os.getenv("INPUT_HOURS_LIMIT"))

    config.overlap_hours = float(
        os.getenv("OVERLAP_HOURS", config.overlap_hours)
    )

    config.parallel_batch_devices = (
        env_list("PARALLEL_BATCH_DEVICES")
        or config.parallel_batch_devices
    )

    config.compile_rk2_rhs = env_bool(
        "COMPILE_RK2_RHS",
        config.compile_rk2_rhs,
    )

    config.loss_weighting = os.getenv(
        "LOSS_WEIGHTING",
        config.loss_weighting,
    )

    if os.getenv("TIME_BATCH_SIZE") is not None:
        config.time_batch_size = int(os.getenv("TIME_BATCH_SIZE"))

    if os.getenv("MAX_MINIBATCHES_PER_EPOCH") is not None:
        config.max_minibatches_per_epoch = int(
            os.getenv("MAX_MINIBATCHES_PER_EPOCH")
        )

    config.temporal_shuffle_minibatches = env_bool(
        "TEMPORAL_SHUFFLE_MINIBATCHES",
        config.temporal_shuffle_minibatches,
    )

    if os.getenv("TEMPORAL_SHUFFLE_SEED") is not None:
        config.temporal_shuffle_seed = int(
            os.getenv("TEMPORAL_SHUFFLE_SEED")
        )

    config.full_window_validation = env_bool(
        "FULL_WINDOW_VALIDATION",
        config.full_window_validation,
    )

    config.resume_from_saved_boundary = env_bool(
        "RESUME_FROM_SAVED_BOUNDARY",
        config.resume_from_saved_boundary,
    )


def validate_config(config: Config):
    """Validate runtime configuration."""

    if config.forward_scheme not in VALID_FORWARD_SCHEMES:
        raise ValueError(
            f"Invalid forward scheme '{config.forward_scheme}'. "
            f"Valid options: {sorted(VALID_FORWARD_SCHEMES)}"
        )

    if int(getattr(config, "time_batch_size", 0)) < 0:
        raise ValueError(
            f"TIME_BATCH_SIZE must be >= 0, got {config.time_batch_size}"
        )

    if int(getattr(config, "max_minibatches_per_epoch", 0)) < 0:
        raise ValueError(
            "MAX_MINIBATCHES_PER_EPOCH must be >= 0, got "
            f"{config.max_minibatches_per_epoch}"
        )

    if int(getattr(config, "temporal_shuffle_seed", 0)) < 0:
        raise ValueError(
            "TEMPORAL_SHUFFLE_SEED must be >= 0, got "
            f"{config.temporal_shuffle_seed}"
        )


def print_config_summary(config: Config):
    """Print runtime configuration."""

    print(f"Device: {config.device}")
    print(f"Grid: Nr={config.Nr}, Nphi={config.Nphi}, Nt={config.Nt}")
    print(
        f"Forward scheme: {config.forward_scheme} "
        f"(options: {', '.join(sorted(VALID_FORWARD_SCHEMES))})"
    )
    if config.input_hours_limit is None:
        print("Input span: full AIA file")
    else:
        print(f"Input span limit: {config.input_hours_limit} hours")

    print(f"Overlap: {config.overlap_hours} hours")
    print(f"Resume from saved boundary: {config.resume_from_saved_boundary}")
    if int(getattr(config, "time_batch_size", 0)) > 0:
        max_mb = int(getattr(config, "max_minibatches_per_epoch", 0))
        cap_text = "none" if max_mb <= 0 else str(max_mb)
        print(
            "Temporal mini-batching: "
            f"enabled (time_batch_size={config.time_batch_size}, "
            f"max_minibatches_per_epoch={cap_text})"
        )
        print(
            "Temporal mini-batch shuffle: "
            f"{'enabled' if config.temporal_shuffle_minibatches else 'disabled'} "
            f"(seed={config.temporal_shuffle_seed})"
        )
        print(
            "Full-window validation: "
            f"{'enabled' if config.full_window_validation else 'disabled'}"
        )
    else:
        print("Temporal mini-batching: disabled (full-window loss per step)")


# =============================================================================
# Observation loading
# =============================================================================

def resolve_data_path(config: Config) -> Path:
    """Resolve observation data path."""

    repo_dir = Path(__file__).resolve().parents[1]   # repository root (this module lives in lib/)

    env_path = os.getenv("OBS_DATA_PATH") or os.getenv("ACE_DATA_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    configured_path = getattr(config, "observation_data_path", None)
    if configured_path is None:
        raise ValueError(
            "Config is missing observation_data_path. "
            "Set Config.observation_data_path or OBS_DATA_PATH."
        )

    configured_path = Path(str(configured_path)).expanduser()

    if configured_path.is_absolute():
        return configured_path.resolve()

    return (repo_dir / configured_path).resolve()


def validate_dataframe(df: pd.DataFrame, config: Config):
    """Validate required dataframe columns."""

    required = {"Time", "Speed", "HC_lon"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use_multi = bool(
        getattr(config, "enable_multispacecraft_loss", False)
    )

    if (
        use_multi
        and "Spacecraft_ID" in df.columns
        and "HC_radius" not in df.columns
    ):
        raise ValueError(
            "HC_radius column is required when "
            "enable_multispacecraft_loss=True."
        )


def sanitize_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean observation dataframe."""

    df = df.copy()

    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")

    if "HC_radius" in df.columns:
        df["HC_radius"] = pd.to_numeric(
            df["HC_radius"],
            errors="coerce",
        )

    df = df.dropna(subset=["Time", "Speed", "HC_lon"])

    df = df.sort_values("Time").reset_index(drop=True)

    invalid = (df["Speed"] <= 0.0) | (df["Speed"] > 2000.0)

    if invalid.any():
        print(
            f"Warning: dropping {invalid.sum()} rows "
            f"with invalid solar wind speed."
        )
        df = df.loc[~invalid].reset_index(drop=True)

    return df


def load_observations(config: Config) -> pd.DataFrame:
    """Load, validate and optionally trim observations."""

    data_path = resolve_data_path(config)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Observation file not found:\n{data_path}"
        )

    print(f"\nLoading observations from\n{data_path}")

    df = pd.read_csv(
        data_path,
        parse_dates=["Time"],
    )

    validate_dataframe(df, config)

    df = sanitize_observations(df)

    if len(df) == 0:
        raise ValueError("Observation dataframe is empty.")

    if config.input_hours_limit is not None:

        start_time = df["Time"].min()
        end_time = start_time + pd.Timedelta(
            hours=config.input_hours_limit
        )

        df = (
            df[df["Time"] <= end_time]
            .reset_index(drop=True)
        )

    print(f"Loaded {len(df)} observations")
    print(
        f"Date range: {df['Time'].min()} "
        f"to {df['Time'].max()}"
    )
    print(
        f"Speed range: "
        f"{df['Speed'].min():.1f} - "
        f"{df['Speed'].max():.1f} km/s"
    )

    return df


# =============================================================================
# Output
# =============================================================================

def prepare_output_directory(config: Config) -> Path:
    """Create output directory if necessary."""

    output_dir = Path(
        getattr(
            config,
            "saved_boundary_dir",
            "inverse_results",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def print_summary(results, output_dir: Path):
    """Print completion summary."""

    print("\n" + "=" * 60)
    print("Batch processing complete!")
    print(f"Processed {len(results['batches'])} batches")
    print(f"Results saved to {output_dir}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():

    config = Config()

    apply_environment_overrides(config)

    validate_config(config)

    print_config_summary(config)

    observations = load_observations(config)

    output_dir = prepare_output_directory(config)

    results = batch_inverse_solver(
        config=config,
        v_obs_df=observations,
        output_dir=str(output_dir),
    )

    if not isinstance(results, dict) or "batches" not in results:
        raise ValueError(
            "Unexpected output from batch_inverse_solver()."
        )

    print_summary(results, output_dir)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    main()