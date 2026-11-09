"""
batch_output.py

Utilities for writing all batch outputs.

This module performs file I/O only.
It should not contain optimization or forward-model logic.
"""

from pathlib import Path

import numpy as np

from boundary import save_boundary_weights


# ==========================================================
# Checkpoint
# ==========================================================

def save_batch_checkpoint(
    output_dir,
    batch_idx,
    batch_start_time,
    config,
    boundary,
    forward_model,
    loss_fn,
    batch_result,
    lon_deg_hourly,
    lat_deg_hourly,
    v_obs_eval_np,
    v_pred_eval_np,
):
    """
    Save the optimized state for one batch.

    Returns
    -------
    pathlib.Path
        Path of the checkpoint.
    """

    output_dir = Path(output_dir)

    batch_path = output_dir / f"batch_{batch_idx:03d}_boundary.npz"

    np.savez(
        batch_path,

        v_inner=batch_result["v_inner"],

        v0=loss_fn.v0.detach().cpu().numpy(),

        gradient_avg=batch_result["gradient_avg"],

        boundary_sensitivity_map=batch_result[
            "boundary_sensitivity_map"
        ],

        batch_start_time=str(batch_start_time),

        dt=config.dt,

        time_hours=np.arange(
            len(v_obs_eval_np),
            dtype=np.float32,
        ) * (config.dt / 3600.0),

        lon_deg_hourly=lon_deg_hourly,
        lat_deg_hourly=lat_deg_hourly,

        v_obs_eval=v_obs_eval_np,
        v_pred_eval=v_pred_eval_np,

        epoch=batch_result["best_epoch"],
        loss=batch_result["best_loss"],
        max_ae=batch_result["best_max_ae"],

        aia_global_mean=float(
            getattr(config, "aia_global_mean", np.nan)
        ),

        aia_global_std=float(
            getattr(config, "aia_global_std", np.nan)
        ),

        hmi_global_mean=float(
            getattr(config, "hmi_global_mean", np.nan)
        ),

        hmi_global_std=float(
            getattr(config, "hmi_global_std", np.nan)
        ),
    )

    save_boundary_weights(
        boundary,
        batch_path,
        forward_extra_state_dict=forward_model.learned_radial_accel.state_dict()
        if getattr(forward_model, 'learned_radial_accel', None) is not None
        else None,
    )

    return batch_path