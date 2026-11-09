import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from forward_model import PropagationForwardModel, steady_state_propagate
from boundary import load_saved_boundary_guess, save_boundary_weights, IntensityToVelocityMLP
from loss import LossFunction, build_training_stages
from save_io import _save_batch_summary_and_loss_plot
import pandas as pd
import numpy as np
from batch_output import save_batch_checkpoint
from helpers import (
    setup_grid,
    save_batch_aia_plot,
    save_batch_hmi_plot,
    sample_satellite_track,
)
from optimization import run_training_stage



def _load_resume_state(
    batch_idx,
    local_config,
    device,
    output_dir,
):
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_best_metrics = None
    saved_guess, saved_v0, saved_path, saved_state_dict, saved_optimizer_state, saved_scheduler_state, saved_forward_extra_state, saved_checkpoint_meta = load_saved_boundary_guess(
        local_config,
        device,
        fallback_dir=output_dir,
        batch_idx=batch_idx,
    )
    resume_optimizer_state = saved_optimizer_state
    resume_scheduler_state = saved_scheduler_state
    if saved_checkpoint_meta:
        resume_best_metrics = {
            'best_loss': float(saved_checkpoint_meta.get('loss', float('inf'))),
            'best_max_ae': float(saved_checkpoint_meta.get('max_ae', float('inf'))),
            'best_rmse': float(saved_checkpoint_meta.get('rmse', float('inf'))),
            'best_epoch': int(saved_checkpoint_meta['epoch']) if saved_checkpoint_meta.get('epoch') is not None else None,
        }
    return (
    saved_guess,
    saved_v0,
    saved_path,
    saved_state_dict,
    resume_optimizer_state,
    resume_scheduler_state,
    saved_forward_extra_state,
    resume_best_metrics,
)

def _get_primary_track(obs_tracks):
    """
    Return the primary observation track used for optimization.

    By convention this is always the first track (ACE when available).
    """

    if not obs_tracks:
        raise ValueError("obs_tracks is empty")

    return obs_tracks[0]

def _prepare_batch_images(
    batch_spec,
    batch_idx,
    batch_start_time,
    local_config,
    output_dir,
    device,
):
    batch_aia_tensor = None
    if batch_spec.get('image_slice') is not None:
        batch_aia_tensor = torch.tensor(batch_spec['image_slice'], dtype=torch.float32, device=device)
        batch_aia_path = output_dir / f"batch_{batch_idx:03d}_AIA.png"
        save_batch_aia_plot(
            batch_spec['image_slice'],
            local_config,
            batch_aia_path,
            batch_start_time=batch_start_time,
            title=f"Batch {batch_idx}: AIA input used by NN",
        )
        print(f"Saved batch {batch_idx} AIA input plot to {batch_aia_path}")
        if batch_spec['image_slice'].ndim == 3 and batch_spec['image_slice'].shape[0] >= 3:
            batch_hmi_path = output_dir / f"batch_{batch_idx:03d}_HMI.png"
            save_batch_hmi_plot(
                batch_spec['image_slice'],
                local_config,
                batch_hmi_path,
                batch_start_time=batch_start_time,
                title=f"Batch {batch_idx}: HMI magnetogram input used by NN",
            )
            print(f"Saved batch {batch_idx} HMI input plot to {batch_hmi_path}")
    if  batch_aia_tensor is None:
        raise RuntimeError(
            f"Batch {batch_idx}: image slice missing for batch; expected 'image_slice' key in batch_spec with shape [C, Nphi, Nt]."
        )

    return batch_aia_tensor

def _validate_batch_observations(
        batch_spec,
        batch_idx,
        device_str,
        v_obs_batch,
    ):    
        valid_mask_np = ~np.isnan(v_obs_batch.detach().cpu().numpy())
        num_valid = int(np.sum(valid_mask_np))
        num_total = len(v_obs_batch)
        valid_fraction = num_valid / num_total if num_total > 0 else 0.0

        batch_kind = "FINAL BATCH" if batch_spec['is_final'] else "BATCH"
        print(f"\n{'='*60}")
        print(
            f"{batch_kind} {batch_idx} [{device_str}]: observation time steps "
            f"{batch_spec['t_start']} to {batch_spec['t_end_obs']}"
        )
        print(f"Valid samples: {num_valid}/{num_total} ({100*valid_fraction:.1f}%)")
        if num_valid > 0:
            v_valid = v_obs_batch[valid_mask_np]
            print(f"Speed range (valid): {v_valid.min():.1f} - {v_valid.max():.1f} km/s")
        print(f"{'='*60}")

        is_valid = num_valid >= num_total * 0.1

        if not is_valid:
            print(
                f"WARNING: Skipping batch {batch_idx} - "
                f"insufficient valid data ({valid_fraction:.1%})"
            )

        return (
            is_valid,
            valid_mask_np,
            num_valid,
            num_total,
        )


def _load_compatible_state_dict(module, state_dict, batch_idx, saved_path):
    """Load only state-dict entries that match the current module shapes."""
    if state_dict is None:
        return False

    current_state = module.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if current_value is None:
            skipped_keys.append(f"{key} (unexpected)")
            continue
        if not isinstance(value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
            skipped_keys.append(f"{key} (non-tensor)")
            continue
        if value.shape != current_value.shape:
            skipped_keys.append(
                f"{key} (saved {tuple(value.shape)} != current {tuple(current_value.shape)})"
            )
            continue
        compatible_state[key] = value

    if not compatible_state:
        print(
            f"Batch {batch_idx}: no compatible weights found in saved checkpoint {saved_path}; "
            "starting from a fresh initialization."
        )
        return False

    missing, unexpected = module.load_state_dict(compatible_state, strict=False)

    if skipped_keys or missing or unexpected:
        print(
            f"Batch {batch_idx}: resumed checkpoint {saved_path} with partial compatibility."
        )
        if skipped_keys:
            print(f"  skipped keys: {', '.join(skipped_keys[:8])}")
            if len(skipped_keys) > 8:
                print(f"  ... and {len(skipped_keys) - 8} more skipped keys")
        if missing:
            print(f"  missing keys after load: {missing}")
        if unexpected:
            print(f"  unexpected keys after load: {unexpected}")

    return True


def _initialize_boundary(
    batch_idx,
    batch_spec,
    saved_state_dict,
    saved_v0,
    saved_path,
    saved_guess,
    batch_aia_tensor,
    local_config,
    device,
    r,
):

    if (saved_guess is not None) or (saved_state_dict is not None):
        boundary = IntensityToVelocityMLP(local_config).to(device)
        if saved_state_dict is not None:
            _load_compatible_state_dict(
                boundary,
                saved_state_dict,
                batch_idx,
                saved_path,
            )
        if saved_v0 is not None:
            v0 = saved_v0.to(device)
            print(f"Batch {batch_idx}: resume boundary weights and v0 from saved file {saved_path}")
        else:
            v0_seed = boundary(batch_aia_tensor).detach()
            v0 = steady_state_propagate(v0_seed[:, 0], r, direction='outward')
            print(f"Batch {batch_idx}: resume boundary weights from saved file {saved_path}")
            print(f"Batch {batch_idx}: v0 source=saved_boundary_t0")
    else:
        history_speed = np.asarray(batch_spec['history_speed_hourly'], dtype=float)
        history_speed = np.where(history_speed < -9999.0, np.nan, history_speed)
        
#        v_boundary_guess = torch.full((local_config.Nphi, local_config.Nt), 400.0, device=device, dtype=torch.float32)
        v0 = torch.full((r.shape[0], local_config.Nphi), 400.0, device=device, dtype=torch.float32)
#        boundary = BoundaryCondition(local_config, initial_guess=v_boundary_guess).to(device)
        boundary = IntensityToVelocityMLP(local_config).to(device)
        print(
            f"Batch {batch_idx}: no history init available; "
            "using constant 400 km/s inner boundary and v0 initialization"
        )
    return boundary, v0

def _build_loss_function(
    local_config,
    forward_model,
    v_obs_batch,
    v0,
    phi_obs_batch_t,
    radius_obs_batch_t,
    time_obs_batch_t,
    obs_tracks,
    batch_start_time, batch_idx,
    device,
):
    if obs_tracks:
        for idx, track in enumerate(obs_tracks):
            n_phi = int(torch.isfinite(track['phi_obs_t']).sum().item())
            n_r = int(torch.isfinite(track['radius_obs_t']).sum().item())
            n_v = int(torch.isfinite(track['v_obs']).sum().item())
            n_tot = int(track['v_obs'].numel())
            print(
                f"Batch {batch_idx}: track {idx} ({track['id']}) finite counts "
                f"phi={n_phi}/{n_tot}, r={n_r}/{n_tot}, v_obs={n_v}/{n_tot}"
            )
    loss_fn = LossFunction(
        local_config,
        forward_model,
        v_obs_batch,
        v0,
        phi_obs_t=phi_obs_batch_t,
        radius_obs_t=radius_obs_batch_t,
        time_obs_t=time_obs_batch_t,
        obs_tracks=obs_tracks,
    ).to(device)
    loss_fn._batch_start_time = str(batch_start_time)

    return loss_fn

def execute_batch_pipeline(batch_spec, local_config, device, output_dir, phi_obs):
    """Core logic for ONE batch. Used by both sequential and parallel loops."""
    import os
    import sys
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    
    device_str = str(device)
    local_config.device = device_str
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(output_dir)
    batch_idx = batch_spec["batch_idx"]
    batch_start_time = pd.to_datetime(batch_spec['batch_start_time'])

    log_path = output_dir / f"batch_{batch_idx:03d}.log"
    log_file = open(log_path, "w", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = log_file

    try:
        r, phi, t, dr, dphi = setup_grid(local_config, print_cfl=False)
        forward_model = PropagationForwardModel(local_config, r, phi, dr, dphi).to(device)

        # 1. Grab our unified obs_tracks list
        obs_tracks = batch_spec.get("obs_tracks")
        if not obs_tracks:
            return None # Skip if no data

        # Push tensors to correct device
        for track in obs_tracks:
            track['v_obs'] = track['v_obs'].to(device)
            track['phi_obs_t'] = track['phi_obs_t'].to(device)
            track['radius_obs_t'] = track['radius_obs_t'].to(device)

        # 2. Extract primary track (ACE/PRIMARY) for primary metrics
        main_track = _get_primary_track(obs_tracks)
        v_obs_batch = main_track["v_obs"]
        phi_obs_batch_t = main_track["phi_obs_t"]
        radius_obs_batch_t = main_track["radius_obs_t"]
        
        time_obs_batch_t = torch.tensor(
            main_track['time_obs_seconds'], dtype=torch.float32, device=device
        )
        lon_deg_hourly = main_track["lon_deg_hourly"]
        lat_deg_hourly = main_track["lat_deg_hourly"]

        # 3. Load Images & Diagnostics
        batch_aia_tensor = _prepare_batch_images(batch_spec, batch_idx, batch_start_time, local_config, output_dir, device)
        batch_ok, valid_mask_np, num_valid, num_total = _validate_batch_observations(batch_spec, batch_idx, device_str, v_obs_batch)
        if not batch_ok: return None

        # 4. Initialize Optimization & Boundaries
        saved_guess, saved_v0, saved_path, saved_state_dict, resume_optimizer_state, resume_scheduler_state, resume_forward_extra_state, resume_best_metrics = _load_resume_state(batch_idx, local_config, device, output_dir)
        boundary, v0 = _initialize_boundary(batch_idx, batch_spec, saved_state_dict, saved_v0, saved_path, saved_guess, batch_aia_tensor, local_config, device, r)
        
        loss_fn = _build_loss_function(local_config, forward_model, v_obs_batch, v0, phi_obs_batch_t, radius_obs_batch_t, time_obs_batch_t, obs_tracks, batch_start_time, batch_idx, device)

        # 5. Execute Training Loop
        skip_warmup_prior = bool(saved_guess is not None or saved_state_dict is not None or saved_v0 is not None)
        loss_history = []
        batch_grad_sum, batch_epoch_count = None, 0
        base_scheme = local_config.forward_scheme

        for stage in build_training_stages(local_config, skip_warmup_prior=skip_warmup_prior):
            local_config.forward_scheme = stage['scheme']
            stage_prefix = f"  [batch-{batch_idx}:{stage['name']}:{stage['scheme']}] "
            
            stage_losses, stage_grads, stage_grad_sum, stage_epoch_count, best_epoch, best_loss, best_max_ae, stage_sensitivity = run_training_stage(
                boundary, loss_fn, r, local_config, stage, stage_prefix=stage_prefix,
                check_graph=(stage['name'] == 'refine'),
                checkpoint_path=str(output_dir / f"batch_{batch_idx:03d}_boundary.npz"),
                batch_lon_deg_hourly=lon_deg_hourly, batch_lat_deg_hourly=lat_deg_hourly, batch_image_slice=batch_aia_tensor,
                resume_optimizer_state_dict=resume_optimizer_state, resume_scheduler_state_dict=resume_scheduler_state, resume_forward_extra_state_dict=resume_forward_extra_state, resume_best_metrics=resume_best_metrics
            )
            loss_history.extend(stage_losses)
            if stage_epoch_count > 0 and stage_grad_sum is not None:
                if batch_grad_sum is None:
                    batch_grad_sum = stage_grad_sum.detach().clone()
                    batch_epoch_count = stage_epoch_count
                else:
                    batch_grad_sum.add_(stage_grad_sum)
                    batch_epoch_count += stage_epoch_count
                    
        local_config.forward_scheme = base_scheme

        # 6. Final Evaluation
        v_inner_final = boundary(batch_aia_tensor).detach()
        v_inner_to_save = v_inner_final.cpu().numpy().copy()

        with torch.no_grad():
            v_full_eval = forward_model(v_inner_final, loss_fn.v0)
            v_pred_eval = sample_satellite_track(v_full_eval, r, phi, phi_obs_batch_t, radius_obs_batch_t, time_obs=time_obs_batch_t, dt=local_config.dt)
            v_pred_eval_np = v_pred_eval.detach().cpu().numpy()
            v_obs_eval_np = v_obs_batch.detach().cpu().numpy()

        sensitivity_np = stage_sensitivity if stage_sensitivity is not None else np.full_like(v_inner_to_save, np.nan, dtype=np.float32)

        batch_result = {
            'batch_idx': batch_idx, 't_start': batch_spec['t_start'], 't_end': batch_spec['t_end_obs'],
            'v_inner': v_inner_to_save,
            'gradient_avg': (batch_grad_sum / float(batch_epoch_count)).cpu().numpy() if batch_grad_sum is not None else None,
            'boundary_sensitivity_map': sensitivity_np, 'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else float('inf'),
            'best_epoch': int(best_epoch) if best_epoch is not None else -1,
            'best_loss': float(best_loss) if best_epoch is not None else float('nan'),
            'best_max_ae': float(best_max_ae) if best_epoch is not None else float('nan'),
        }


        batch_path = save_batch_checkpoint(
                output_dir,
                batch_idx,
                batch_start_time,
                local_config,
                boundary,
            forward_model,
                loss_fn,
                batch_result,
                lon_deg_hourly,
                lat_deg_hourly,
                v_obs_eval_np,
                v_pred_eval_np,
            )
        return batch_result
    finally:
        sys.stdout.flush()
        sys.stdout = old_stdout
        log_file.close()


def _run_parallel_batches(config, batch_specs, all_solutions, output_dir, phi_obs):
    """Run independent batches in parallel worker processes."""
    if not batch_specs:
        _save_batch_summary_and_loss_plot(all_solutions, output_dir)
        return all_solutions

    configured_devices = getattr(config, 'parallel_batch_devices', None)
    if configured_devices is None:
        devices = _detect_free_gpus()
        if not devices:
            devices = ['cuda:0'] if torch.cuda.is_available() else ['cpu']
        print(f"Auto-detected {len(devices)} free GPU(s): {devices}")
    elif isinstance(configured_devices, str):
        devices = [configured_devices]
    else:
        devices = list(configured_devices)

    # Ensure every CUDA device string has an explicit index (e.g. 'cuda' -> 'cuda:0')
    devices = [
        f"{d}:0" if d == 'cuda' else d
        for d in devices
    ]

    any_cuda = any(str(device).startswith('cuda') for device in devices)
    allow_shared_device = bool(getattr(config, 'parallel_allow_shared_device', False))
    max_workers = len(devices)
    if not any_cuda:
        max_workers = max(1, int(getattr(config, 'parallel_batch_workers', 1)))
    elif not allow_shared_device:
        max_workers = len(devices)
    max_workers = min(max_workers, len(batch_specs))

    assigned_specs = []
    for idx, spec in enumerate(batch_specs):
        spec_copy = dict(spec)
        spec_copy['device'] = devices[idx % len(devices)]
        assigned_specs.append(spec_copy)

    print(
        f"\nParallel batch execution: {len(batch_specs)} batches, {max_workers} workers, "
        f"devices={devices}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        # Keep only max_workers active futures at a time.
        # This avoids launching/logging batches far ahead of currently running workers.
        active_futures = {}
        submit_idx = 0

        while submit_idx < min(max_workers, len(assigned_specs)):
            spec = assigned_specs[submit_idx]
            print(f"Starting batch {spec['batch_idx']} on {spec['device']}", flush=True)
            fut = executor.submit(execute_batch_pipeline, spec, deepcopy(config), spec["device"], output_dir, phi_obs)
            active_futures[fut] = spec
            submit_idx += 1

        completed = 0
        while active_futures:
            future = next(as_completed(active_futures))
            spec = active_futures.pop(future)

            try:
                result = future.result()
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            completed += 1
            print(
                f"Finished batch {spec['batch_idx']} "
                f"({completed}/{len(batch_specs)} completed)",
                flush=True,
            )
            if result is not None:
                all_solutions['batches'].append(result)

            if submit_idx < len(assigned_specs):
                next_spec = assigned_specs[submit_idx]
                fut = executor.submit(execute_batch_pipeline, next_spec, deepcopy(config), next_spec["device"], output_dir, phi_obs)
                active_futures[fut] = next_spec
                submit_idx += 1

    all_solutions['batches'].sort(key=lambda batch: batch['batch_idx'])
    _save_batch_summary_and_loss_plot(all_solutions, output_dir)
    return all_solutions

def _detect_free_gpus(mem_threshold_mb=1000):
    """Return list of CUDA device strings for GPUs with < mem_threshold_mb used memory."""
    if not torch.cuda.is_available():
        return []
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            mem_info = torch.cuda.mem_get_info(i)  # (free, total) in bytes
            used_mb = (mem_info[1] - mem_info[0]) / (1024 ** 2)
            if used_mb < mem_threshold_mb:
                free.append(f"cuda:{i}")
        except Exception:
            # mem_get_info may fail if CUDA context can't be created on that device;
            # fall back to assuming it's available.
            free.append(f"cuda:{i}")
    if not free:
        # All GPUs appear busy, but still return them all so we don't deadlock
        free = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return free

