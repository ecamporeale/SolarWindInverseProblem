import torch
import torch.optim as optim
from forward_model import steady_state_propagate
import numpy as np
import pandas as pd
from helpers import (
    sample_satellite_track,
    save_batch_solution_plot,
    )

from sensitivity import compute_boundary_sensitivity_map
from boundary import save_boundary_weights


def _to_float_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_learned_radial_accel_status(loss_fn):
    """Return a compact status string for learned radial acceleration logging."""
    fm = getattr(loss_fn, 'forward_model', None)
    if fm is None:
        return None

    accel = getattr(fm, 'learned_radial_accel', None)
    if accel is None:
        if bool(getattr(fm.config, 'use_learned_radial_acceleration', False)):
            return "RadialAccel: enabled-but-missing"
        return "RadialAccel: off"

    try:
        params = accel.physical_parameters()
    except Exception as exc:
        return f"RadialAccel: error({exc})"
    learnable = bool(getattr(accel, 'learnable', True))

    if hasattr(accel, 'ddelta_v_dr_rsun'):
        if len(params) != 2:
            return f"RadialAccel: learnable={learnable} (unavailable)"
        p0_f = _to_float_scalar(params[0])
        p1_f = _to_float_scalar(params[1])
        if None in (p0_f, p1_f):
            return f"RadialAccel: learnable={learnable} (unavailable)"
        return (
            f"RadialAccel(delta_v): L={p0_f:.3f} Rs  "
            f"Vref={p1_f:.2f} km/s  learnable={learnable}"
        )

    if len(params) != 3:
        return f"RadialAccel: learnable={learnable} (unavailable)"

    p0_f = _to_float_scalar(params[0])
    p1_f = _to_float_scalar(params[1])
    p2_f = _to_float_scalar(params[2])
    if None in (p0_f, p1_f, p2_f):
        return f"RadialAccel: learnable={learnable} (unavailable)"

    return (
        f"RadialAccel(relax): alpha={p0_f:.3e}  L={p1_f:.3f} Rs  "
        f"v_inf={p2_f:.2f} km/s  learnable={learnable}"
    )


def _format_learned_radial_accel_grad_status(loss_fn):
    """Return a compact gradient-status string for learned radial acceleration."""
    fm = getattr(loss_fn, 'forward_model', None)
    accel = getattr(fm, 'learned_radial_accel', None) if fm is not None else None
    if accel is None or not bool(getattr(accel, 'learnable', False)):
        return None

    grad_parts = []
    if hasattr(accel, 'ddelta_v_dr_rsun'):
        grad_names = ('raw_L', 'raw_Vref')
    else:
        grad_names = ('raw_alpha', 'raw_L', 'raw_v_inf')

    for name in grad_names:
        p = getattr(accel, name, None)
        if p is None:
            continue
        if p.grad is None:
            grad_parts.append(f"{name}=None")
        else:
            grad_parts.append(f"{name}={float(p.grad.detach().norm().item()):.3e}")

    if not grad_parts:
        return None
    return "RadialAccelGrad: " + "  ".join(grad_parts)


def _build_optimizer_param_groups(boundary, loss_fn, boundary_lr, accel_lr_scale=0.05):
    """Collect trainable parameters for optimization, including forward-model extras."""
    param_groups = []

    boundary_params = [p for p in boundary.parameters() if p.requires_grad]
    if boundary_params:
        param_groups.append({'params': boundary_params, 'lr': boundary_lr})

    fm = getattr(loss_fn, 'forward_model', None)
    accel = getattr(fm, 'learned_radial_accel', None) if fm is not None else None
    if accel is not None:
        accel_params = [p for p in accel.parameters() if p.requires_grad]
        if accel_params:
            param_groups.append({
                'params': accel_params,
                'lr': boundary_lr * float(accel_lr_scale),
            })

    return param_groups

def _build_plot_track_series(v_full_eval, r_grid, phi_grid, loss_fn, config, batch_start_time_str):
        extra_track_series = []
        track_lon_series = []
        raw_lon_series = getattr(loss_fn, 'track_lon_plot_series', [])
        raw_obs_series = getattr(loss_fn, 'track_obs_plot_series', [])
        raw_obs_by_id = {
            str(series.get('id', '')).upper(): series
            for series in raw_obs_series
        }
        extra_track_ids = set()
        bst = pd.to_datetime(batch_start_time_str) if batch_start_time_str else None

        # Always carry ACE longitude track for panel-1 overlay.
        # Use ACE by ID when available; otherwise fall back to primary buffers.
        ace_idx = None
        for idx, track_id in enumerate(getattr(loss_fn, 'track_ids', [])):
            if str(track_id).strip().upper() == 'ACE':
                ace_idx = idx
                break

        if ace_idx is not None:
            phi_obs_main = getattr(loss_fn, f'phi_obs_track_{ace_idx}').detach().cpu().numpy()
            v_obs_main = getattr(loss_fn, f'v_obs_track_{ace_idx}').detach().cpu().numpy()
            time_obs_main = getattr(loss_fn, f'time_obs_track_{ace_idx}').detach().cpu().numpy().astype(float)
        else:
            phi_obs_main = loss_fn.phi_obs_t.detach().cpu().numpy()
            v_obs_main = loss_fn.v_obs.detach().cpu().numpy()
            time_obs_main = loss_fn.time_obs_t.detach().cpu().numpy().astype(float)

        if bst is not None:
            t_axis_ace = bst + pd.to_timedelta(time_obs_main, unit='s')
        else:
            t_axis_ace = time_obs_main / 3600.0
        ace_track_entry = {
            'id': 'ACE',
            'time': t_axis_ace,
            'lon_deg': np.rad2deg(phi_obs_main),
            'speed': v_obs_main,
        }

        if bool(getattr(loss_fn, 'use_multispacecraft', False)) and int(getattr(loss_fn, 'num_tracks', 0)) > 0:
            for idx in range(loss_fn.num_tracks):
                track_id = str(loss_fn.track_ids[idx]) if idx < len(loss_fn.track_ids) else f"TRACK_{idx}"
                v_obs_track = getattr(loss_fn, f'v_obs_track_{idx}')
                phi_obs_track = getattr(loss_fn, f'phi_obs_track_{idx}')
                radius_obs_track = getattr(loss_fn, f'radius_obs_track_{idx}')
                time_obs_track = getattr(loss_fn, f'time_obs_track_{idx}')
                time_obs_track_np = time_obs_track.detach().cpu().numpy().astype(float)
                v_pred_track = sample_satellite_track(
                    v_full_eval,
                    r_grid,
                    phi_grid,
                    phi_obs_track,
                    radius_obs_track,
                    time_obs=time_obs_track,
                    dt=config.dt,
                )
                sc_raw = None

                if bst is not None:
                    t_axis_track = bst + pd.to_timedelta(time_obs_track_np, unit='s')
                else:
                    t_axis_track = time_obs_track_np / 3600.0

                if track_id.upper() != 'ACE':
                    v_pred_track_np = v_pred_track.detach().cpu().numpy()
                    v_obs_track_np = v_obs_track.detach().cpu().numpy()
                    sc_raw = raw_obs_by_id.get(track_id.upper())

                    if sc_raw is not None:
                        t_raw = pd.to_datetime(np.asarray(sc_raw.get('time', [])))
                        v_raw = np.asarray(sc_raw.get('speed', []), dtype=float)

                        if bst is not None and t_raw.size > 0:
                            t0_model = pd.to_datetime(bst)
                            model_s = time_obs_track_np
                            raw_s = (t_raw - t0_model).total_seconds().astype(float)
                            v_pred_raw = np.full(t_raw.shape[0], np.nan, dtype=float)
                            # Usually raw timestamps and model sample timestamps match exactly.
                            if raw_s.shape[0] == model_s.shape[0] and np.allclose(raw_s, model_s, atol=1e-6, equal_nan=False):
                                v_pred_raw[:] = v_pred_track_np
                            else:
                                in_range = (raw_s >= model_s[0]) & (raw_s <= model_s[-1])
                                if np.any(in_range):
                                    v_pred_raw[in_range] = np.interp(raw_s[in_range], model_s, v_pred_track_np)
                            extra_track_series.append(
                                {
                                    'id': track_id,
                                    'time': t_raw,
                                    'v_obs': v_raw,
                                    'v_pred': v_pred_raw,
                                }
                            )
                            extra_track_ids.add(track_id.upper())
                        else:
                            extra_track_series.append(
                                {
                                    'id': track_id,
                                    'time': t_axis_track,
                                    'v_obs': v_obs_track_np,
                                    'v_pred': v_pred_track_np,
                                }
                            )
                            extra_track_ids.add(track_id.upper())
                    else:
                        extra_track_series.append(
                            {
                                'id': track_id,
                                'time': t_axis_track,
                                'v_obs': v_obs_track.detach().cpu().numpy(),
                                'v_pred': v_pred_track.detach().cpu().numpy(),
                            }
                        )
                        extra_track_ids.add(track_id.upper())

                track_entry = {
                    'id': track_id,
                    'time': t_axis_track,
                    'lon_deg': np.rad2deg(phi_obs_track.detach().cpu().numpy()),
                }
                if sc_raw is not None and sc_raw.get('speed') is not None:
                    track_entry['speed'] = np.asarray(sc_raw.get('speed', []), dtype=float)
                elif v_obs_track is not None:
                    track_entry['speed'] = v_obs_track.detach().cpu().numpy()
                if not raw_obs_series:
                    track_lon_series.append(track_entry)
        else:
            if not raw_lon_series:
                track_lon_series.append(ace_track_entry)

        if raw_obs_series:
            track_lon_series = []
            for series in raw_lon_series:
                track_entry = {
                    'id': series.get('id', 'TRACK'),
                    'time': pd.to_datetime(np.asarray(series.get('time', []))),
                    'time_src': pd.to_datetime(np.asarray(series.get('time_src', series.get('time', [])))),
                    'lon_deg': np.asarray(series.get('lon_deg', []), dtype=float),
                }
                track_lon_series.append(track_entry)
            if raw_obs_series and extra_track_ids:
                # Keep panel-1 overlays as actual spacecraft longitudes.
                # Mapped/ballistic tracks are used elsewhere in extra_track_series.
                pass

        # Keep ACE track in panel-1 overlay payload even when remapping extras.
        if not any(str(track.get('id', '')).upper() == 'ACE' for track in track_lon_series):
            track_lon_series.append(ace_track_entry)

        return extra_track_series, track_lon_series

def _save_solution_plot(v_inner_eval, v0_eval, loss_fn, r, config, plot_path, title_template, batch_start_time_str, sensitivity_np=None):
        with torch.no_grad():
            v_full_eval = loss_fn.forward_model(v_inner_eval, v0_eval)
            r_grid = loss_fn.forward_model.r
            phi_grid = loss_fn.forward_model.phi
            v_pred_eval = sample_satellite_track(
                v_full_eval,
                r_grid,
                phi_grid,
                loss_fn.phi_obs_t,
                loss_fn.radius_obs_t,
                time_obs=loss_fn.time_obs_t,
                dt=config.dt,
            )

            v_inner_np = v_inner_eval.detach().cpu().numpy()
            v_pred_np = v_pred_eval.cpu().numpy()
            v_obs_np = loss_fn.v_obs.detach().cpu().numpy()

            _skip = int(getattr(config, 'overlap_hours', 0.0) * 3600.0 / config.dt)
            _t0 = min(_skip, len(v_obs_np) - 1)
            _vp, _vo = v_pred_np[_t0:], v_obs_np[_t0:]
            valid = np.isfinite(_vo) & np.isfinite(_vp)
            rmse = np.sqrt(np.mean((_vp[valid] - _vo[valid]) ** 2)) if np.any(valid) else float('nan')
            max_ae = np.max(np.abs(_vp[valid] - _vo[valid])) if np.any(valid) else float('nan')

        extra_track_series, track_lon_series = _build_plot_track_series(
            v_full_eval,
            r_grid,
            phi_grid,
            loss_fn,
            config,
            batch_start_time_str,
        )

        if title_template is None:
            title = None
        else:
            try:
                title = title_template.format(rmse=rmse, max_ae=max_ae)
            except Exception:
                title = title_template

        save_batch_solution_plot(
            v_inner_np,
            v_obs_np,
            v_pred_np,
            config,
            plot_path,
            title=title,
            batch_start_time=pd.to_datetime(batch_start_time_str) if batch_start_time_str else None,
            sensitivity_np=sensitivity_np,
            corrected_mask=None,
            extra_track_series=extra_track_series,
            track_lon_series=track_lon_series,
        )
        return rmse, max_ae, v_pred_np, v_obs_np


def run_training_stage(boundary, loss_fn, r, config, stage, stage_prefix="", check_graph=False,
                       checkpoint_path=None, batch_lon_deg_hourly=None, batch_lat_deg_hourly=None,
                       batch_image_slice=None, resume_optimizer_state_dict=None,
                       resume_scheduler_state_dict=None, resume_forward_extra_state_dict=None,
                       resume_best_metrics=None):
    """Run one optimization stage for a fixed forward scheme."""
    accel_lr_scale = float(getattr(config, 'learned_radial_accel_lr_scale', 0.05))
    optimizer_param_groups = _build_optimizer_param_groups(boundary, loss_fn, stage['lr'], accel_lr_scale=accel_lr_scale)
    optimizer = optim.Adam(optimizer_param_groups)
    total_optimized_params = sum(len(group['params']) for group in optimizer_param_groups)
    boundary_param_count = sum(1 for p in boundary.parameters() if p.requires_grad)
    if total_optimized_params != boundary_param_count:
        print(
            f"{stage_prefix}Optimizer includes {total_optimized_params} tensors "
            f"({boundary_param_count} from boundary + "
            f"{total_optimized_params - boundary_param_count} auxiliary)."
        )
    scheduler = None
    if getattr(config, 'adaptive_lr', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(config, 'lr_scheduler_factor', 0.5),
            patience=getattr(config, 'lr_scheduler_patience', 25),
            min_lr=getattr(config, 'lr_scheduler_min_lr', 1e-5),
        )

    if resume_optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state_dict)
            incompatible = False
            for group in optimizer.param_groups:
                for param in group['params']:
                    state = optimizer.state.get(param)
                    if state is None:
                        continue
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if key in state and state[key].shape != param.shape:
                            print(
                                f"{stage_prefix}WARNING: optimizer state shape mismatch for param "
                                f"{param.shape} vs {state[key].shape}; discarding resumed optimizer state"
                            )
                            incompatible = True
                            break
                    if incompatible:
                        break
                if incompatible:
                    break
            if incompatible:
                optimizer = optim.Adam(
                    _build_optimizer_param_groups(
                        boundary,
                        loss_fn,
                        stage['lr'],
                        accel_lr_scale=accel_lr_scale,
                    )
                )
                if scheduler is not None:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=getattr(config, 'lr_scheduler_factor', 0.5),
                        patience=getattr(config, 'lr_scheduler_patience', 25),
                        min_lr=getattr(config, 'lr_scheduler_min_lr', 1e-5),
                    )
        except Exception as e:
            print(f"{stage_prefix}WARNING: failed to load optimizer state for resume: {e}")

    if resume_forward_extra_state_dict is not None:
        try:
            accel = getattr(getattr(loss_fn, 'forward_model', None), 'learned_radial_accel', None)
            if accel is not None:
                accel.load_state_dict(resume_forward_extra_state_dict, strict=False)
        except Exception as e:
            print(f"{stage_prefix}WARNING: failed to load forward-model accel state for resume: {e}")

    if scheduler is not None and resume_scheduler_state_dict is not None:
        try:
            scheduler.load_state_dict(resume_scheduler_state_dict)
        except Exception as e:
            print(f"{stage_prefix}WARNING: failed to load scheduler state for resume: {e}")

    adaptive_epochs = bool(getattr(config, 'adaptive_epochs', False))
    min_epochs = int(stage.get('min_epochs', 0))
    max_epochs = int(stage.get('max_epochs', config.N_epochs))
    target_mse = float(stage.get('target_mse', -1.0))
    if target_mse < 0:
        target_mse = None

    early_stop_patience = int(getattr(config, 'early_stop_patience', 100))
    early_stop_min_delta = float(getattr(config, 'early_stop_min_delta', 1e-3))
    print_every = max(1, int(getattr(config, 'print_every', 20)))

    checkpoint_every = int(getattr(config, 'checkpoint_every', 0))
    #    save_best_checkpoint_on_improve = checkpoint_every == 0
    stage_scheme = str(stage.get('scheme', getattr(config, 'forward_scheme', 'fv')))
    refresh_v0_each_epoch = stage_scheme in {'fv', 'rk4_time_upwind'}

    loss_history = []
    grad_history = []
    reduced_basis = bool(getattr(boundary, 'is_reduced_basis', False))
    has_direct_boundary_field = hasattr(boundary, 'v_inner')
    cumulative_grad_sum = None
    if (not reduced_basis) and has_direct_boundary_field:
        cumulative_grad_sum = torch.zeros_like(boundary.v_inner, device=boundary.v_inner.device)
    epoch_count = 0
    best_loss = float('inf')
    best_loss_for_patience = float('inf')
    best_max_ae = float('inf')
    best_rmse = float('inf')
    best_epoch = None
    best_epoch_reported = None
    best_boundary_state = None
    best_v0_state = None
    epochs_no_improve = 0
    first_epoch_done = False
    zero_grad_streak = 0

    if resume_best_metrics is not None:
        best_loss = float(resume_best_metrics.get('best_loss', best_loss))
        best_max_ae = float(resume_best_metrics.get('best_max_ae', best_max_ae))
        best_rmse = float(resume_best_metrics.get('best_rmse', best_rmse))
        best_epoch = int(resume_best_metrics['best_epoch']) if resume_best_metrics.get('best_epoch') is not None else None
        best_epoch_reported = best_epoch
        best_loss_for_patience = best_loss
        if resume_best_metrics.get('best_boundary_state') is not None:
            best_boundary_state = resume_best_metrics.get('best_boundary_state')
        if resume_best_metrics.get('best_v0_state') is not None:
            best_v0_state = resume_best_metrics.get('best_v0_state')
        if best_boundary_state is None:
            best_boundary_state = {
                key: value.detach().clone()
                for key, value in boundary.state_dict().items()
            }
            best_v0_state = loss_fn.v0.detach().clone()

    def _eval_boundary():
        if batch_image_slice is None:
            raise ValueError("AIA slice required for batch; expected 'aia_slice' key in batch_spec with shape [C, Nphi, Nt].")
        return boundary(batch_image_slice)

    boundary.warmup_prior = str(stage.get('loss_mode', 'standard')).strip().lower() == 'prior'

    loss_mode = str(stage.get('loss_mode', 'standard')).strip().lower()
    est_points = int(getattr(loss_fn, 'estimated_train_points', 0))
    time_batch_size = int(getattr(config, 'time_batch_size', 0))
    max_minibatches_per_epoch = int(getattr(config, 'max_minibatches_per_epoch', 0))
    temporal_shuffle = bool(getattr(config, 'temporal_shuffle_minibatches', True))
    temporal_shuffle_seed = int(getattr(config, 'temporal_shuffle_seed', 42))
    full_window_validation = bool(getattr(config, 'full_window_validation', True))

    num_minibatches = 1
    if loss_mode != 'prior' and time_batch_size > 0 and est_points > 0:
        num_minibatches = int(np.ceil(est_points / float(max(1, time_batch_size))))
        num_minibatches = max(1, num_minibatches)
        if max_minibatches_per_epoch > 0:
            num_minibatches = min(num_minibatches, max_minibatches_per_epoch)

    if num_minibatches > 1:
        print(
            f"{stage_prefix}Temporal mini-batching enabled: "
            f"{num_minibatches} optimizer steps/epoch "
            f"(~{time_batch_size} samples/step, estimated_points={est_points})"
        )
        if temporal_shuffle:
            print(
                f"{stage_prefix}Temporal mini-batch shuffle enabled "
                f"(seed={temporal_shuffle_seed})."
            )
        if full_window_validation:
            print(
                f"{stage_prefix}Full-window validation enabled for monitoring metrics."
            )


    

    for epoch in range(max_epochs):
        mini_losses = []
        grad_norm_pre = 0.0
        invalid_loss_epoch = False
        regularization_scale = 1.0 / float(num_minibatches)
        batch_order = np.arange(num_minibatches, dtype=int)
        if num_minibatches > 1 and temporal_shuffle:
            epoch_rng = np.random.default_rng(temporal_shuffle_seed + epoch)
            epoch_rng.shuffle(batch_order)

        for step_idx, minibatch_idx in enumerate(batch_order):
            optimizer.zero_grad()

            v_inner = _eval_boundary()

            if check_graph and epoch == 0 and step_idx == 0 and not first_epoch_done:
                if not v_inner.requires_grad:
                    print(f"{stage_prefix}WARNING: v_inner does NOT require grad!")
                else:
                    print(f"{stage_prefix}v_inner requires grad: ✓")
                first_epoch_done = True

            if loss_mode == 'prior':
                loss = loss_fn(v_inner, loss_mode='prior')
            else:
                loss = loss_fn(
                    v_inner,
                    minibatch_idx=minibatch_idx,
                    num_minibatches=num_minibatches,
                    regularization_scale=regularization_scale,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"{stage_prefix}Epoch {epoch:3d} | "
                    f"minibatch {step_idx + 1}/{num_minibatches} "
                    f"(chunk={minibatch_idx + 1}) loss is NaN/Inf."
                )
                invalid_loss_epoch = True
                break

            if check_graph and epoch == 0 and step_idx == 0 and loss.grad_fn is None:
                print(f"{stage_prefix}WARNING: Loss has no grad_fn! Computational graph is broken!")
                invalid_loss_epoch = True
                break

            loss.backward()

            debug_nan_checks = bool(getattr(config, 'debug_nan_checks', False))
            grad_norm_step = 0.0
            for name, param in boundary.named_parameters():
                if param.grad is None:
                    if check_graph and epoch == 0 and step_idx == 0:
                        print(f"{stage_prefix}WARNING: Epoch 0, param {name} has NO gradient (None)!")
                    continue

                if debug_nan_checks:
                    bad_mask = ~torch.isfinite(param.grad)
                    if torch.any(bad_mask):
                        nan_count = torch.isnan(param.grad).sum().item()
                        inf_count = torch.isinf(param.grad).sum().item()
                        if epoch < 5 or epoch % print_every == 0:
                            print(
                                f"{stage_prefix}Epoch {epoch}: {name} grad has {nan_count} NaN, {inf_count} Inf "
                                f"out of {param.grad.numel()} - zeroing them out"
                            )
                        param.grad = torch.where(
                            bad_mask,
                            torch.zeros_like(param.grad),
                            param.grad
                        )

                param_grad_norm = param.grad.norm().item()
                grad_norm_step = max(grad_norm_step, param_grad_norm)
                if check_graph and epoch == 0 and step_idx == 0 and param_grad_norm == 0.0:
                    print(f"{stage_prefix}WARNING: Epoch 0, param {name} has zero gradient!")

                if (name == 'v_inner') and (cumulative_grad_sum is not None):
                    cumulative_grad_sum.add_(param.grad.abs())

            grad_norm_pre = max(grad_norm_pre, grad_norm_step)

            if grad_norm_step > 0:
                clip_params = []
                for group in optimizer.param_groups:
                    clip_params.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            optimizer.step()
            mini_losses.append(loss.item())

        if invalid_loss_epoch:
            print(f"{stage_prefix}Epoch {epoch:3d} | Loss is NaN/Inf! Stopping stage.")
            break

        if grad_norm_pre > 0:
            zero_grad_streak = 0
        else:
            zero_grad_streak += 1
            if zero_grad_streak == 5:
                print(
                    f"{stage_prefix}WARNING: gradient norm has been zero for 5 consecutive epochs; "
                    "model likely saturated or detached."
                )
            if zero_grad_streak >= 20 and (epoch + 1) >= min_epochs:
                print(
                    f"{stage_prefix}Stopping stage early at epoch {epoch}: "
                    "gradient norm remained zero for 20 epochs."
                )
                break

        if mini_losses:
            train_loss_item = float(np.mean(mini_losses))
        else:
            print(f"{stage_prefix}Epoch {epoch:3d} produced no minibatch losses; stopping stage.")
            break
        full_val_loss_item = float('nan')
        loss_item = train_loss_item

        if loss_mode != 'prior' and num_minibatches > 1 and full_window_validation:
            with torch.no_grad():
                v_inner_val = _eval_boundary()
                full_val_loss = loss_fn(
                    v_inner_val,
                    minibatch_idx=0,
                    num_minibatches=1,
                    regularization_scale=1.0,
                )
            if torch.isnan(full_val_loss) or torch.isinf(full_val_loss):
                print(
                    f"{stage_prefix}Epoch {epoch:3d} | full-window validation loss is NaN/Inf; "
                    "stopping stage."
                )
                break
            full_val_loss_item = float(full_val_loss.item())
            loss_item = full_val_loss_item

        max_ae = getattr(loss_fn, '_max_abs_error', float('nan'))
        rmse = getattr(loss_fn, '_rmse', float('nan'))
        # Keep RMSE-based best-model selection aligned with the plotted primary-track metric
        # (ACE when available) to avoid selecting a model that looks worse in saved plots.
        rmse_for_selection = rmse
        track_terms = getattr(loss_fn, '_track_loss_terms', [])
        if track_terms:
            ace_term = None
            first_finite_rmse = None
            for term in track_terms:
                term_rmse = float(term.get('rmse', float('nan')))
                if np.isfinite(term_rmse) and first_finite_rmse is None:
                    first_finite_rmse = term_rmse
                if str(term.get('id', '')).strip().upper() == 'ACE':
                    ace_term = term
                    break
            if ace_term is not None:
                ace_rmse = float(ace_term.get('rmse', float('nan')))
                if np.isfinite(ace_rmse):
                    rmse_for_selection = ace_rmse
            elif first_finite_rmse is not None:
                rmse_for_selection = first_finite_rmse

        metric_mode = str(getattr(config, 'best_model_metric', 'rmse')).strip().lower()
        if metric_mode not in {'loss', 'rmse', 'maxae'}:
            metric_mode = 'rmse'

        should_store_best_model = False
        if metric_mode == 'loss':
            if (loss_item < best_loss) or (
                np.isclose(loss_item, best_loss) and (not np.isnan(max_ae)) and (max_ae < best_max_ae)
            ):
                should_store_best_model = True
        elif metric_mode == 'rmse':
            if not np.isnan(rmse_for_selection):
                if (rmse_for_selection < best_rmse) or (
                    np.isclose(rmse_for_selection, best_rmse) and loss_item < best_loss
                ):
                    should_store_best_model = True
        else:  # metric_mode == 'maxae'
            if not np.isnan(max_ae):
                if (max_ae < best_max_ae) or (
                    np.isclose(max_ae, best_max_ae) and loss_item < best_loss
                ):
                    should_store_best_model = True

        if should_store_best_model:
            best_loss = loss_item
            if not np.isnan(max_ae):
                best_max_ae = max_ae
            if not np.isnan(rmse_for_selection):
                best_rmse = rmse_for_selection
            best_epoch = epoch
            best_boundary_state = {
                key: value.detach().clone()
                for key, value in boundary.state_dict().items()
            }
            best_v0_state = loss_fn.v0.detach().clone()
            best_optimizer_state_dict = optimizer.state_dict()
            best_scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
 

        loss_history.append(loss_item)
        grad_history.append(grad_norm_pre)
        epoch_count += 1

        if scheduler is not None:
            scheduler.step(loss_item)

        # Use a relative improvement threshold for early stopping.
        # If loss improves by more than early_stop_min_delta fraction,
        # reset the no-improve counter.
        rel_delta = max(early_stop_min_delta, 0.0)
        loss_ref = max(best_loss_for_patience, 1e-12)
        if loss_item < loss_ref * (1.0 - rel_delta):
            best_loss_for_patience = loss_item
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr_parts = []
        if len(optimizer.param_groups) > 0:
            lr_parts.append(f"LR_boundary: {optimizer.param_groups[0]['lr']:.3e}")
        if len(optimizer.param_groups) > 1:
            lr_parts.append(f"LR_accel: {optimizer.param_groups[1]['lr']:.3e}")
        lr_str = " | ".join(lr_parts) if lr_parts else "LR: n/a"
        if epoch % print_every == 0:

            if should_store_best_model:  # Check if we should store the best model

                current_state = {
                    k: v.detach().clone()
                    for k, v in boundary.state_dict().items()
                }

                boundary.load_state_dict(best_boundary_state)

                save_boundary_weights(
                    boundary,
                    checkpoint_path,
                    optimizer_state_dict=best_optimizer_state_dict,
                    scheduler_state_dict=best_scheduler_state_dict,
                    forward_extra_state_dict=loss_fn.forward_model.learned_radial_accel.state_dict()
                    if getattr(getattr(loss_fn, 'forward_model', None), 'learned_radial_accel', None) is not None
                    else None,
                )

                boundary.load_state_dict(current_state)
                print(
                    f"{stage_prefix}Best model updated "
                    f"(epoch {best_epoch}, {metric_mode}={best_rmse:.3f}, loss={best_loss:.6f})"
                )

                should_store_best_model = False
                
            rmse = getattr(loss_fn, '_rmse', float('nan'))
            max_ae = getattr(loss_fn, '_max_abs_error', float('nan'))
            loss_data_dbg = float(getattr(loss_fn, '_loss_data', float('nan')))
            loss_phi_dbg = float(getattr(loss_fn, '_loss_phi', float('nan')))
            loss_t_dbg = float(getattr(loss_fn, '_loss_t', float('nan')))
            track_terms = getattr(loss_fn, '_track_loss_terms', [])
            if track_terms:
                term_str = " | ".join(
                    [
                        f"{term['id']}:L={term['loss']:.1f},RMSE={term['rmse']:.1f},MaxAE={term['max_ae']:.1f}"
                        for term in track_terms
                    ]
                )
            else:
                term_str = ""
            print(
                f"{stage_prefix}Epoch {epoch:3d} | TrainLoss: {train_loss_item:.6f}"
                f"{(f' | FullValLoss: {full_val_loss_item:.6f}') if np.isfinite(full_val_loss_item) else ''} | "
                f"LossData: {loss_data_dbg:.6f} | LossPhi: {loss_phi_dbg:.6f} | LossT: {loss_t_dbg:.6f} | "
                f"Grad norm: {grad_norm_pre:.6e} | RMSE: {rmse:.1f} km/s | "
                f"MaxAE: {max_ae:.1f} km/s | {lr_str}"
                f"{(' | ' + term_str) if term_str else ''}"
            )
            accel_status = _format_learned_radial_accel_status(loss_fn)
            if accel_status:
                print(f"{stage_prefix}{accel_status}")
            accel_grad_status = _format_learned_radial_accel_grad_status(loss_fn)
            if accel_grad_status:
                print(f"{stage_prefix}{accel_grad_status}")
            if best_epoch is not None and best_epoch != best_epoch_reported:
                print(
                    f"{stage_prefix}Best model saved at epoch {best_epoch} | "
                    f"loss={best_loss:.6f} | RMSE={best_rmse:.1f} km/s | maxAE={best_max_ae:.1f} km/s"
                )
                best_epoch_reported = best_epoch

        early_stop_max_ae = float(getattr(config, 'early_stop_max_ae', -1.0))
        if early_stop_max_ae > 0:
            max_ae = getattr(loss_fn, '_max_abs_error', float('nan'))
            if not np.isnan(max_ae) and max_ae < early_stop_max_ae:
                print(
                    f"{stage_prefix}Reached early_stop_max_ae={early_stop_max_ae:.1f} km/s "
                    f"at epoch {epoch}; stopping stage."
                )
                break

        if adaptive_epochs and (epoch + 1) >= min_epochs:
            if target_mse is not None and loss_item <= target_mse:
                print(f"{stage_prefix}Reached target_mse={target_mse:.3f} at epoch {epoch}; stopping stage.")
                break
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"{stage_prefix}Early stopping at epoch {epoch}: "
                    f"no significant improvement for {early_stop_patience} epochs."
                )
                break

        should_save_solution = False
        if checkpoint_path:
            if epoch == 0:
                should_save_solution = True
            elif checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                should_save_solution = True

        if should_save_solution:
            with torch.no_grad():
                v_inner_eval = _eval_boundary()
                v_inner_ckpt = v_inner_eval.detach().cpu().numpy()
                if refresh_v0_each_epoch:
                    v0_eval = steady_state_propagate(v_inner_eval.detach()[:, 0], r, direction='outward')
                else:
                    v0_eval = loss_fn.v0
                batch_start_time_str = getattr(loss_fn, '_batch_start_time', None)

            plot_path = checkpoint_path.replace('_boundary.npz', '_solution.png')
            rmse, max_ae, v_pred_np, v_obs_np = _save_solution_plot(
                v_inner_eval,
                v0_eval,
                loss_fn,
                r,
                config,
                plot_path,
                title_template=f"{stage['name'].capitalize()} Epoch {epoch} | RMSE={{rmse:.1f}}  MaxAE={{max_ae:.1f}} km/s",
                batch_start_time_str=batch_start_time_str,
                sensitivity_np=None,
            )

            save_kwargs = {
                'v_inner': v_inner_ckpt,
                'v0': loss_fn.v0.detach().cpu().numpy(),
                # Keep checkpoint schema stable while avoiding object-dtype None values in NPZ.
                'boundary_sensitivity_map': np.zeros_like(v_inner_ckpt, dtype=np.float32),
                'start_time_batch': batch_start_time_str,
                'dt': config.dt,
                'time_hours': np.arange(len(v_obs_np), dtype=np.float32) * (config.dt / 3600.0),
                'v_obs_eval': v_obs_np,
                'v_pred_eval': v_pred_np,
                'epoch': epoch,
                'loss': loss_item,
                'rmse': rmse,
                'max_ae': max_ae,
            }
            if batch_lon_deg_hourly is not None:
                save_kwargs['lon_deg_hourly'] = np.asarray(batch_lon_deg_hourly, dtype=float)
            if batch_lat_deg_hourly is not None:
                save_kwargs['lat_deg_hourly'] = np.asarray(batch_lat_deg_hourly, dtype=float)
            np.savez(checkpoint_path, **save_kwargs)
            save_boundary_weights(
                boundary,
                checkpoint_path,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
                forward_extra_state_dict=loss_fn.forward_model.learned_radial_accel.state_dict()
                if getattr(getattr(loss_fn, 'forward_model', None), 'learned_radial_accel', None) is not None
                else None,
            )

   
            print(
            f"{stage_prefix}Solution saved to {checkpoint_path} (epoch {epoch}) | "
            f"RMSE={rmse:.1f}  MaxAE={max_ae:.1f} km/s"
            )

    if best_boundary_state is not None:
        boundary.load_state_dict(best_boundary_state, strict=False)
        if best_v0_state is not None:
            loss_fn.v0 = best_v0_state.detach().clone()
        metric_mode = str(getattr(config, 'best_model_metric', 'rmse')).strip().lower()
        if metric_mode not in {'loss', 'rmse', 'maxae'}:
            metric_mode = 'rmse'
        print(
            f"{stage_prefix}Restored best epoch {best_epoch} by {metric_mode} | "
            f"loss={best_loss:.6f}, RMSE={best_rmse:.3f} km/s, maxAE={best_max_ae:.3f} km/s "
            f"before returning from stage."
        )
        
        # Generate final plot from the restored best state
        
        with torch.no_grad():
            v_inner_best = _eval_boundary()
            v_inner_best_np = v_inner_best.detach().cpu().numpy()
            
        print('Computing final sensitivity map for best epoch solution...')
        final_sensitivity_np = compute_boundary_sensitivity_map(
            boundary,
            loss_fn,
            config,
            batch_image_slice=batch_image_slice,
        )
        if final_sensitivity_np is None:
            print(
                f"{stage_prefix}WARNING: final sensitivity map invalid; "
                "saving zero sensitivity map instead."
            )
            final_sensitivity_np = np.zeros_like(v_inner_best_np, dtype=np.float32)
        plot_path = checkpoint_path.replace('_boundary.npz', '_solution.png')
        rmse_best, max_ae_best, v_pred_best_np, v_obs_best_np = _save_solution_plot(
            v_inner_best,
            loss_fn.v0,
            loss_fn,
            r,
            config,
            plot_path,
            title_template="Best solution | RMSE={rmse:.1f}  MaxAE={max_ae:.1f} km/s",
            batch_start_time_str=getattr(loss_fn, '_batch_start_time', None),
            sensitivity_np=final_sensitivity_np,
        )
        print(f"{stage_prefix}Saved best-epoch solution plot to {plot_path}")

        # Re-save checkpoint NPZ and weights to ensure best-epoch state is persisted
        # This prevents next batch from resuming with final-epoch (potentially worse) checkpoint
        best_save_kwargs = {
            'v_inner': v_inner_best_np,
            'v0': loss_fn.v0.detach().cpu().numpy(),
            'boundary_sensitivity_map': final_sensitivity_np,
            'start_time_batch': getattr(loss_fn, '_batch_start_time', None),
            'dt': config.dt,
            'time_hours': np.arange(len(v_obs_best_np), dtype=np.float32) * (config.dt / 3600.0),
            'v_obs_eval': v_obs_best_np,
            'v_pred_eval': v_pred_best_np,
            'epoch': best_epoch,
            'loss': best_loss,
            'rmse': rmse_best,
            'max_ae': max_ae_best,
        }
        if batch_lon_deg_hourly is not None:
            best_save_kwargs['lon_deg_hourly'] = np.asarray(batch_lon_deg_hourly, dtype=float)
        if batch_lat_deg_hourly is not None:
            best_save_kwargs['lat_deg_hourly'] = np.asarray(batch_lat_deg_hourly, dtype=float)

        np.savez(checkpoint_path, **best_save_kwargs)
        save_boundary_weights(
            boundary,   
            checkpoint_path,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            forward_extra_state_dict=loss_fn.forward_model.learned_radial_accel.state_dict()
            if getattr(getattr(loss_fn, 'forward_model', None), 'learned_radial_accel', None) is not None
            else None,
        )
        print(f"{stage_prefix}Re-saved checkpoint and weights for best epoch {best_epoch} to {checkpoint_path}")

    else:
        raise RuntimeError(
            "No valid optimization state was found."
        )
    
    return (
        loss_history,
        grad_history,
        cumulative_grad_sum,
        epoch_count,
        best_epoch,
        best_loss,
        best_max_ae,
        final_sensitivity_np,
    )


