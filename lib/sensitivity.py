import torch
from helpers import sample_satellite_track

def _compute_data_loss_only(loss_fn, v_inner, dt):
    """Compute aggregated data-misfit term only (no regularization)."""
    v_full = loss_fn.forward_model(v_inner, loss_fn.v0)
    if torch.isnan(v_full).any():
        print("Sensitivity: v_full contains NaNs")
        return None


    r = loss_fn.forward_model.r
    phi = loss_fn.forward_model.phi

    track_losses = []
    if bool(getattr(loss_fn, 'use_multispacecraft', False)) and int(getattr(loss_fn, 'num_tracks', 0)) > 0:
        for idx in range(loss_fn.num_tracks):
       
            track_id = loss_fn.track_ids[idx]
            
            v_obs_track = getattr(loss_fn, f'v_obs_track_{idx}')
            phi_obs_track = getattr(loss_fn, f'phi_obs_track_{idx}')
            radius_obs_track = getattr(loss_fn, f'radius_obs_track_{idx}')

            time_obs_track = getattr(loss_fn, f"time_obs_track_{idx}")

            v_pred_track = sample_satellite_track(
                v_full,
                r,
                phi,
                phi_obs_track,
                radius_obs_track,
                time_obs=time_obs_track,
                dt=dt,
            )

            if torch.isnan(v_pred_track).any():

                nan_mask = torch.isnan(v_pred_track)

                if nan_mask.any():

                    tmin = 0.0
                    tmax = float(dt) * float(v_full.shape[2] - 1)

                    expected_nan_mask = (
                        ~torch.isfinite(phi_obs_track)
                        | ~torch.isfinite(radius_obs_track)
                        | ~torch.isfinite(time_obs_track)
                        | (time_obs_track < tmin)
                        | (time_obs_track > tmax)
                    )

                    unexpected_nan_mask = nan_mask & ~expected_nan_mask

                    if unexpected_nan_mask.any():
                        print(f"Sensitivity: track {idx} prediction contains unexpected NaNs at indices {unexpected_nan_mask.nonzero(as_tuple=True)}")
                   

            result = loss_fn._compute_track_misfit(v_pred_track, v_obs_track)
            if result is None:
                continue

            track_losses.append(result["loss_data"])
            
            #if result is None:
            #    print(f"Sensitivity: track {idx} produced no valid compute_track_misfit result")
            #else:
            #    track_losses.append(result["loss_data"])

        if not track_losses:
            print("Sensitivity: no valid track losses")
            return None

    else:
        v_pred_track = sample_satellite_track(
            v_full,
            r,
            phi,
            loss_fn.phi_obs_t,
            loss_fn.radius_obs_t,
            time_obs=loss_fn.time_obs_t,
            dt=dt,
        )
        if not torch.isfinite(v_full).all():
            print("Sensitivity: v_full contains NaNs")
            return None

        result = loss_fn._compute_track_misfit(v_pred_track, loss_fn.v_obs)
        if result is not None:
            track_losses.append(result["loss_data"])

    if not track_losses:
        return None

    return torch.stack(track_losses).mean()

def compute_boundary_sensitivity_map(
    boundary,
    loss_fn,
    config,
    batch_image_slice,
):
    """
    Compute sensitivity as d(data_misfit)/d(v_inner) at each (phi, t).

    Regularization terms remain in optimization loss, but are excluded from
    this sensitivity map by design.
    """

    if batch_image_slice is None:
        raise ValueError("batch_image_slice is required")


    boundary.zero_grad(set_to_none=True)
    v_inner = boundary(batch_image_slice)

    loss_val = _compute_data_loss_only(loss_fn, v_inner, config.dt)
    if loss_val is None or torch.isnan(loss_val) or torch.isinf(loss_val):
        print("  Sensitivity map: data loss is invalid, skipping sensitivity map")
        return None

    grad = torch.autograd.grad(loss_val, v_inner, retain_graph=False, create_graph=False)[0]
    if grad is None:
        print("  Sensitivity map: failed to compute loss gradient")
        return None

    aggregation = str(
        getattr(config, "sensitivity_aggregation", "squared")
    ).lower()

    if aggregation == "squared":
        sensitivity = grad.square()
    elif aggregation == "abs":
        sensitivity = grad.abs()
    else:
        raise ValueError(
            f"Unknown sensitivity aggregation '{aggregation}'"
        )

    return sensitivity.detach().cpu().numpy()


