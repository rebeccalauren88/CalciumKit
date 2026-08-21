import math
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import jax.numpy as jnp
import CalciumKit.kalman as kalman

RAMIREZ_WHEEL_RADIUS = 0.00875  # meters -> 8.75 cm

def convert_degrees_to_positions(degrees: jnp.ndarray, wheel_radius: float = RAMIREZ_WHEEL_RADIUS) -> jnp.ndarray:
    """
    Convert wheel rotation in degrees to cumulative linear displacement in meters.
    Handles wrap-around discontinuities (e.g., -180/180 boundary).

    Args:
        degrees: The degrees of wheel rotation over time. Typically in range (-180, 180.1].
        wheel_radius: The radius of the wheel in meters.

    Returns:
        The estimated position in meters over time.
    """
    # Unwrap handles discontinuities automatically
    unwrapped = jnp.unwrap(degrees * jnp.pi / 180.0)
    
    # Convert to linear displacement
    positions = (unwrapped / (2 * jnp.pi)) * (2 * jnp.pi * wheel_radius)
    
    return positions


def create_state_matrix(delta_t: float) -> jnp.ndarray:
    """
    Create the state transition matrix A for a constant-acceleration model.

    Returns:
        The (3x3) state transition matrix.
    """
    return jnp.array([
        [1.0, delta_t, 0.5 * delta_t**2],
        [0.0, 1.0,     delta_t],
        [0.0, 0.0,     1.0]
    ])


def wheel_params(position_vector: jnp.ndarray, delta_t: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Initialize state and noise parameters for wheel motion inference.

    Args:
        position_vector: The observed positions in meters.
        delta_t: Time step between samples.

    Returns:
        Tuple of (initial_state, initial_covariance, A, C, Q, R).
    """
    initial_position = position_vector[0]
    initial_velocity = (position_vector[1] - position_vector[0]) / delta_t
    initial_acceleration = (position_vector[2] - 2 * position_vector[1] + position_vector[0]) / (delta_t ** 2)

    initial_state = jnp.array([initial_position, initial_velocity, initial_acceleration])
    initial_covariance = jnp.diag(jnp.array([1e-4, 1e-1, 1e-1]))

    A = create_state_matrix(delta_t)

    Q = 0.04 * jnp.array([
        [0.25 * delta_t**4, 0.5 * delta_t**3, 0.5 * delta_t**2],
        [0.5 * delta_t**3,     delta_t**2,       delta_t],
        [0.5 * delta_t**2,     delta_t,          1.0]
    ])

    C = jnp.array([[1.0, 0.0, 0.0]])  # Observe only position
    R = jnp.array([[1e-8]])          # Low measurement noise

    return initial_state, initial_covariance, A, C, Q, R


def wheel_kinematics(position_vector: jnp.ndarray, delta_t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimate position, velocity, and acceleration of the wheel using Kalman smoothing.

    Args:
        position_vector: Observed wheel positions (meters), shape (T,).
        delta_t: Sampling interval (seconds).

    Returns:
        smoothed_means: State estimates at each time step, shape (T, 3)
        smoothed_covs: Covariance estimates at each time step, shape (T, 3, 3)
    """
    if jnp.max(jnp.abs(position_vector)) > 10.0:
        warnings.warn("Input may be in degrees. Did you forget to convert to meters?")

    # Reshape to (T, 1) for Kalman filter
    y = position_vector.reshape(-1, 1)
    
    initial_state, initial_covariance, A, C, Q, R = wheel_params(position_vector, delta_t)

    # run EM for the covariance parameters
    Q, R, _, _ = kalman.kalman_em(
        y=y,
        A=A,
        C=C,
        Q_init=Q,
        R_init=R,
        m0=initial_state,
        P0=initial_covariance
    )

    smoothed_means, smoothed_covs = kalman.kalman_filter_smoother(
        y=y,
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=initial_state,
        P0=initial_covariance
    )
    return smoothed_means, smoothed_covs


def process_rotary_csv(csv_path, output_dir, D=18, sigma_deg=0.1):
    """Turn a raw rotary CSV (time_s, position_deg) into velocity/acceleration via spline fit."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {'time_s', 'position_deg'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'{csv_path.name} is missing required columns: {sorted(missing)}')

    t = df['time_s'].to_numpy(dtype=float)
    pos_deg_wrapped = df['position_deg'].to_numpy(dtype=float)

    # drop NaN rows, keeping the rest aligned
    valid = np.isfinite(t) & np.isfinite(pos_deg_wrapped)
    if not np.all(valid):
        df = df.loc[valid].reset_index(drop=True)
        t = df['time_s'].to_numpy(dtype=float)
        pos_deg_wrapped = df['position_deg'].to_numpy(dtype=float)

    if len(t) < 4:
        raise ValueError(f'{csv_path.name} has fewer than 4 valid samples; cannot fit cubic spline.')

    # UnivariateSpline needs strictly increasing time
    order = np.argsort(t)
    if not np.all(order == np.arange(len(t))):
        df = df.iloc[order].reset_index(drop=True)
        t = df['time_s'].to_numpy(dtype=float)
        pos_deg_wrapped = df['position_deg'].to_numpy(dtype=float)

    if np.any(np.diff(t) <= 0):
        raise ValueError(f'{csv_path.name} has duplicate or non-increasing time values.')

    theta_rad = np.deg2rad(pos_deg_wrapped)
    theta_rad_unwrapped = np.unwrap(theta_rad)
    theta_deg = np.rad2deg(theta_rad_unwrapped)

    # re-reference to start at 0, forward = increasing angle
    theta_deg0 = -(theta_deg - theta_deg[0])

    N = len(t)
    s = N * (sigma_deg ** 2)
    spl = UnivariateSpline(t, theta_deg0, k=3, s=s)

    vel_deg_s = spl.derivative(1)(t)
    acc_deg_s2 = spl.derivative(2)(t)

    theta_fit_deg = spl(t)
    dtheta_fit_deg = np.diff(theta_fit_deg)

    dist_total_deg = np.sum(np.abs(dtheta_fit_deg))
    dist_net_deg = theta_fit_deg[-1] - theta_fit_deg[0]

    scale = math.pi * D / 360.0
    vel_linear = vel_deg_s * scale
    acc_linear = acc_deg_s2 * scale
    dist_total_linear = dist_total_deg * scale
    dist_net_linear = dist_net_deg * scale
    cum_dist_linear = np.r_[0.0, np.cumsum(np.abs(dtheta_fit_deg)) * scale]

    out = df.copy()
    out['theta_deg_unwrapped0'] = theta_deg0
    out['theta_deg_spline'] = theta_fit_deg
    out['vel_deg_s_spline'] = vel_deg_s
    out['acc_deg_s2_spline'] = acc_deg_s2
    out['vel_linear_spline'] = vel_linear
    out['acc_linear_spline'] = acc_linear
    out['cum_dist_linear_spline'] = cum_dist_linear

    output_path = output_dir / f'processedRotary_{csv_path.stem}.csv'
    out.to_csv(output_path, index=False)

    return {
        'file': csv_path.name,
        'output_file': output_path.name,
        'n_samples': len(out),
        'total_distance_cm': dist_total_linear,
        'net_displacement_cm': dist_net_linear,
        'output_path': str(output_path),
    }


def mean_in_epoch(time, signal, start_times, duration):
    """One mean value per epoch/trial; drops a trial only if it has no samples in the window."""
    vals = []
    for start in start_times:
        mask = (time >= start) & (time < start + duration)
        if np.sum(mask) > 0:
            vals.append(np.nanmean(signal[mask]))
    return np.array(vals)