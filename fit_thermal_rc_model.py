#!/usr/bin/env python3
"""Fit the vessel thermal RC model with measured coldhead temperature input.

Model (all powers are positive in the arrow direction used below):

    C_t  dT_t/dt  = I_2 + i0(T_t)
                     - (T_t - T_b)  / R_tb
                     - (T_t - T_CH_measured(t)) / R_tc

    C_b  dT_b/dt  = I_1
                     + (T_t - T_b)  / R_tb
                     - (T_b - T_CH_measured(t)) / R_bc

    i0(T_t) = -0.345 * T_t + 110.7  [W], with T_t in K.

Only R_tb, C_t, C_b are fitted. The steady-state calibration fixes
R_tc = 1/0.32768 K/W and R_bc = 1/0.75838 K/W, even if callers pass old bounds.
C_CH remains a compatibility field fixed to zero; there is NO CH balance
equation in this model. Measured T_CH drives the two vessel states and is
never included as a predicted output in the loss or validation metrics.

R is in K/W, thermal capacitance C is in J/K,
temperature MUST be in K for the calibrated leak law, time is in seconds,
and every I column is a thermal power in W. CH_power is retained for the
existing data/plot interfaces but is not used in the model dynamics. Existing
data preparation is unchanged, including its finite-value checks on CH_power.
Validation is conditional on the measured T_CH trajectory in the new interval.

fit_experiment_arrays() and validate_experiment_arrays() retain their calling
signatures. Direct simulate_temperatures() calls must additionally pass the
entire boundary trajectory as CH_temp=... . Three-column temperature arrays
are retained: their third column is the supplied CH boundary, NOT a prediction.

The code fits the temperatures by integrating the ODE.  It does not
differentiate measured temperatures, which avoids amplifying sensor noise.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import cont2discrete


PARAMETER_NAMES = ("R_tb", "R_tc", "R_bc", "C_t", "C_b", "C_CH")
FITTED_CHANNELS = ("T_t", "T_b")
COLDHEAD_MODE = "measured_boundary"

# Steady-state calibration. Edit the conductances here and reload the module
# if a future calibration changes. These values are NOT fitted in this version.
FIXED_G_TC = 0.32768  # W/K
FIXED_G_BC = 0.75838  # W/K
FIXED_R_TC = 1.0 / FIXED_G_TC
FIXED_R_BC = 1.0 / FIXED_G_BC


def i0(T_top: float | np.ndarray) -> float | np.ndarray:
    """External heat input [W] as a function of top temperature [K].

    Change a and b HERE to update fitting, validation, and exported heat flows.
    The simulator evaluates this law using its predicted top temperature.
    a and b are prescribed calibration coefficients, not fitted parameters.

    This exact state-space simulator requires an affine law a*T_top + b.
    A nonlinear replacement needs a nonlinear integrator; it must not be
    silently treated as an affine law by changing this function alone.
    No clipping is applied: check the calibration range before extrapolation.
    """
    a = -0.3348  # W/K
    b = 110.7   # W
    temperature = np.asarray(T_top, dtype=float)
    power = a * temperature + b
    return float(power) if power.ndim == 0 else power


def _heat_leak_coefficients() -> tuple[float, float]:
    """Obtain the affine coefficients from i0(), the single source of truth."""
    a = float((i0(200.0) - i0(100.0)) / 100.0)
    b = float(i0(100.0) - 100.0 * a)
    probes = np.array([0.0, 50.0, 100.0, 150.0, 175.0, 200.0, 250.0, 300.0, 400.0])
    values = np.asarray(i0(probes), dtype=float)
    if (not np.isfinite([a, b]).all() or not np.isfinite(values).all()
            or not np.allclose(values, a * probes + b, rtol=1e-11, atol=1e-10)):
        raise ValueError(
            "i0(T_top) must be an affine law a*T_top + b for this exact "
            "state-space solver. A nonlinear law needs a nonlinear integrator."
        )
    return a, b


def _heat_leak_metadata() -> dict[str, object]:
    a, b = _heat_leak_coefficients()
    return {"formula": "i0(T_top) = a*T_top + b", "a_W_per_K": a,
            "b_W": b, "temperature_unit": "K", "fitted": False}


@dataclass(frozen=True)
class ThermalParameters:
    R_tb: float
    R_tc: float
    R_bc: float
    C_t: float
    C_b: float
    C_CH: float = 0.0


@dataclass(frozen=True)
class ColumnNames:
    time: str = "Time"
    T_b: str = "T_b"
    I_1: str = "I_1"
    T_t: str = "T_t"
    I_2: str = "I_2"
    T_CH: str = "T_CH"
    I_CH: str = "I_CH"


@dataclass
class FitResult:
    """Temperature columns: [fitted T_t, fitted T_b, supplied T_CH boundary]."""
    parameters: ThermalParameters
    parameter_std: dict[str, float]
    success: bool
    message: str
    cost: float
    reduced_chi_square: float
    time_s: np.ndarray
    measured_temperature: np.ndarray
    fitted_temperature: np.ndarray
    inputs_w: np.ndarray
    fixed_parameter_names: tuple[str, ...] = ()
    heat_leak_model: dict[str, object] = field(default_factory=_heat_leak_metadata)
    fitted_channels: tuple[str, ...] = FITTED_CHANNELS
    coldhead_mode: str = COLDHEAD_MODE


@dataclass
class ValidationResult:
    """Vessel prediction conditional on measured CH, without parameter refit.

    predicted_temperature[:, 2] copies the supplied boundary for compatibility;
    residual_temperature[:, 2] is NaN because CH is not a predicted output.
    metrics includes only T_t and T_b.
    """

    parameters: ThermalParameters
    time_s: np.ndarray
    measured_temperature: np.ndarray
    predicted_temperature: np.ndarray
    residual_temperature: np.ndarray
    inputs_w: np.ndarray
    metrics: dict[str, dict[str, float]]
    metric_start_s: float
    heat_leak_model: dict[str, object] = field(default_factory=_heat_leak_metadata)
    fitted_channels: tuple[str, ...] = FITTED_CHANNELS
    coldhead_mode: str = COLDHEAD_MODE


DEFAULT_INITIAL = ThermalParameters(
    R_tb=0.4,   # K/W
    R_tc=FIXED_R_TC,  # K/W; fixed
    R_bc=FIXED_R_BC,  # K/W; fixed
    C_t=1.0e2,   # J/K
    C_b=1.0e2,   # J/K
    C_CH=0.0,   # J/K; fixed by equal lower/upper bounds
)


DEFAULT_LOWER = ThermalParameters(
    R_tb=0.1,
    R_tc=FIXED_R_TC,
    R_bc=FIXED_R_BC,
    C_t=1.0,
    C_b=1.0,
    C_CH=0.0,
)


DEFAULT_UPPER = ThermalParameters(
    R_tb=2,
    R_tc=FIXED_R_TC,
    R_bc=FIXED_R_BC,
    C_t=1.0e5,
    C_b=1.0e5,
    C_CH=0.0,
)


def _parameter_array(parameters: ThermalParameters | Mapping[str, float]) -> np.ndarray:
    if is_dataclass(parameters) and not isinstance(parameters, type):
        values = asdict(parameters)
    else:
        values = parameters
    return np.asarray([float(values[name]) for name in PARAMETER_NAMES], dtype=float)


def _parameters_from_array(values: Sequence[float]) -> ThermalParameters:
    if len(values) != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} parameter values.")
    return ThermalParameters(**dict(zip(PARAMETER_NAMES, map(float, values))))


def _time_to_seconds(series: pd.Series) -> np.ndarray:
    """Accept datetime Series/Index/arrays, timestamp strings, or numeric seconds.

    Detect datetime BEFORE numeric conversion: pandas may otherwise turn a
    datetime64[ns] column into integers measured in nanoseconds. Subtract the
    earliest timestamp before conversion to float to retain time precision.
    Missing entries remain NaN and are removed jointly with their signal rows.
    All timestamps must refer to the same clock/time zone.
    """
    series = pd.Series(series).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = series
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        present = series.notna()
        if pd.api.types.is_numeric_dtype(series.dtype) or numeric[present].notna().all():
            values = numeric.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            if not finite.any():
                raise ValueError("No valid numeric times.")
            values[~finite] = np.nan
            return values - np.min(values[finite])
        cleaned = series.astype("string").str.strip().str.replace(
            r"(?<=\d)_(?=\d)", ":", regex=True
        )
        # Accept both logger strings (HH_MM_SS) and ordinary timestamps.
        parsed = cleaned.map(
            lambda value: pd.to_datetime(value, errors="coerce")
            if pd.notna(value) else pd.NaT
        )
        parsed = pd.to_datetime(parsed, errors="coerce")
    if not parsed.notna().any():
        raise ValueError("No valid timestamps. Check the Time column format.")
    return (parsed - parsed.min()).dt.total_seconds().to_numpy(dtype=float)


def _piecewise_linear_bin_average(
    time_s: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Time-average a signal in bins while preserving its integrated area.

    The measured signal is treated as piecewise linear.  For power columns,
    average_power * bin_width therefore equals the integrated bin energy.
    """
    knots = np.unique(np.concatenate((time_s, edges)))
    knots = knots[(knots >= edges[0]) & (knots <= edges[-1])]
    y = np.interp(knots, time_s, values)
    segment_area = 0.5 * (y[:-1] + y[1:]) * np.diff(knots)
    bin_index = np.searchsorted(edges, knots[:-1], side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(edges) - 1)
    area = np.bincount(
        bin_index[valid], weights=segment_area[valid], minlength=len(edges) - 1
    )
    return area / np.diff(edges)


def prepare_data(
    dataframe: pd.DataFrame,
    columns: ColumnNames = ColumnNames(),
    fit_dt_s: float | None = None,
    max_points: int = 8000,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Clean, sort, and energy-average the data onto a uniform time grid.

    Returns
    -------
    time_s:
        Uniform bin-center times.
    measured_temperature:
        Array with columns [T_t, T_b, T_CH].
    inputs_w:
        Array with columns [I_2, I_1, I_CH].
    fit_dt_s:
        Actual fitting time step.
    """
    required = [
        columns.time,
        columns.T_t,
        columns.T_b,
        columns.T_CH,
        columns.I_2,
        columns.I_1,
        columns.I_CH,
    ]
    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return prepare_array_data(
        time_relative=dataframe[columns.time],
        bot_flange=dataframe[columns.T_b],
        top_flange=dataframe[columns.T_t],
        CH_temp=dataframe[columns.T_CH],
        CH_power=dataframe[columns.I_CH],
        Heater1=dataframe[columns.I_1],
        Heater2=dataframe[columns.I_2],
        fit_dt_s=fit_dt_s,
        max_points=max_points,
        max_gap_s=max_gap_s,
    )


def prepare_array_data(
    time_relative: Sequence[object],
    bot_flange: Sequence[float],
    top_flange: Sequence[float],
    CH_temp: Sequence[float],
    CH_power: Sequence[float],
    Heater1: Sequence[float],
    Heater2: Sequence[float],
    fit_dt_s: float | None = None,
    max_points: int = 8000,
    max_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prepare numeric seconds OR original datetime timestamps from the notebook.

    Despite its legacy name, time_relative accepts df["Time"] directly.
    Use a common absolute Time column when joining files whose elapsed counters
    restart. All channels are positional: supply arrays from the same rows.
    Datetime values are converted to elapsed SECONDS, then complete rows are
    sorted and duplicate timestamps averaged. Series row-index labels are ignored.
    max_gap_s defaults to max(60 seconds, 10 * median raw cadence); larger
    gaps are rejected rather than silently inventing missing heater histories.

    Parameters map to the thermal model as follows:

    * top_flange -> T_t
    * bot_flange -> T_b
    * CH_temp -> T_CH
    * Heater2 -> I_2 (top heater)
    * Heater1 -> I_1 (bottom heater)
    * CH_power -> I_CH (positive cooling power removed by the coldhead)

    The returned temperature column order is [T_t, T_b, T_CH], and the power
    column order is [I_2, I_1, I_CH].
    """
    names = (
        "time_relative",
        "bot_flange",
        "top_flange",
        "CH_temp",
        "CH_power",
        "Heater1",
        "Heater2",
    )
    raw_arrays = (
        time_relative,
        bot_flange,
        top_flange,
        CH_temp,
        CH_power,
        Heater1,
        Heater2,
    )
    arrays = [_time_to_seconds(pd.Series(time_relative))]
    for value in raw_arrays[1:]:
        values = np.asarray(value)
        if values.ndim != 1:
            raise ValueError("Each temperature/power input must be one-dimensional.")
        arrays.append(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float))
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        shape_text = ", ".join(
            f"{name}={len(value)}" for name, value in zip(names, arrays)
        )
        raise ValueError(f"All input arrays must have the same length: {shape_text}")

    frame = pd.DataFrame(dict(zip(names, arrays)))
    before = len(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < before:
        warnings.warn(f"Dropped {before - len(frame)} rows with invalid times/signals.", stacklevel=2)
    # Sort WHOLE rows: all six channels stay paired with their timestamps.
    frame = frame.sort_values("time_relative", kind="stable")
    duplicates = int(frame["time_relative"].duplicated().sum())
    if duplicates:
        warnings.warn(
            f"Merged {duplicates} duplicate timestamp rows by channel-wise mean. "
            "Check overlapping files if simultaneous readings disagree.", stacklevel=2
        )
    frame = frame.groupby("time_relative", as_index=False).mean()
    if len(frame) < 10:
        raise ValueError("At least 10 finite, unique time samples are required.")

    time_raw = frame["time_relative"].to_numpy(dtype=float)
    time_raw = time_raw - time_raw[0]
    if np.any(np.diff(time_raw) <= 0):
        raise ValueError("time_relative must increase monotonically.")

    median_dt = float(np.median(np.diff(time_raw)))
    duration = float(time_raw[-1])
    gap_limit = max(60.0, 10.0 * median_dt) if max_gap_s is None else float(max_gap_s)
    if np.isnan(gap_limit) or gap_limit <= 0:
        raise ValueError("max_gap_s must be positive (or None for automatic detection).")
    largest_gap = float(np.max(np.diff(time_raw)))
    if largest_gap > gap_limit:
        raise ValueError(
            f"Data gap of {largest_gap:g} s exceeds max_gap_s={gap_limit:g} s. "
            "Select continuous intervals separately: the heater input during the gap "
            "is unknown. Only increase max_gap_s if interpolation is justified."
        )
    if max_points < 6:
        raise ValueError("max_points must be at least 6.")
    if fit_dt_s is None:
        fit_dt_s = max(median_dt, duration / max(max_points - 1, 1))
    if not np.isfinite(fit_dt_s) or fit_dt_s <= 0:
        raise ValueError("fit_dt_s must be positive.")

    n_bins = int(np.floor(duration / fit_dt_s))
    if n_bins < 5:
        raise ValueError(
            "fit_dt_s is too large for this data interval; fewer than 5 bins remain."
        )
    edges = np.arange(n_bins + 1, dtype=float) * fit_dt_s
    centers = 0.5 * (edges[:-1] + edges[1:])

    averaged: dict[str, np.ndarray] = {}
    for name in names[1:]:
        averaged[name] = _piecewise_linear_bin_average(
            time_raw, frame[name].to_numpy(dtype=float), edges
        )

    measured_temperature = np.column_stack(
        (averaged["top_flange"], averaged["bot_flange"], averaged["CH_temp"])
    )
    inputs_w = np.column_stack(
        (averaged["Heater2"], averaged["Heater1"], averaged["CH_power"])
    )
    return centers - centers[0], measured_temperature, inputs_w, float(fit_dt_s)


def continuous_state_matrices(
    parameters: ThermalParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A, B for x_dot = A x + B [I2, I1, T_CH_measured, 1].

    x = [T_t, T_b]. The third input is temperature [K], NOT cooling power.
    Includes i0(predicted T_t) as state feedback. C_CH has no dynamics.
    """
    p = parameters
    values = _parameter_array(p)
    if not np.isfinite(values).all() or np.any(values[:5] <= 0):
        raise ValueError("R_tb, R_tc, R_bc, C_t, C_b must be positive and finite.")
    if p.C_CH != 0:
        raise ValueError("This measured-CH model requires C_CH=0 (compatibility field).")
    g_tb = 1.0 / p.R_tb
    g_tc = 1.0 / p.R_tc
    g_bc = 1.0 / p.R_bc
    leak_a, leak_b = _heat_leak_coefficients()

    a = np.array(
        [
            [(leak_a - g_tb - g_tc) / p.C_t, g_tb / p.C_t],
            [g_tb / p.C_b, -(g_tb + g_bc) / p.C_b],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [1.0 / p.C_t, 0.0, g_tc / p.C_t, leak_b / p.C_t],
            [0.0, 1.0 / p.C_b, g_bc / p.C_b, 0.0],
        ],
        dtype=float,
    )
    return a, b


def simulate_temperatures(
    time_s: np.ndarray,
    inputs_w: np.ndarray,
    initial_temperature: Sequence[float],
    parameters: ThermalParameters,
    *,
    CH_temp: Sequence[float] | None = None,
) -> np.ndarray:
    """Predict T_t/T_b, given an entire measured CH_temp trajectory.

    Heater powers are held over [time_s[k], time_s[k+1]); measured CH temperature
    is linearly interpolated between samples. The update is exact for these
    forcing assumptions and the affine leak law. No measured vessel temperature
    after the initial sample is fed into the simulated vessel states.

    inputs_w retains columns [I2, I1, ICH]; ICH is unused. Return columns remain
    [predicted T_t, predicted T_b, supplied CH_temp] for interface compatibility.
    initial_temperature accepts two or three entries; its CH entry is unused.
    Direct callers must supply CH_temp; fitting/validation wrappers do this.
    """
    time_s = np.asarray(time_s, dtype=float)
    inputs_w = np.asarray(inputs_w, dtype=float)
    initial_temperature = np.asarray(initial_temperature, dtype=float)
    if time_s.ndim != 1 or len(time_s) < 2 or not np.isfinite(time_s).all():
        raise ValueError("time_s must contain at least two finite times.")
    if inputs_w.shape != (len(time_s), 3):
        raise ValueError("inputs_w must have shape (n_samples, 3): [I2, I1, ICH].")
    if not np.isfinite(inputs_w[:, :2]).all():
        raise ValueError("Heater powers must be finite.")
    if (initial_temperature.shape not in ((2,), (3,))
            or not np.isfinite(initial_temperature[:2]).all()):
        raise ValueError("initial_temperature must contain finite [T_t, T_b] (optional T_CH).")
    if CH_temp is None:
        raise ValueError(
            "Pass CH_temp=<measured CH temperature array> for the new boundary model. "
            "The initial CH temperature alone does not specify the full boundary."
        )
    boundary = np.asarray(CH_temp, dtype=float)
    if boundary.shape != (len(time_s),) or not np.isfinite(boundary).all():
        raise ValueError("CH_temp must contain one finite temperature [K] per time sample.")
    dt = float(np.median(np.diff(time_s)))
    if dt <= 0 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    if not np.allclose(np.diff(time_s), dt, rtol=1e-5, atol=max(1e-9, dt * 1e-8)):
        raise ValueError("time_s must be uniformly spaced.")

    a, b = continuous_state_matrices(parameters)
    # Augment by the prescribed CH ramp only to obtain its exact integral.
    # This auxiliary coordinate is NOT a thermal state or a fitted CH model.
    a_aug = np.zeros((3, 3))
    a_aug[:2, :2] = a
    a_aug[:2, 2] = b[:, 2]
    b_aug = np.zeros((3, 4))
    b_aug[:2, :3] = b[:, [0, 1, 3]]  # heater2, heater1, constant 1
    b_aug[2, 3] = 1.0                # prescribed slope of the CH boundary
    ad, bd, _, _, _ = cont2discrete(
        (a_aug, b_aug, np.eye(3), np.zeros((3, 4))), dt, method="zoh"
    )
    u = np.column_stack((inputs_w[:-1, :2], np.ones(len(time_s) - 1),
                         np.diff(boundary) / dt))
    predicted = np.empty((len(time_s), 3), dtype=float)
    predicted[:, 2] = boundary  # compatibility column, not a prediction
    predicted[0, :2] = initial_temperature[:2]
    for index in range(len(time_s) - 1):
        predicted[index + 1, :2] = (ad @ predicted[index] + bd @ u[index])[:2]
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Temperature simulation produced non-finite values.")
    return predicted


def fit_thermal_model(
    time_s: np.ndarray,
    measured_temperature: np.ndarray,
    inputs_w: np.ndarray,
    initial: ThermalParameters = DEFAULT_INITIAL,
    lower: ThermalParameters = DEFAULT_LOWER,
    upper: ThermalParameters = DEFAULT_UPPER,
    temperature_sigma_k: Sequence[float] = (0.02, 0.02, 0.02),
    max_nfev: int = 300,
) -> FitResult:
    """Fit R_tb, C_t, C_b to vessel temperatures with measured T_CH forcing.

    R_tc/R_bc are always fixed to FIXED_R_TC/FIXED_R_BC, and C_CH to zero.
    These constraints override legacy initial values/bounds for those fields.
    Bounds and initial values for R_tb/C_t/C_b keep their previous behavior;
    equal bounds can additionally fix any of these three parameters.
    temperature_sigma_k accepts [sigma_top, sigma_bottom] or the legacy three
    entries; its third entry is ignored. Only the two vessel channels enter
    the residual vector. The CH column is an input, never a fit target.
    """
    measured_temperature = np.asarray(measured_temperature, dtype=float)
    sigma = np.asarray(temperature_sigma_k, dtype=float)
    if measured_temperature.shape != (len(time_s), 3):
        raise ValueError("measured_temperature must have columns [T_t, T_b, T_CH].")
    if not np.isfinite(measured_temperature).all():
        raise ValueError("Measured temperatures must be finite.")
    if sigma.shape not in ((2,), (3,)):
        raise ValueError("temperature_sigma_k must contain two or three entries.")
    sigma = sigma[:2]
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Top/bottom temperature uncertainties must be finite and positive.")

    p0 = _parameter_array(initial)
    lb = _parameter_array(lower)
    ub = _parameter_array(upper)
    locked_indices = [1, 2, 5]  # R_tc, R_bc, C_CH
    locked_values = np.array([FIXED_R_TC, FIXED_R_BC, 0.0])
    if not np.isfinite(locked_values).all() or np.any(locked_values[:2] <= 0):
        raise ValueError("The fixed coldhead-link resistances must be positive and finite.")
    if (not np.array_equal(lb[locked_indices], locked_values)
            or not np.array_equal(ub[locked_indices], locked_values)):
        warnings.warn(
            "R_tc/R_bc/C_CH bounds were overridden: this version fixes them to "
            "the steady-state calibration and C_CH=0. Only R_tb, C_t, C_b can be fitted.",
            stacklevel=2,
        )
    p0[locked_indices] = lb[locked_indices] = ub[locked_indices] = locked_values
    if not np.isfinite(np.concatenate((p0, lb, ub))).all():
        raise ValueError("Initial parameters and bounds must be finite.")
    if np.any(lb > ub):
        raise ValueError("Each lower bound must be <= its upper bound.")
    free = lb < ub
    fixed = ~free
    if np.any(lb[:5] <= 0):
        raise ValueError("R and vessel C bounds must be positive.")
    p0[fixed] = lb[fixed]
    if np.any(p0[free] <= lb[free]) or np.any(p0[free] >= ub[free]):
        raise ValueError("Each FREE initial parameter must lie strictly inside its bounds.")

    def unpack(log_free: np.ndarray) -> np.ndarray:
        values = lb.copy()
        values[free] = np.exp(log_free)
        return values

    initial_temperature = measured_temperature[0]

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        parameters = _parameters_from_array(unpack(log_parameters))
        predicted = simulate_temperatures(
            time_s, inputs_w, initial_temperature, parameters,
            CH_temp=measured_temperature[:, 2],
        )
        # Initial top/bottom values are imposed, so omit their trivial residuals.
        return ((predicted[1:, :2] - measured_temperature[1:, :2]) / sigma).ravel()

    if np.any(free):
        optimum = least_squares(
            residual,
            np.log(p0[free]),
            bounds=(np.log(lb[free]), np.log(ub[free])),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=max_nfev,
            verbose=1,
        )
        fitted_array = unpack(optimum.x)
        fit_residual = optimum.fun
        success, message, cost = bool(optimum.success), str(optimum.message), float(optimum.cost)
    else:
        fitted_array = lb.copy()
        fit_residual = residual(np.empty(0))
        success, message = True, "All parameters fixed; simulated without optimization."
        cost = float(np.sum(np.sqrt(1.0 + fit_residual**2) - 1.0))
    fitted = _parameters_from_array(fitted_array)
    predicted = simulate_temperatures(
        time_s, inputs_w, initial_temperature, fitted,
        CH_temp=measured_temperature[:, 2],
    )

    degrees_of_freedom = max(fit_residual.size - int(free.sum()), 1)
    reduced_chi_square = float(np.sum(fit_residual**2) / degrees_of_freedom)

    # Approximate one-sigma uncertainties.  With robust loss these are useful as
    # diagnostics, but bootstrap intervals are preferable for publication.
    std_values = np.zeros_like(fitted_array)
    if np.any(free):
        try:
            # An inverse of J.T@J can hide weak directions. Inspect the SVD
            # of the free-parameter Jacobian before estimating uncertainties.
            _, singular_values, vt = np.linalg.svd(optimum.jac, full_matrices=False)
            weak = (singular_values[-1] <= max(singular_values[0] * 1e-10, 1e-14))
            if weak:
                warnings.warn(
                    "Free parameters are weakly identifiable in this interval; "
                    "their approximate uncertainties are reported as NaN.", stacklevel=2
                )
                std_values[free] = np.nan
            else:
                covariance_log = reduced_chi_square * (vt.T / singular_values**2) @ vt
                std_values[free] = fitted_array[free] * np.sqrt(
                    np.maximum(np.diag(covariance_log), 0.0)
                )
        except np.linalg.LinAlgError:
            std_values[free] = np.nan
    parameter_std = dict(zip(PARAMETER_NAMES, map(float, std_values)))

    return FitResult(
        parameters=fitted,
        parameter_std=parameter_std,
        success=success,
        message=message,
        cost=cost,
        reduced_chi_square=reduced_chi_square,
        time_s=np.asarray(time_s),
        measured_temperature=measured_temperature,
        fitted_temperature=predicted,
        inputs_w=np.asarray(inputs_w),
        fixed_parameter_names=tuple(name for name, yes in zip(PARAMETER_NAMES, fixed) if yes),
    )


def fit_experiment_arrays(
    time_relative: Sequence[object],
    bot_flange: Sequence[float],
    top_flange: Sequence[float],
    CH_temp: Sequence[float],
    CH_power: Sequence[float],
    Heater1: Sequence[float],
    Heater2: Sequence[float],
    *,
    fit_dt_s: float | None = 5.0,
    max_points: int = 8000,
    max_gap_s: float | None = None,
    initial: ThermalParameters = DEFAULT_INITIAL,
    lower: ThermalParameters = DEFAULT_LOWER,
    upper: ThermalParameters = DEFAULT_UPPER,
    temperature_sigma_k: Sequence[float] = (0.02, 0.02, 0.02),
    max_nfev: int = 300,
) -> tuple[FitResult, float]:
    """Fit arrays; time_relative accepts original datetime timestamps or seconds.

    Example: fit_experiment_arrays(time_relative=df_selected["Time"], ...).
    Use max_gap_s to set the largest permitted interpolation gap in seconds.
    The calling interface is unchanged. CH_temp supplies the full measured
    boundary; CH_power is retained for preprocessing and diagnostic plots only.
    Fit targets are top_flange/bot_flange; the default free parameters are
    R_tb, C_t, C_b. The two coldhead-link resistances remain fixed even when
    a notebook passes the old wider bounds for them.
    """
    time_s, measured_temperature, inputs_w, actual_dt = prepare_array_data(
        time_relative=time_relative,
        bot_flange=bot_flange,
        top_flange=top_flange,
        CH_temp=CH_temp,
        CH_power=CH_power,
        Heater1=Heater1,
        Heater2=Heater2,
        fit_dt_s=fit_dt_s,
        max_points=max_points,
        max_gap_s=max_gap_s,
    )
    result = fit_thermal_model(
        time_s=time_s,
        measured_temperature=measured_temperature,
        inputs_w=inputs_w,
        initial=initial,
        lower=lower,
        upper=upper,
        temperature_sigma_k=temperature_sigma_k,
        max_nfev=max_nfev,
    )
    return result, actual_dt


def print_fit_summary(result: FitResult) -> None:
    """Print fitted values with units and approximate one-sigma errors."""
    print("Fitted parameters")
    for name, value in asdict(result.parameters).items():
        unit = "K/W" if name.startswith("R_") else "J/K"
        error = result.parameter_std[name]
        if name in result.fixed_parameter_names:
            print(f"  {name:5s} = {value:.8g} {unit} (fixed)")
        else:
            print(f"  {name:5s} = {value:.8g} +/- {error:.3g} {unit}")
    law = result.heat_leak_model
    print(f"  i0(T_top) = {law['a_W_per_K']:.8g} * T_top + {law['b_W']:.8g} W (T_top in K)")
    print("Fit targets         = T_t, T_b; measured T_CH is the boundary input")
    print(f"success             = {result.success}")
    print(f"reduced chi-square  = {result.reduced_chi_square:.6g}")
    print(f"optimizer message   = {result.message}")


def plot_fit(result: FitResult) -> tuple[plt.Figure, np.ndarray]:
    """Plot two fitted temperatures, the measured CH boundary, and powers."""
    labels = ("Top flange", "Bottom flange", "Coldhead")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0, 0.9)},
    )
    time_h = result.time_s / 3600.0
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(
            time_h,
            result.measured_temperature[:, index],
            color="black",
            linewidth=0.8,
            alpha=0.7,
            label="measured boundary (input)" if index == 2 else "measured",
        )
        if index < 2:
            axis.plot(
                time_h,
                result.fitted_temperature[:, index],
                color="tab:red",
                linewidth=1.3,
                label="fitted",
            )
        axis.set_ylabel(f"{label} [K]")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    # inputs_w columns are [I_2 (top heater), I_1 (bottom heater), I_CH].
    axes[3].plot(
        time_h,
        result.inputs_w[:, 1],
        color="tab:blue",
        linewidth=1.0,
        label="Heater1 (bottom)",
    )
    axes[3].plot(
        time_h,
        result.inputs_w[:, 0],
        color="tab:red",
        linewidth=1.0,
        label="Heater2 (top)",
    )
    axes[3].plot(
        time_h,
        result.inputs_w[:, 2],
        color="tab:green",
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
        label="CH power (diagnostic only)",
    )
    axes[3].set_ylabel("Power [W]")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="best", ncol=3)
    axes[-1].set_xlabel("Time from fit start [h]")
    fig.tight_layout()
    return fig, axes


def _temperature_validation_metrics(
    measured: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Return common prediction metrics for one temperature channel."""
    residual = measured - predicted
    centered_sum_squares = float(np.sum((measured - np.mean(measured)) ** 2))
    residual_sum_squares = float(np.sum(residual**2))
    r_squared = (
        1.0 - residual_sum_squares / centered_sum_squares
        if centered_sum_squares > 0
        else float("nan")
    )
    return {
        "RMSE_K": float(np.sqrt(np.mean(residual**2))),
        "MAE_K": float(np.mean(np.abs(residual))),
        "bias_K": float(np.mean(residual)),
        "max_abs_error_K": float(np.max(np.abs(residual))),
        "R_squared": r_squared,
    }


def validate_experiment_arrays(
    parameters: ThermalParameters,
    time_relative: Sequence[object],
    bot_flange: Sequence[float],
    top_flange: Sequence[float],
    CH_temp: Sequence[float],
    CH_power: Sequence[float],
    Heater1: Sequence[float],
    Heater2: Sequence[float],
    *,
    fit_dt_s: float | None = 5.0,
    max_points: int = 8000,
    max_gap_s: float | None = None,
    metric_start_s: float = 0.0,
) -> tuple[ValidationResult, float]:
    """Predict a new interval without changing any fitted parameter.

    time_relative accepts df_validation["Time"] (datetime) or numeric seconds.

    Only the first top/bottom measurements initialize the vessel states.
    Measured CH_temp supplies the boundary throughout the new interval.
    Heater2, Heater1, and i0(predicted T_top) provide the other forcing.
    CH_power is retained for preparation/plots but does not drive the model.
    Parameters are NOT refitted. Metrics are reported for T_t and T_b only.
    The third predicted_temperature column copies the boundary for compatibility;
    the third residual column is NaN (not a perfect zero-error CH prediction).

    metric_start_s can exclude a short initialization period from the reported
    metrics without excluding it from the simulation.
    """
    time_s, measured_temperature, inputs_w, actual_dt = prepare_array_data(
        time_relative=time_relative,
        bot_flange=bot_flange,
        top_flange=top_flange,
        CH_temp=CH_temp,
        CH_power=CH_power,
        Heater1=Heater1,
        Heater2=Heater2,
        fit_dt_s=fit_dt_s,
        max_points=max_points,
        max_gap_s=max_gap_s,
    )
    if metric_start_s < 0 or metric_start_s >= time_s[-1]:
        raise ValueError("metric_start_s must be within the validation interval.")

    predicted_temperature = simulate_temperatures(
        time_s=time_s,
        inputs_w=inputs_w,
        initial_temperature=measured_temperature[0],
        parameters=parameters,
        CH_temp=measured_temperature[:, 2],
    )
    residual_temperature = measured_temperature - predicted_temperature
    residual_temperature[:, 2] = np.nan
    metric_mask = time_s >= metric_start_s
    channel_names = FITTED_CHANNELS
    metrics = {
        name: _temperature_validation_metrics(
            measured_temperature[metric_mask, index],
            predicted_temperature[metric_mask, index],
        )
        for index, name in enumerate(channel_names)
    }

    result = ValidationResult(
        parameters=parameters,
        time_s=time_s,
        measured_temperature=measured_temperature,
        predicted_temperature=predicted_temperature,
        residual_temperature=residual_temperature,
        inputs_w=inputs_w,
        metrics=metrics,
        metric_start_s=float(metric_start_s),
    )
    return result, actual_dt


def print_validation_summary(result: ValidationResult) -> None:
    """Print out-of-sample temperature prediction metrics."""
    print("Validation metrics (measured - predicted)")
    print("T_CH is a measured boundary input; it has no prediction-error metric.")
    print(f"Metrics start at t = {result.metric_start_s:.3f} s")
    for channel, values in result.metrics.items():
        print(f"\n{channel}")
        print(f"  RMSE          = {values['RMSE_K']:.6g} K")
        print(f"  MAE           = {values['MAE_K']:.6g} K")
        print(f"  bias          = {values['bias_K']:.6g} K")
        print(f"  max abs error = {values['max_abs_error_K']:.6g} K")
        print(f"  R^2           = {values['R_squared']:.6g}")


def plot_validation(
    result: ValidationResult,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot vessel validation, measured CH boundary, and applied powers."""
    labels = ("Top flange", "Bottom flange", "Coldhead")
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(14, 11),
        sharex="col",
        gridspec_kw={"width_ratios": (2.2, 1.0)},
    )
    time_h = result.time_s / 3600.0
    for index, label in enumerate(labels):
        axes[index, 0].plot(
            time_h,
            result.measured_temperature[:, index],
            color="black",
            linewidth=0.8,
            alpha=0.7,
            label="measured boundary (input)" if index == 2 else "measured",
        )
        if index < 2:
            axes[index, 0].plot(
                time_h,
                result.predicted_temperature[:, index],
                color="tab:blue",
                linewidth=1.3,
                label="prediction (fixed parameters)",
            )
        axes[index, 0].set_ylabel(f"{label} [K]")
        axes[index, 0].grid(alpha=0.25)
        axes[index, 0].legend(loc="best")

        if index < 2:
            axes[index, 1].axhline(0.0, color="black", linewidth=0.8)
            axes[index, 1].plot(
                time_h,
                result.residual_temperature[:, index],
                color="tab:orange",
                linewidth=0.8,
            )
            axes[index, 1].set_ylabel("Measured - predicted [K]")
            axes[index, 1].grid(alpha=0.25)
        else:
            axes[index, 1].text(
                0.5, 0.5, "Measured CH boundary\nNo CH prediction or residual",
                ha="center", va="center", transform=axes[index, 1].transAxes,
            )
            axes[index, 1].set_axis_off()

    # Display the experimental forcing directly beneath the temperature plots.
    axes[3, 0].plot(
        time_h,
        result.inputs_w[:, 1],
        color="tab:blue",
        linewidth=1.0,
        label="Heater1 (bottom)",
    )
    axes[3, 0].plot(
        time_h,
        result.inputs_w[:, 0],
        color="tab:red",
        linewidth=1.0,
        label="Heater2 (top)",
    )
    axes[3, 0].set_ylabel("Heater power [W]")
    axes[3, 0].grid(alpha=0.25)
    axes[3, 0].legend(loc="best")

    axes[3, 1].plot(
        time_h,
        result.inputs_w[:, 2],
        color="tab:green",
        linewidth=1.0,
        linestyle="--",
        label="CH power (diagnostic only)",
    )
    axes[3, 1].set_ylabel("CH power [W]")
    axes[3, 1].grid(alpha=0.25)
    axes[3, 1].legend(loc="best")

    axes[-1, 0].set_xlabel("Time from validation start [h]")
    axes[-1, 1].set_xlabel("Time from validation start [h]")
    fig.tight_layout()
    return fig, axes


def load_fitted_parameters(path: str | Path) -> ThermalParameters:
    """Load ThermalParameters from fitted_parameters.csv."""
    table = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(table.columns):
        raise ValueError("Parameter CSV must contain 'parameter' and 'value' columns.")
    values = dict(zip(table["parameter"], table["value"]))
    if "i0" in values:
        raise ValueError(
            "This CSV comes from the old constant-i0 model. Refit with the new "
            "temperature-dependent leak law before loading fitted parameters."
        )
    missing = [name for name in PARAMETER_NAMES if name not in values]
    if missing:
        raise ValueError(f"Parameter CSV is missing: {missing}")
    parameters = _parameters_from_array([values[name] for name in PARAMETER_NAMES])
    continuous_state_matrices(parameters)  # Validate signs and the current leak law.
    summary_path = Path(path).with_name("fit_summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("coldhead_mode") != COLDHEAD_MODE:
            raise ValueError(
                "These parameters were saved for a different CH model. "
                "Refit using the measured-CH boundary and fixed copper-link resistances."
            )
        saved_law = summary.get("heat_leak_model")
        if saved_law is not None and saved_law != _heat_leak_metadata():
            warnings.warn(
                "The current i0(T_top) differs from the law used for this fit. "
                "Validation will use the current function; refit for a consistent model.",
                stacklevel=2,
            )
    return parameters


def _saved_heat_leak(temperature: np.ndarray, metadata: Mapping[str, object]) -> np.ndarray:
    """Use the law recorded at fit/validation time, even after a notebook edit."""
    return float(metadata["a_W_per_K"]) * temperature + float(metadata["b_W"])


def save_validation_results(
    result: ValidationResult,
    output_dir: str | Path,
) -> None:
    """Save validation time series, metrics, parameters, and figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series = pd.DataFrame(
        {
            "time_s": result.time_s,
            "T_t_measured": result.measured_temperature[:, 0],
            "T_t_predicted": result.predicted_temperature[:, 0],
            "T_t_residual": result.residual_temperature[:, 0],
            "T_b_measured": result.measured_temperature[:, 1],
            "T_b_predicted": result.predicted_temperature[:, 1],
            "T_b_residual": result.residual_temperature[:, 1],
            "T_CH_measured": result.measured_temperature[:, 2],
            "T_CH_boundary": result.predicted_temperature[:, 2],
            "I_2": result.inputs_w[:, 0],
            "I_1": result.inputs_w[:, 1],
            "I_CH": result.inputs_w[:, 2],
            "i0_model_W": _saved_heat_leak(
                result.predicted_temperature[:, 0], result.heat_leak_model
            ),
            "i0_at_measured_top_W": _saved_heat_leak(
                result.measured_temperature[:, 0], result.heat_leak_model
            ),
        }
    )
    series.to_csv(output_dir / "validation_timeseries.csv", index=False)

    metrics_rows = []
    for channel, values in result.metrics.items():
        metrics_rows.append({"channel": channel, **values})
    pd.DataFrame(metrics_rows).to_csv(
        output_dir / "validation_metrics.csv", index=False
    )

    summary = {
        "metric_start_s": result.metric_start_s,
        "fixed_parameters": asdict(result.parameters),
        "heat_leak_model": result.heat_leak_model,
        "coldhead_mode": result.coldhead_mode,
        "predicted_channels": list(result.fitted_channels),
        "boundary_channel": "T_CH_measured",
        "CH_power_role": "diagnostic_only",
        "metrics": result.metrics,
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, _ = plot_validation(result)
    fig.savefig(output_dir / "validation_plot.png", dpi=180)
    plt.close(fig)


def save_results(result: FitResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_rows = []
    for name, value in asdict(result.parameters).items():
        unit = "K/W" if name.startswith("R_") else "J/K"
        parameter_rows.append(
            {
                "parameter": name,
                "value": value,
                "approx_std": result.parameter_std[name],
                "unit": unit,
                "fixed": name in result.fixed_parameter_names,
            }
        )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "fitted_parameters.csv", index=False
    )

    series = pd.DataFrame(
        {
            "time_s": result.time_s,
            "T_t_measured": result.measured_temperature[:, 0],
            "T_t_fitted": result.fitted_temperature[:, 0],
            "T_b_measured": result.measured_temperature[:, 1],
            "T_b_fitted": result.fitted_temperature[:, 1],
            "T_CH_measured": result.measured_temperature[:, 2],
            "T_CH_boundary": result.fitted_temperature[:, 2],
            "I_2": result.inputs_w[:, 0],
            "I_1": result.inputs_w[:, 1],
            "I_CH": result.inputs_w[:, 2],
            "i0_model_W": _saved_heat_leak(
                result.fitted_temperature[:, 0], result.heat_leak_model
            ),
            "i0_at_measured_top_W": _saved_heat_leak(
                result.measured_temperature[:, 0], result.heat_leak_model
            ),
        }
    )
    series.to_csv(output_dir / "fitted_timeseries.csv", index=False)

    summary = {
        "success": result.success,
        "message": result.message,
        "cost": result.cost,
        "reduced_chi_square": result.reduced_chi_square,
        "parameters": asdict(result.parameters),
        "approx_parameter_std": result.parameter_std,
        "fixed_parameter_names": list(result.fixed_parameter_names),
        "heat_leak_model": result.heat_leak_model,
        "coldhead_mode": result.coldhead_mode,
        "fitted_channels": list(result.fitted_channels),
        "boundary_channel": "T_CH_measured",
        "CH_power_role": "diagnostic_only",
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, _ = plot_fit(result)
    fig.savefig(output_dir / "temperature_fit.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("thermal_fit_results"))
    parser.add_argument("--fit-dt", type=float, default=None, help="Fit time step [s]")
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--time", default="Time")
    parser.add_argument("--T-b", dest="T_b", default="T_b")
    parser.add_argument("--I-1", dest="I_1", default="I_1")
    parser.add_argument("--T-t", dest="T_t", default="T_t")
    parser.add_argument("--I-2", dest="I_2", default="I_2")
    parser.add_argument("--T-CH", dest="T_CH", default="T_CH")
    parser.add_argument("--I-CH", dest="I_CH", default="I_CH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = ColumnNames(
        time=args.time,
        T_b=args.T_b,
        I_1=args.I_1,
        T_t=args.T_t,
        I_2=args.I_2,
        T_CH=args.T_CH,
        I_CH=args.I_CH,
    )
    dataframe = pd.read_csv(args.csv_file)
    time_s, temperatures, inputs_w, actual_dt = prepare_data(
        dataframe,
        columns=columns,
        fit_dt_s=args.fit_dt,
        max_points=args.max_points,
    )
    print(f"Using {len(time_s)} points with fit_dt = {actual_dt:.6g} s")
    result = fit_thermal_model(time_s, temperatures, inputs_w)
    save_results(result, args.output_dir)

    print_fit_summary(result)
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
