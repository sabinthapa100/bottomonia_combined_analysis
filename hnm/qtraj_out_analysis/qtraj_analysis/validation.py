"""
Validation utilities for qtraj_analysis.

Provides reusable validation functions for arrays and physics constraints.
"""
import numpy as np
from typing import Optional, Tuple, Union

from qtraj_analysis.exceptions import (
    ArrayShapeError,
    ArrayValueError,
    ConfigurationError,
    PhysicsConstraintError,
)
from qtraj_analysis.observable_registry import get_observable_spec, parse_mathematica_x_values


def validate_array_shape(
    arr: np.ndarray,
    name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_ndim: Optional[int] = None,
    max_ndim: Optional[int] = None,
) -> None:
    """Validate array shape constraints."""
    if min_ndim is not None and arr.ndim < min_ndim:
        raise ArrayShapeError(name, f"ndim >= {min_ndim}", f"ndim = {arr.ndim}")
    
    if max_ndim is not None and arr.ndim > max_ndim:
        raise ArrayShapeError(name, f"ndim <= {max_ndim}", f"ndim = {arr.ndim}")
    
    if expected_shape is not None:
        if len(expected_shape) != arr.ndim:
            raise ArrayShapeError(name, expected_shape, arr.shape)
        
        for dim_idx, (exp, act) in enumerate(zip(expected_shape, arr.shape)):
            if exp != -1 and exp != act:
                raise ArrayShapeError(f"{name}[dim={dim_idx}]", expected_shape, arr.shape)


def validate_array_values(
    arr: np.ndarray,
    name: str,
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> None:
    """Validate array values are within acceptable ranges."""
    # Check for NaN
    if not allow_nan and np.any(np.isnan(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        raise ArrayValueError(
            f"Array '{name}' contains {nan_count} NaN values",
            context={"array_name": name, "nan_count": nan_count, "total_elements": arr.size}
        )
    
    # Check for Inf
    if not allow_inf and np.any(np.isinf(arr)):
        inf_count = int(np.sum(np.isinf(arr)))
        raise ArrayValueError(
            f"Array '{name}' contains {inf_count} Inf values",
            context={"array_name": name, "inf_count": inf_count, "total_elements": arr.size}
        )
    
    # Finite values only for min/max checks
    finite_arr = arr[np.isfinite(arr)] if (allow_nan or allow_inf) else arr
    
    # Check minimum value
    if min_value is not None and len(finite_arr) > 0:
        below_min_count = int(np.sum(finite_arr < min_value))
        if below_min_count > 0:
            actual_min = float(np.min(finite_arr))
            raise ArrayValueError(
                f"Array '{name}' has {below_min_count} values below minimum {min_value}",
                context={"array_name": name, "min_value": min_value, "actual_min": actual_min}
            )
    
    # Check maximum value
    if max_value is not None and len(finite_arr) > 0:
        above_max_count = int(np.sum(finite_arr > max_value))
        if above_max_count > 0:
            actual_max = float(np.max(finite_arr))
            raise ArrayValueError(
                f"Array '{name}' has {above_max_count} values above maximum {max_value}",
                context={"array_name": name, "max_value": max_value, "actual_max": actual_max}
            )


def validate_survival_probability(arr: np.ndarray, name: str = "survival") -> None:
    """Validate that array contains valid survival probabilities in [0, 1]."""
    validate_array_values(arr, name, allow_nan=False, allow_inf=False, min_value=0.0, max_value=1.0)


def validate_impact_parameter(b: Union[float, np.ndarray], name: str = "b", max_b: float = 30.0) -> None:
    """Validate impact parameter is physically reasonable."""
    arr = np.atleast_1d(b)
    try:
        validate_array_values(arr, name, allow_nan=False, allow_inf=False, min_value=0.0, max_value=max_b)
    except ArrayValueError as e:
        raise PhysicsConstraintError(
            f"Impact parameter '{name}' out of physical range [0, {max_b}] fm",
            context=e.context
        ) from e


def validate_matched_lengths(*arrays: Tuple[str, np.ndarray]) -> None:
    """Validate that multiple arrays have matching lengths."""
    if len(arrays) < 2:
        return
    
    names = [name for name, _ in arrays]
    lengths = [len(arr) for _, arr in arrays]
    
    if len(set(lengths)) > 1:
        mismatch_info = {name: length for name, length in zip(names, lengths)}
        raise ArrayShapeError("array_lengths", f"all equal (expected {lengths[0]})", mismatch_info)


def validate_registry_grid_matches_mathematica(
    observable_id: str,
    *,
    atol: float = 1e-12,
) -> None:
    """
    Validate that the registry x-grid for an observable exactly matches the
    x-values exported in all attached Mathematica source files.
    """
    spec = get_observable_spec(observable_id)
    expected = np.asarray(spec.grid.values, dtype=np.float64)
    for source in spec.mathematica_sources:
        actual = np.asarray(parse_mathematica_x_values(source), dtype=np.float64)
        if expected.shape != actual.shape:
            raise ConfigurationError(
                f"Registry grid length mismatch for '{observable_id}'",
                context={
                    "observable_id": observable_id,
                    "source": source.path,
                    "expected_length": int(expected.shape[0]),
                    "actual_length": int(actual.shape[0]),
                    "expected": expected.tolist(),
                    "actual": actual.tolist(),
                },
            )
        if not np.allclose(expected, actual, atol=atol, rtol=0.0):
            raise ConfigurationError(
                f"Registry grid values do not match Mathematica for '{observable_id}'",
                context={
                    "observable_id": observable_id,
                    "source": source.path,
                    "expected": expected.tolist(),
                    "actual": actual.tolist(),
                    "max_abs_diff": float(np.max(np.abs(expected - actual))),
                },
            )
