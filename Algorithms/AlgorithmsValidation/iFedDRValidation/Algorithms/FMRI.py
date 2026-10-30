"""Shared native-volume state and result types for the Haxby fMRI experiment.

The distributed optimizers keep the coefficient volume in its original 3-D
shape and keep the unregularized logistic intercept as a separate scalar.
This module deliberately contains no experiment data-loading or plotting code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from Algorithms.Base_Optimizer import Float, FloatArray, ProxFunction


@dataclass(slots=True)
class FMRIModel:
    """A native three-dimensional coefficient volume and scalar intercept."""

    omega: FloatArray
    intercept: Float

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "FMRIModel":
        return cls(np.zeros(shape, dtype=np.float64), Float(0.0))

    def copy(self) -> "FMRIModel":
        return FMRIModel(
            np.asarray(self.omega, dtype=np.float64).copy(),
            Float(self.intercept),
        )

    def __add__(self, other: "FMRIModel") -> "FMRIModel":
        return FMRIModel(
            self.omega + other.omega,
            Float(self.intercept + other.intercept),
        )

    def __sub__(self, other: "FMRIModel") -> "FMRIModel":
        return FMRIModel(
            self.omega - other.omega,
            Float(self.intercept - other.intercept),
        )

    def __mul__(self, scalar: float) -> "FMRIModel":
        return FMRIModel(
            self.omega * scalar,
            Float(self.intercept * scalar),
        )

    def __rmul__(self, scalar: float) -> "FMRIModel":
        return self * scalar

    def __truediv__(self, scalar: float) -> "FMRIModel":
        return self * (1.0 / scalar)


FMRIObjectiveFunction: TypeAlias = Callable[
    [FMRIModel], tuple[Float, FMRIModel]
]
FMRIValueFunction: TypeAlias = Callable[[FMRIModel], Float]
FMRIAccuracyFunction: TypeAlias = Callable[[FMRIModel], Float]


@dataclass(slots=True)
class FMRIAlgorithmResults:
    """Scalar histories and final model produced by a fMRI-specific run."""

    oracle_calling: np.ndarray
    process_time: FloatArray
    function_value: FloatArray
    test_accuracy: FloatArray
    final_point: FMRIModel
    index: np.ndarray
    refinement_count: np.ndarray
    communication_rounds: np.ndarray


def model_sum(models: list[FMRIModel], shape: tuple[int, ...]) -> FMRIModel:
    total = FMRIModel.zeros(shape)
    for model in models:
        total.omega += model.omega
        total.intercept += model.intercept
    return total


def model_mean(models: list[FMRIModel], shape: tuple[int, ...]) -> FMRIModel:
    return model_sum(models, shape) / len(models)


def squared_norm(model: FMRIModel) -> Float:
    return Float(np.sum(model.omega**2) + model.intercept**2)


def model_norm(model: FMRIModel) -> Float:
    return Float(np.sqrt(squared_norm(model)))


def model_dot(first: FMRIModel, second: FMRIModel) -> Float:
    return Float(
        np.sum(first.omega * second.omega)
        + first.intercept * second.intercept
    )


def proximal_model(
    model: FMRIModel,
    proximal_function: ProxFunction,
    parameter: Float,
    constant: Float,
) -> FMRIModel:
    """Apply the regularizer only to omega and preserve the intercept."""

    omega = proximal_function(model.omega, parameter, constant, False)
    return FMRIModel(
        np.asarray(omega, dtype=np.float64),
        Float(model.intercept),
    )


def record_statistics(
    model: FMRIModel,
    value_function: FMRIValueFunction,
    accuracy_function: FMRIAccuracyFunction,
    proximal_function: ProxFunction,
    constant: Float,
) -> tuple[Float, Float]:
    smooth_value = value_function(model)
    regularizer_value = proximal_function(
        model.omega, Float(1.0), constant, True
    )
    return (
        Float(smooth_value + regularizer_value),
        Float(accuracy_function(model)),
    )
