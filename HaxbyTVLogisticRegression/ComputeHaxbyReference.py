"""Compute or resume the centralized Haxby reference point.

The formal experiment never runs this file automatically.  The existing
70,000-iteration cache is already accepted by the current tolerance, so users
normally only need to copy the cache and run HaxbyLogisticRegressionExperiment.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from Algorithms.FMRI import FMRIModel, model_norm
from Algorithms.ProximalOperator import TV
from Data.Haxby.load_haxby import load_haxby
from HaxbyLogisticRegressionExperiment import (
    DATA_DIRECTORY,
    REFERENCE_FILE,
    TV_REGULARIZATION,
    full_objective,
    make_objective,
    make_value_function,
)


# Reference-only settings.
REFERENCE_MAX_ITERATIONS = 70_000          # cumulative; increase to continue
REFERENCE_TOLERANCE = 2.12e-3
REFERENCE_TV_ACCURACY = 1e-8
REFERENCE_TV_MAX_ITERATIONS = 1_000
CHECK_INTERVAL = 50
CHECKPOINT_INTERVAL = 250
POWER_ITERATIONS = 250
POWER_TOLERANCE = 1e-8
SMOOTHNESS_SAFETY_FACTOR = 1.01
REFERENCE_PROBLEM_KEY = (
    "1caca2ff63cafb0f6fe6c50060101b859bed80206b7265d8a68bef807bff8160"
)


def reference_prox(model: FMRIModel, step_size: float) -> FMRIModel:
    omega = TV(
        model.omega,
        np.float64(step_size),
        np.float64(TV_REGULARIZATION),
        False,
        model.omega.shape,
        accuracy=np.float64(REFERENCE_TV_ACCURACY),
        maximumIteration=REFERENCE_TV_MAX_ITERATIONS,
    )
    return FMRIModel(np.asarray(omega, dtype=np.float64), model.intercept)


def global_smoothness(images: np.ndarray) -> float:
    """Power iteration for ||[X,1]||^2/(4n)."""

    rng = np.random.default_rng(2026)
    model = FMRIModel(
        rng.normal(size=images.shape[1:]), np.float64(rng.normal())
    )
    model = model / model_norm(model)
    previous = 0.0

    for _ in range(POWER_ITERATIONS):
        scores = np.asarray(
            np.tensordot(
                images,
                np.asarray(model.omega, dtype=images.dtype),
                axes=((1, 2, 3), (0, 1, 2)),
            ),
            dtype=np.float64,
        ) + model.intercept
        image_part = np.tensordot(
            np.asarray(scores, dtype=images.dtype), images, axes=(0, 0)
        )
        product = FMRIModel(
            np.asarray(image_part, dtype=np.float64), np.float64(np.sum(scores))
        )
        norm = float(model_norm(product))
        model = product / norm
        eigenvalue = float(
            np.sum(model.omega * product.omega)
            + model.intercept * product.intercept
        )
        if abs(eigenvalue - previous) <= POWER_TOLERANCE * max(1.0, eigenvalue):
            break
        previous = eigenvalue

    return SMOOTHNESS_SAFETY_FACTOR * eigenvalue / (4.0 * images.shape[0])


def residual(
    model: FMRIModel,
    objective,
    step_size: float,
) -> float:
    _, gradient = objective(model)
    next_model = reference_prox(model - step_size * gradient, step_size)
    return float(model_norm(model - next_model) / step_size)


def save_reference(
    model: FMRIModel,
    objective_value: float,
    iterations: int,
    current_residual: float,
    smoothness: float,
    step_size: float,
    problem_description: str,
) -> None:
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        REFERENCE_FILE,
        cache_format_version=np.int64(2),
        problem_seed=np.int64(2026),
        problem_key=np.asarray(REFERENCE_PROBLEM_KEY),
        problem_description=np.asarray(problem_description),
        omega=model.omega,
        intercept=model.intercept,
        objective_value=np.float64(objective_value),
        iterations=np.int64(iterations),
        residual=np.float64(current_residual),
        converged=np.bool_(current_residual <= REFERENCE_TOLERANCE),
        residual_tolerance=np.float64(REFERENCE_TOLERANCE),
        global_smoothness_bound=np.float64(smoothness),
        fixed_step_size=np.float64(step_size),
        tv_proximal_accuracy=np.float64(REFERENCE_TV_ACCURACY),
        tv_proximal_max_iterations=np.int64(REFERENCE_TV_MAX_ITERATIONS),
        updated_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
    )


def main() -> None:
    data = load_haxby(DATA_DIRECTORY, mmap_mode="r")
    objective = make_objective(data["train_X"], data["train_y"])
    value_function = make_value_function(data["train_X"], data["train_y"])
    volume_shape = tuple(data["train_X"].shape[1:])

    if REFERENCE_FILE.is_file():
        with np.load(REFERENCE_FILE) as cached:
            if str(cached["problem_key"].item()) != REFERENCE_PROBLEM_KEY:
                raise ValueError("The existing cache belongs to another problem.")
            current = FMRIModel(
                np.asarray(cached["omega"], dtype=np.float64),
                np.float64(cached["intercept"]),
            )
            completed = int(cached["iterations"])
            current_residual = float(cached["residual"])
            smoothness = float(cached["global_smoothness_bound"])
            step_size = float(cached["fixed_step_size"])
            problem_description = str(cached["problem_description"].item())
    else:
        current = FMRIModel.zeros(volume_shape)
        completed = 0
        smoothness = global_smoothness(data["train_X"])
        step_size = 1.0 / smoothness
        current_residual = residual(current, objective, step_size)
        problem_description = (
            "Fixed processed Haxby face-versus-house TV-logistic problem"
        )

    print(
        f"reference iteration={completed}, residual={current_residual:.6e}, "
        f"tolerance={REFERENCE_TOLERANCE:.6e}"
    )
    if current_residual <= REFERENCE_TOLERANCE:
        print("The cached reference is accepted; no iteration was performed.")
        return
    if completed >= REFERENCE_MAX_ITERATIONS:
        print("Increase REFERENCE_MAX_ITERATIONS to continue this checkpoint.")
        return

    # Restart FISTA momentum from the cached point; the point itself is resumed.
    accelerated = current.copy()
    momentum = 1.0
    for iteration in range(completed + 1, REFERENCE_MAX_ITERATIONS + 1):
        _, gradient = objective(accelerated)
        next_model = reference_prox(
            accelerated - step_size * gradient, step_size
        )
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = next_model + ((momentum - 1.0) / next_momentum) * (
            next_model - current
        )
        current = next_model
        momentum = next_momentum

        checked = iteration % CHECK_INTERVAL == 0
        if checked:
            current_residual = residual(current, objective, step_size)
            print(f"iteration={iteration}, residual={current_residual:.6e}")

        if iteration % CHECKPOINT_INTERVAL == 0 or (
            checked and current_residual <= REFERENCE_TOLERANCE
        ):
            save_reference(
                current,
                full_objective(current, value_function),
                iteration,
                current_residual,
                smoothness,
                step_size,
                problem_description,
            )
        if checked and current_residual <= REFERENCE_TOLERANCE:
            break


if __name__ == "__main__":
    main()
