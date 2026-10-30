"""Rerun Haxby step-size selection with the corrected iFedDR condition.

The tuning results use their own directory, so legacy tuning files cannot be
mistaken for resumable checkpoints. Interrupted candidates resume from their
candidate-specific checkpoints.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

import HaxbyLogisticRegressionExperiment as experiment


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

ALGORITHMS = experiment.EXPERIMENT_ALGORITHMS
STEP_SIZE_FACTORS = (
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    80.0,
    100.0,
    120.0,
    150.0,
    200.0,
    250.0,
    300.0,
)
TUNING_ITERATION_BUDGET = 100
TUNING_PROGRESS_INTERVAL = 100
TUNING_ALGORITHM_SEED = 123
RESULT_DIRECTORY = experiment.RESULT_DIRECTORY / "step_size_selection"


def factor_label(factor: float) -> str:
    """Return a stable path component for one positive factor."""

    return f"{factor:g}".replace(".", "p")


@contextmanager
def candidate_configuration(factor: float):
    """Temporarily configure the formal runner as one isolated tuning job."""

    names = (
        "ITERATION_BUDGET",
        "PROGRESS_INTERVAL",
        "STEP_SIZE_FACTOR",
        "ALGORITHM_SEEDS",
        "RESULT_DIRECTORY",
        "CONTINUE_PREVIOUS_RUN",
    )
    previous = {name: getattr(experiment, name) for name in names}
    experiment.ITERATION_BUDGET = TUNING_ITERATION_BUDGET
    experiment.PROGRESS_INTERVAL = min(
        TUNING_PROGRESS_INTERVAL, TUNING_ITERATION_BUDGET
    )
    experiment.STEP_SIZE_FACTOR = float(factor)
    experiment.ALGORITHM_SEEDS = (TUNING_ALGORITHM_SEED,)
    experiment.RESULT_DIRECTORY = RESULT_DIRECTORY / f"factor_{factor_label(factor)}"
    experiment.CONTINUE_PREVIOUS_RUN = True
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(experiment, name, value)


def normalized_log_error_auc(
    history: experiment.History,
    reference_value: float,
) -> float:
    """Communication-normalized log10 relative function-error area."""

    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))
    error = np.maximum(history.function_values - reference_value, floor)
    relative_error = error / error[0]
    communication = np.asarray(history.communications, dtype=np.float64)
    span = float(communication[-1] - communication[0])
    if span <= 0.0:
        return float(np.log10(relative_error[-1]))
    return float(np.trapezoid(np.log10(relative_error), communication) / span)


def candidate_row(
    factor: float,
    step_size: float,
    algorithm: str,
    history: experiment.History,
    reference_value: float,
) -> dict[str, object]:
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))
    error = np.maximum(history.function_values - reference_value, floor)
    increases = np.diff(history.function_values)
    return {
        "factor": factor,
        "step_size": step_size,
        "algorithm": algorithm,
        "normalized_log_error_auc": normalized_log_error_auc(
            history, reference_value
        ),
        "final_function_error": float(error[-1]),
        "relative_final_error": float(error[-1] / error[0]),
        "final_test_accuracy": float(history.test_accuracy[-1]),
        "rebound_rounds": int(np.count_nonzero(increases > 0.0)),
        "max_objective_rebound": float(max(0.0, np.max(increases))),
        "gradient_oracles_total": int(
            history.initialization_oracles + np.sum(history.oracle_per_round)
        ),
        "all_finite": bool(
            np.all(np.isfinite(history.function_values))
            and np.all(np.isfinite(history.test_accuracy))
        ),
    }


def unstable_algorithms(
    candidate_histories: dict[str, experiment.History],
) -> tuple[str, ...]:
    """Return algorithms whose final objective no longer improves on round 0."""

    unstable = []
    for algorithm, history in candidate_histories.items():
        values = np.asarray(history.function_values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or values[-1] > values[0]:
            unstable.append(algorithm)
    return tuple(unstable)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows were supplied for {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_step_size_plots(
    histories: dict[str, list[tuple[float, experiment.History]]],
    reference_value: float,
    rows: list[dict[str, object]],
) -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(STEP_SIZE_FACTORS))
    )
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))

    for algorithm, candidates in histories.items():
        figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
        for color, (factor, history) in zip(colors, candidates):
            error = np.maximum(history.function_values - reference_value, floor)
            axes[0].semilogy(
                history.communications,
                error / error[0],
                color=color,
                linewidth=LINE_WIDTH,
                label=f"{factor:g}",
            )
            axes[1].plot(
                history.communications,
                history.test_accuracy,
                color=color,
                linewidth=LINE_WIDTH,
                label=f"{factor:g}",
            )
        axes[0].set(
            xlabel="Communication rounds",
            ylabel="Relative function-value error",
            title=f"{algorithm}: objective",
        )
        axes[1].set(
            xlabel="Communication rounds",
            ylabel="Test accuracy",
            ylim=(0.0, 1.0),
            title=f"{algorithm}: accuracy",
        )
        for axis in axes:
            axis.grid(True, which="both", alpha=0.28)
        axes[1].legend(
            title="Step-size factor",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
            ncol=2,
        )
        figure.tight_layout()
        stem = f"{algorithm.lower()}_step_size_curves"
        figure.savefig(
            RESULT_DIRECTORY / f"{stem}.png",
            dpi=experiment.FIGURE_DPI,
            bbox_inches="tight",
        )
        figure.savefig(
            RESULT_DIRECTORY / f"{stem}.pdf",
            dpi=experiment.FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for algorithm in ALGORITHMS:
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        axis.plot(
            [row["factor"] for row in algorithm_rows],
            [row["normalized_log_error_auc"] for row in algorithm_rows],
            marker="o",
            markersize=3.5,
            linewidth=LINE_WIDTH,
            label=algorithm,
        )
    axis.set_xscale("log")
    axis.set(
        xlabel="Step-size factor",
        ylabel="Normalized log-error area (lower is better)",
    )
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(frameon=False)
    figure.savefig(
        RESULT_DIRECTORY / "step_size_score_summary.png",
        dpi=experiment.FIGURE_DPI,
        bbox_inches="tight",
    )
    figure.savefig(
        RESULT_DIRECTORY / "step_size_score_summary.pdf",
        dpi=experiment.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    histories: dict[str, list[tuple[float, experiment.History]]] = {
        algorithm: [] for algorithm in ALGORITHMS
    }
    rows: list[dict[str, object]] = []
    evaluated_factors: list[float] = []
    reference_value = None
    smoothness = None

    for factor in STEP_SIZE_FACTORS:
        print("=" * 78)
        print(f"Haxby step-size factor {factor:g}")
        with candidate_configuration(factor):
            (
                candidate_histories,
                settings,
                _,
                current_reference_value,
                _,
            ) = experiment.run_algorithm_seed(TUNING_ALGORITHM_SEED)
        current_step_size = float(
            settings["algorithm_settings"]["iFedDR"]["step_size"]
        )
        current_smoothness = factor / current_step_size
        if reference_value is None:
            reference_value = current_reference_value
            smoothness = current_smoothness
        elif not np.isclose(reference_value, current_reference_value):
            raise ValueError("Candidates did not use the same reference value.")

        for algorithm in ALGORITHMS:
            history = candidate_histories[algorithm]
            histories[algorithm].append((factor, history))
            rows.append(candidate_row(
                factor,
                current_step_size,
                algorithm,
                history,
                current_reference_value,
            ))

        evaluated_factors.append(factor)
        unstable = unstable_algorithms(candidate_histories)
        if unstable:
            print(
                "Stopping larger step sizes after factor "
                f"{factor:g}: final objective did not improve for "
                f"{', '.join(unstable)}.",
                flush=True,
            )
            break

    if reference_value is None or smoothness is None:
        raise RuntimeError("No step-size candidates were evaluated.")

    write_csv(RESULT_DIRECTORY / "tuning_per_algorithm.csv", rows)
    selected = []
    for algorithm in ALGORITHMS:
        candidates = [row for row in rows if row["algorithm"] == algorithm]
        best = min(candidates, key=lambda row: row["normalized_log_error_auc"])
        selected.append(dict(best))
    write_csv(RESULT_DIRECTORY / "selected_step_sizes.csv", selected)

    summary_rows = []
    for factor in evaluated_factors:
        candidates = [row for row in rows if row["factor"] == factor]
        summary_rows.append({
            "factor": factor,
            "step_size": candidates[0]["step_size"],
            "mean_normalized_log_error_auc": float(np.mean([
                row["normalized_log_error_auc"] for row in candidates
            ])),
            "worst_normalized_log_error_auc": float(np.max([
                row["normalized_log_error_auc"] for row in candidates
            ])),
            "all_finite": all(row["all_finite"] for row in candidates),
        })
    write_csv(RESULT_DIRECTORY / "tuning_summary.csv", summary_rows)

    with (RESULT_DIRECTORY / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "format": "haxby_step_size_selection_v1",
                "corrected_ifeddr_implementation": (
                    experiment.ALGORITHM_IMPLEMENTATIONS["iFedDR"]
                ),
                "step_size_factors": evaluated_factors,
                "iteration_budget": TUNING_ITERATION_BUDGET,
                "algorithm_seed": TUNING_ALGORITHM_SEED,
                "smoothness": smoothness,
                "step_size_definition": "factor / smoothness",
                "reference_value": reference_value,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )

    save_step_size_plots(histories, reference_value, rows)
    print(f"Corrected step-size selection saved to {RESULT_DIRECTORY}")


if __name__ == "__main__":
    main()
