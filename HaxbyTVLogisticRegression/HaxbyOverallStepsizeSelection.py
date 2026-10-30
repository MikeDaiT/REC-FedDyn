"""Select a direct overall step size for the Haxby fMRI experiment.

Each candidate ``eta`` is used directly by FedDR and iFedDR.  REC_FedDyn and
FedDyn use the equivalent penalty ``1 / eta``.  This script does not modify or
reuse the older factor-based tuning results.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

import HaxbyLogisticRegressionExperiment as experiment
from HaxbyStepsizeSelection import normalized_log_error_auc


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

# ============================================================================
# Tuning settings
# ============================================================================

ALGORITHMS = experiment.EXPERIMENT_ALGORITHMS
OVERALL_STEP_SIZES = (0.5, 1.0, 3.0, 5.0, 8.0, 10.0)
TUNING_ITERATION_BUDGET = 100
TUNING_PROGRESS_INTERVAL = 100
TUNING_ALGORITHM_SEED = 123
RESULT_DIRECTORY = experiment.RESULT_DIRECTORY / "overall_step_size_selection"


def number_label(value: float) -> str:
    """Return a stable directory-name component for a positive number."""

    return f"{value:g}".replace(".", "p")


def load_smoothness() -> float:
    """Read the same local smoothness bound used by the formal experiment."""

    if not experiment.SMOOTHNESS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing smoothness cache: {experiment.SMOOTHNESS_FILE}"
        )
    with np.load(experiment.SMOOTHNESS_FILE) as cached:
        return float(np.max(cached["local_bounds"]))


@contextmanager
def candidate_configuration(step_size: float, smoothness: float):
    """Temporarily configure one independently checkpointed candidate."""

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
    # The formal runner forms eta = STEP_SIZE_FACTOR / L.  Multiplying the
    # requested eta by this problem's L therefore passes eta through exactly.
    experiment.STEP_SIZE_FACTOR = float(step_size * smoothness)
    experiment.ALGORITHM_SEEDS = (TUNING_ALGORITHM_SEED,)
    experiment.RESULT_DIRECTORY = (
        RESULT_DIRECTORY / f"stepsize_{number_label(step_size)}"
    )
    experiment.CONTINUE_PREVIOUS_RUN = True
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(experiment, name, value)


def candidate_row(
    step_size: float,
    algorithm: str,
    history: experiment.History,
    reference_value: float,
) -> dict[str, object]:
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))
    error = np.maximum(history.function_values - reference_value, floor)
    increases = np.diff(history.function_values)
    return {
        "overall_step_size": step_size,
        "penalty": 1.0 / step_size,
        "algorithm": algorithm,
        "normalized_log_error_auc": normalized_log_error_auc(
            history, reference_value
        ),
        "final_function_error": float(error[-1]),
        "relative_final_error": float(error[-1] / error[0]),
        "final_test_accuracy": float(history.test_accuracy[-1]),
        "rebound_rounds": int(np.count_nonzero(increases > 0.0)),
        "max_objective_rebound": float(
            max(0.0, np.max(increases)) if increases.size else 0.0
        ),
        "gradient_oracles_total": int(
            history.initialization_oracles + np.sum(history.oracle_per_round)
        ),
        "all_finite": bool(
            np.all(np.isfinite(history.function_values))
            and np.all(np.isfinite(history.test_accuracy))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    histories: dict[str, list[tuple[float, experiment.History]]],
    reference_value: float,
    rows: list[dict[str, object]],
) -> None:
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(OVERALL_STEP_SIZES))
    )
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))

    for algorithm, candidates in histories.items():
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        for color, (step_size, history) in zip(colors, candidates):
            error = np.maximum(history.function_values - reference_value, floor)
            axes[0].semilogy(
                history.communications,
                error / error[0],
                color=color,
                linewidth=LINE_WIDTH,
                label=f"{step_size:g}",
            )
            axes[1].plot(
                history.communications,
                history.test_accuracy,
                color=color,
                linewidth=LINE_WIDTH,
                label=f"{step_size:g}",
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
            title="Overall step size",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
        )
        figure.tight_layout()
        stem = f"{algorithm.lower()}_overall_step_size_curves"
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
            [row["overall_step_size"] for row in algorithm_rows],
            [row["normalized_log_error_auc"] for row in algorithm_rows],
            marker="o",
            linewidth=LINE_WIDTH,
            label=algorithm,
        )
    axis.set(
        xlabel="Overall step size",
        ylabel="Normalized log-error area (lower is better)",
    )
    axis.grid(True, alpha=0.28)
    axis.legend(frameon=False)
    figure.savefig(
        RESULT_DIRECTORY / "overall_step_size_score_summary.png",
        dpi=experiment.FIGURE_DPI,
        bbox_inches="tight",
    )
    figure.savefig(
        RESULT_DIRECTORY / "overall_step_size_score_summary.pdf",
        dpi=experiment.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    smoothness = load_smoothness()
    reference_value = None
    rows: list[dict[str, object]] = []
    histories = {algorithm: [] for algorithm in ALGORITHMS}

    for step_size in OVERALL_STEP_SIZES:
        print("=" * 78)
        print(
            f"Haxby overall step size {step_size:g}, "
            f"equivalent penalty {1.0 / step_size:g}"
        )
        with candidate_configuration(step_size, smoothness):
            (
                candidate_histories,
                settings,
                _,
                current_reference_value,
                _,
            ) = experiment.run_algorithm_seed(TUNING_ALGORITHM_SEED)

        actual_step_size = float(
            settings["algorithm_settings"]["iFedDR"]["step_size"]
        )
        if not np.isclose(actual_step_size, step_size):
            raise ValueError(
                f"Requested step size {step_size}, got {actual_step_size}."
            )
        if reference_value is None:
            reference_value = current_reference_value
        elif not np.isclose(reference_value, current_reference_value):
            raise ValueError("Candidates did not use the same reference value.")

        for algorithm in ALGORITHMS:
            history = candidate_histories[algorithm]
            histories[algorithm].append((step_size, history))
            rows.append(candidate_row(
                step_size, algorithm, history, current_reference_value
            ))

    if reference_value is None:
        raise RuntimeError("No overall-step-size candidates were evaluated.")

    write_csv(RESULT_DIRECTORY / "tuning_per_algorithm.csv", rows)
    selected = []
    for algorithm in ALGORITHMS:
        candidates = [
            row for row in rows
            if row["algorithm"] == algorithm and row["all_finite"]
        ]
        if candidates:
            selected.append(min(
                candidates, key=lambda row: row["normalized_log_error_auc"]
            ))
    write_csv(RESULT_DIRECTORY / "selected_step_sizes.csv", selected)

    summary_rows = []
    for step_size in OVERALL_STEP_SIZES:
        candidates = [
            row for row in rows if row["overall_step_size"] == step_size
        ]
        summary_rows.append({
            "overall_step_size": step_size,
            "penalty": 1.0 / step_size,
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
                "format": "haxby_overall_step_size_selection_v1",
                "overall_step_sizes": list(OVERALL_STEP_SIZES),
                "parameter_mapping": {
                    "FedDR": "ProximalParameter = overall_step_size",
                    "iFedDR": "ProximalParameter = overall_step_size",
                    "REC_FedDyn": "PenaltyParameter = 1 / overall_step_size",
                    "FedDyn": "PenaltyParameter = 1 / overall_step_size",
                },
                "iteration_budget": TUNING_ITERATION_BUDGET,
                "local_updates": experiment.LOCAL_UPDATES,
                "algorithm_seed": TUNING_ALGORITHM_SEED,
                "smoothness": smoothness,
                "reference_value": reference_value,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )

    save_plots(histories, reference_value, rows)
    print(f"Overall-step-size selection saved to {RESULT_DIRECTORY}")


if __name__ == "__main__":
    main()
