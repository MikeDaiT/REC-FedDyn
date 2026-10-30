"""File output and plotting only; no optimization logic lives here."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Algorithms.FMRI import FMRIModel
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

PLOT_LABELS = {"REC_FedDyn": "REC_FedDyn"}


def save_results(
    run_directory: Path,
    settings: dict,
    histories: dict,
    reference_model: FMRIModel,
    reference_value: float,
    reference_information: dict,
    data_directory: Path,
    figure_dpi: int,
    save_plots: bool = True,
) -> None:
    """Save scalar histories, final 3-D models, and the three requested plots."""

    run_directory.mkdir(parents=True, exist_ok=True)

    with (run_directory / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "algorithm",
            "outer_iteration",
            "communication_round",
            "function_value",
            "function_value_error",
            "test_accuracy",
            "gradient_oracles_this_round",
            "gradient_oracles_initialization",
            "gradient_oracles_cumulative",
            "algorithm_time_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, history in histories.items():
            cumulative = history.initialization_oracles + np.cumsum(
                history.oracle_per_round, dtype=np.int64
            )
            for index in range(history.rounds.size):
                writer.writerow(
                    {
                        "algorithm": name,
                        "outer_iteration": int(history.rounds[index]),
                        "communication_round": int(history.communications[index]),
                        "function_value": f"{history.function_values[index]:.16e}",
                        "function_value_error": (
                            f"{history.function_values[index] - reference_value:.16e}"
                        ),
                        "test_accuracy": f"{history.test_accuracy[index]:.16e}",
                        "gradient_oracles_this_round": int(
                            history.oracle_per_round[index]
                        ),
                        "gradient_oracles_initialization": (
                            history.initialization_oracles if index == 0 else 0
                        ),
                        "gradient_oracles_cumulative": int(cumulative[index]),
                        "algorithm_time_seconds": (
                            f"{history.process_time[index]:.16e}"
                        ),
                    }
                )

    arrays = {}
    for name, history in histories.items():
        prefix = name.lower()
        arrays[f"{prefix}_rounds"] = history.rounds
        arrays[f"{prefix}_communications"] = history.communications
        arrays[f"{prefix}_function_values"] = history.function_values
        arrays[f"{prefix}_test_accuracy"] = history.test_accuracy
        arrays[f"{prefix}_oracle_per_round"] = history.oracle_per_round
        arrays[f"{prefix}_process_time"] = history.process_time
        arrays[f"{prefix}_initialization_oracles"] = np.int64(
            history.initialization_oracles
        )
    np.savez_compressed(run_directory / "histories.npz", **arrays)

    model_directory = run_directory / "models"
    model_directory.mkdir(exist_ok=True)
    reference_image = nib.load(data_directory / "model_reference_mni152_3mm.nii.gz")
    models = {"reference": reference_model}
    models.update({
        name.lower(): history.final_model
        for name, history in histories.items()
        if history.final_model is not None
    })
    for name, model in models.items():
        np.savez_compressed(
            model_directory / f"{name}_model.npz",
            omega=model.omega,
            intercept=model.intercept,
        )
        nib.save(
            nib.Nifti1Image(
                np.asarray(model.omega, dtype=np.float32),
                reference_image.affine,
                reference_image.header,
            ),
            model_directory / f"{name}_omega.nii.gz",
        )

    summary = {
        "settings": settings,
        "reference": reference_information,
        "algorithms": {
            name: {
                "completed_rounds": int(history.rounds[-1]),
                "final_function_value": float(history.function_values[-1]),
                "final_test_accuracy": float(history.test_accuracy[-1]),
                "gradient_oracles_in_rounds": int(np.sum(history.oracle_per_round)),
                "gradient_oracles_initialization": history.initialization_oracles,
                "gradient_oracles_total": (
                    history.initialization_oracles
                    + int(np.sum(history.oracle_per_round))
                ),
                "unsatisfied_rounds": history.unsatisfied_rounds,
                "refinements": history.refinements,
                "model_available_at_reported_round": (
                    history.final_model is not None
                ),
                "truncated_from_round": getattr(
                    history, "truncated_from_round", None
                ),
            }
            for name, history in histories.items()
        },
    }
    with (run_directory / "settings_and_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    if not save_plots:
        return

    styles = {
        "REC_FedDyn": ("-", LINE_WIDTH),
        "FedDR": ("--", LINE_WIDTH),
        "iFedDR": ("-.", LINE_WIDTH),
        "FedDyn": (":", LINE_WIDTH),
    }
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for name, history in histories.items():
        style, width = styles[name]
        error = np.maximum(history.function_values - reference_value, floor)
        axes[0].semilogy(
            history.communications, error, style, linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )
        axes[1].plot(
            history.communications,
            history.test_accuracy,
            style,
            linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )
    axes[0].set(xlabel="Communication rounds", ylabel="Function value error")
    axes[1].set(xlabel="Communication rounds", ylabel="Test accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(run_directory / "convergence.png", dpi=figure_dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    for name, history in histories.items():
        style, width = styles[name]
        error = np.maximum(history.function_values - reference_value, floor)
        axis.semilogy(
            history.communications, error, style, linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )
    axis.set(xlabel="Communication rounds", ylabel="Function value error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=False)
    figure.savefig(
        run_directory / "function_error_vs_communication.pdf",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    for name, history in histories.items():
        style, width = styles[name]
        axis.plot(
            history.communications,
            history.test_accuracy,
            style,
            linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )
    axis.set(xlabel="Communication rounds", ylabel="Test accuracy", ylim=(0, 1))
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure.savefig(
        run_directory / "test_accuracy_vs_communication.pdf",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def history_range(histories: list, field: str):
    """Return mean/min/max on the common seed-history prefix."""

    if not histories:
        raise ValueError("At least one Haxby history is required.")
    common_size = min(history.rounds.size for history in histories)
    rounds = np.asarray(histories[0].rounds[:common_size])
    communications = np.asarray(histories[0].communications[:common_size])
    for history in histories[1:]:
        if not np.array_equal(history.rounds[:common_size], rounds):
            raise ValueError("Seed histories do not share a common outer-round prefix.")
        if not np.array_equal(history.communications[:common_size], communications):
            raise ValueError("Seed histories do not share a common communication prefix.")
    values = np.stack([
        np.asarray(getattr(history, field))[:common_size]
        for history in histories
    ])
    return (
        rounds,
        communications,
        np.mean(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
    )


def save_aggregate_results(
    aggregate_directory: Path,
    settings: dict,
    histories: dict[str, list],
    reference_model: FMRIModel,
    reference_value: float,
    reference_information: dict,
    data_directory: Path,
    figure_dpi: int,
) -> None:
    """Save five-seed mean/range curves and pixelwise-mean final models."""

    aggregate_directory.mkdir(parents=True, exist_ok=True)
    fields = (
        "algorithm",
        "outer_iteration",
        "communication_round",
        "function_error_mean",
        "function_error_min",
        "function_error_max",
        "test_accuracy_mean",
        "test_accuracy_min",
        "test_accuracy_max",
    )
    arrays = {}
    summary_algorithms = {}
    with (aggregate_directory / "metrics_mean_range.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, seed_histories in histories.items():
            rounds, communications, value_mean, value_min, value_max = (
                history_range(seed_histories, "function_values")
            )
            _, _, accuracy_mean, accuracy_min, accuracy_max = history_range(
                seed_histories, "test_accuracy"
            )
            error_mean = value_mean - reference_value
            error_min = value_min - reference_value
            error_max = value_max - reference_value
            for index in range(rounds.size):
                writer.writerow({
                    "algorithm": name,
                    "outer_iteration": int(rounds[index]),
                    "communication_round": int(communications[index]),
                    "function_error_mean": f"{error_mean[index]:.16e}",
                    "function_error_min": f"{error_min[index]:.16e}",
                    "function_error_max": f"{error_max[index]:.16e}",
                    "test_accuracy_mean": f"{accuracy_mean[index]:.16e}",
                    "test_accuracy_min": f"{accuracy_min[index]:.16e}",
                    "test_accuracy_max": f"{accuracy_max[index]:.16e}",
                })

            prefix = name.lower()
            arrays[f"{prefix}_rounds"] = rounds
            arrays[f"{prefix}_communications"] = communications
            arrays[f"{prefix}_function_error_mean"] = error_mean
            arrays[f"{prefix}_function_error_min"] = error_min
            arrays[f"{prefix}_function_error_max"] = error_max
            arrays[f"{prefix}_test_accuracy_mean"] = accuracy_mean
            arrays[f"{prefix}_test_accuracy_min"] = accuracy_min
            arrays[f"{prefix}_test_accuracy_max"] = accuracy_max
            same_final_round = all(
                int(history.rounds[-1]) == int(rounds[-1])
                for history in seed_histories
            )
            summary_algorithms[name] = {
                "seed_count": len(seed_histories),
                "final_function_error_mean": float(error_mean[-1]),
                "final_function_error_min": float(error_min[-1]),
                "final_function_error_max": float(error_max[-1]),
                "final_test_accuracy_mean": float(accuracy_mean[-1]),
                "final_test_accuracy_min": float(accuracy_min[-1]),
                "final_test_accuracy_max": float(accuracy_max[-1]),
                "mean_model_available_at_reported_round": all(
                    history.final_model is not None
                    for history in seed_histories
                ) and same_final_round,
            }

    np.savez_compressed(
        aggregate_directory / "histories_mean_range.npz", **arrays
    )

    model_directory = aggregate_directory / "models"
    model_directory.mkdir(exist_ok=True)
    reference_image = nib.load(data_directory / "model_reference_mni152_3mm.nii.gz")
    mean_models = {"reference": reference_model}
    for name, seed_histories in histories.items():
        common_final_round = min(
            int(history.rounds[-1]) for history in seed_histories
        )
        if (
            any(history.final_model is None for history in seed_histories)
            or any(
                int(history.rounds[-1]) != common_final_round
                for history in seed_histories
            )
        ):
            continue
        mean_models[name.lower()] = FMRIModel(
            np.mean(
                np.stack([history.final_model.omega for history in seed_histories]),
                axis=0,
            ),
            np.float64(np.mean([
                history.final_model.intercept for history in seed_histories
            ])),
        )
    for name, model in mean_models.items():
        np.savez_compressed(
            model_directory / f"{name}_mean_model.npz",
            omega=model.omega,
            intercept=model.intercept,
        )
        nib.save(
            nib.Nifti1Image(
                np.asarray(model.omega, dtype=np.float32),
                reference_image.affine,
                reference_image.header,
            ),
            model_directory / f"{name}_mean_omega.nii.gz",
        )

    with (aggregate_directory / "settings_and_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "settings": settings,
                "reference": reference_information,
                "algorithms": summary_algorithms,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )

    styles = {
        "REC_FedDyn": ("-", LINE_WIDTH),
        "FedDR": ("--", LINE_WIDTH),
        "iFedDR": ("-.", LINE_WIDTH),
        "FedDyn": (":", LINE_WIDTH),
    }
    floor = 10.0 * np.finfo(np.float64).eps * max(1.0, abs(reference_value))

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for name, seed_histories in histories.items():
        style, width = styles[name]
        _, communication, value_mean, value_min, value_max = history_range(
            seed_histories, "function_values"
        )
        _, _, accuracy_mean, accuracy_min, accuracy_max = history_range(
            seed_histories, "test_accuracy"
        )
        error_mean = np.maximum(value_mean - reference_value, floor)
        error_min = np.maximum(value_min - reference_value, floor)
        error_max = np.maximum(value_max - reference_value, floor)
        line = axes[0].semilogy(
            communication, error_mean, style, linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )[0]
        axes[0].fill_between(
            communication, error_min, error_max,
            color=line.get_color(), alpha=0.16, linewidth=0.0,
        )
        line = axes[1].plot(
            communication, accuracy_mean, style, linewidth=width,
            label=PLOT_LABELS.get(name, name),
        )[0]
        axes[1].fill_between(
            communication, accuracy_min, accuracy_max,
            color=line.get_color(), alpha=0.16, linewidth=0.0,
        )
    axes[0].set(xlabel="Communication rounds", ylabel="Function value error")
    axes[1].set(xlabel="Communication rounds", ylabel="Test accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(aggregate_directory / "convergence.png", dpi=figure_dpi)
    plt.close(figure)

    for field, ylabel, filename, logarithmic in (
        ("function_values", "Function value error", "function_error_vs_communication.pdf", True),
        ("test_accuracy", "Test accuracy", "test_accuracy_vs_communication.pdf", False),
    ):
        figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
        for name, seed_histories in histories.items():
            style, width = styles[name]
            _, communication, mean, minimum, maximum = history_range(
                seed_histories, field
            )
            if field == "function_values":
                mean = np.maximum(mean - reference_value, floor)
                minimum = np.maximum(minimum - reference_value, floor)
                maximum = np.maximum(maximum - reference_value, floor)
            plot = axis.semilogy if logarithmic else axis.plot
            line = plot(
                communication, mean, style, linewidth=width,
                label=PLOT_LABELS.get(name, name),
            )[0]
            axis.fill_between(
                communication, minimum, maximum,
                color=line.get_color(), alpha=0.16, linewidth=0.0,
            )
        axis.set(xlabel="Communication rounds", ylabel=ylabel)
        if field == "test_accuracy":
            axis.set_ylim(0, 1)
        axis.grid(True, which="both" if logarithmic else "major", alpha=0.3)
        axis.legend(frameon=False)
        figure.savefig(
            aggregate_directory / filename,
            dpi=figure_dpi,
            bbox_inches="tight",
        )
        plt.close(figure)
