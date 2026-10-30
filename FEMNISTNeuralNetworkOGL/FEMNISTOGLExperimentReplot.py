"""Replot completed FEMNIST OGL experiments without running algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

ALGORITHM_NAMES = (
    "REC_FedDyn",
    "FedDR",
    "iFedDR",
    "FedNMap",
    "Zhang",
)
PLOT_LABELS = {"REC_FedDyn": "REC-FedDyn"}
SAMPLING_NUMBERS = (200, 100, 50)
ERROR_FLOOR = 1e-16
FIGURE_DPI = 300

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RESULT_DIRECTORY = SCRIPT_DIRECTORY / "Results" / "FEMNISTOGLNeuralNetwork"

PLOTS = (
    (
        "loss",
        "Training loss",
        True,
        "loss_decay_vs_communication.pdf",
    ),
    (
        "test_accuracy",
        "Test accuracy",
        False,
        "test_accuracy_vs_communication.pdf",
    ),
    (
        "train_accuracy",
        "Training accuracy",
        False,
        "train_accuracy_vs_communication.pdf",
    ),
    (
        "inactive_group_fraction",
        "Inactive group fraction",
        False,
        "inactive_groups_vs_communication.pdf",
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=RESULT_DIRECTORY,
        help="Directory containing sample_*/problem_* result folders.",
    )
    parser.add_argument(
        "--sampling-numbers",
        type=int,
        nargs="+",
        default=list(SAMPLING_NUMBERS),
    )
    return parser.parse_args()


def load_curve(path: Path, field: str) -> tuple[np.ndarray, ...]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    required = (
        "communication",
        f"{field}_mean",
        f"{field}_std",
    )
    missing = [name for name in required if name not in table.dtype.names]
    if missing:
        raise KeyError(f"Missing columns in {path}: {', '.join(missing)}")
    return tuple(np.atleast_1d(table[name]) for name in required)


def shared_vertical_limits(
    problem_directories: list[Path],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for field, _, logarithmic, _ in PLOTS:
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for problem_directory in problem_directories:
            for name in ALGORITHM_NAMES:
                path = problem_directory / f"{name}_mean_std.csv"
                if not path.is_file():
                    continue
                _, mean, standard_deviation = load_curve(path, field)
                lower = mean - standard_deviation
                upper = mean + standard_deviation
                valid = np.isfinite(lower) & np.isfinite(upper)
                if logarithmic:
                    valid &= lower > 0.0
                if np.any(valid):
                    lower_bounds.append(float(np.min(lower[valid])))
                    upper_bounds.append(float(np.max(upper[valid])))

        if not lower_bounds:
            continue
        lower = min(lower_bounds)
        upper = max(upper_bounds)
        if logarithmic:
            log_lower = np.log10(lower)
            log_upper = np.log10(upper)
            padding = 0.03 * max(log_upper - log_lower, 1e-12)
            limits[field] = (
                float(10 ** (log_lower - padding)),
                float(10 ** (log_upper + padding)),
            )
        else:
            padding = 0.03 * max(upper - lower, 1e-12)
            lower_limit = lower - padding
            upper_limit = upper + padding
            if field in {
                "test_accuracy",
                "train_accuracy",
                "inactive_group_fraction",
            }:
                lower_limit = max(0.0, lower_limit)
                upper_limit = min(1.0, upper_limit)
            limits[field] = (lower_limit, upper_limit)
    return limits


def replot(
    problem_directory: Path,
    vertical_limits: dict[str, tuple[float, float]],
) -> bool:
    settings_path = problem_directory / "problem_settings.json"
    if not settings_path.is_file():
        print(f"Skipped {problem_directory}: missing problem_settings.json")
        return False

    paths = {
        name: problem_directory / f"{name}_mean_std.csv"
        for name in ALGORITHM_NAMES
    }
    available = {
        name: path for name, path in paths.items() if path.is_file()
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if not available:
        print(f"Skipped {problem_directory}: no aggregate CSV files")
        return False
    if missing:
        print(
            f"Warning: {problem_directory} is missing "
            f"{', '.join(missing)}; plotting available algorithms only.",
            flush=True,
        )

    with settings_path.open("r", encoding="utf-8") as stream:
        settings = json.load(stream)
    sampled = int(settings["sampling_number"])
    clients = int(settings["clients_number"])

    for field, vertical_label, logarithmic, filename in PLOTS:
        figure = plt.figure(figsize=(8.4, 5.6))
        for name, path in available.items():
            communication, mean, standard_deviation = load_curve(path, field)
            lower = mean - standard_deviation
            upper = mean + standard_deviation
            if logarithmic:
                mean = np.maximum(mean, ERROR_FLOOR)
                lower = np.maximum(lower, ERROR_FLOOR)
                upper = np.maximum(upper, ERROR_FLOOR)
            plot = plt.semilogy if logarithmic else plt.plot
            line = plot(
                communication,
                mean,
                linewidth=LINE_WIDTH,
                linestyle="-" if name == "REC_FedDyn" else "--",
                label=PLOT_LABELS.get(name, name),
            )[0]
            plt.fill_between(
                communication,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.16,
                linewidth=0.0,
            )
        if logarithmic:
            axis = plt.gca()
            plain_formatter = LogFormatter(base=10, labelOnlyBase=False)
            axis.yaxis.set_major_formatter(plain_formatter)
            axis.yaxis.set_minor_formatter(plain_formatter)
        if field in vertical_limits:
            plt.ylim(*vertical_limits[field])
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(
            f"FEMNIST-62 OGL: sampled clients = {sampled}"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        figure.savefig(
            problem_directory / filename,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    print(f"Replotted {problem_directory}", flush=True)
    return True


def replot_results(
    result_directory: Path,
    sampling_numbers: tuple[int, ...] | list[int],
) -> int:
    problem_directories = [
        problem_directory
        for sampling_number in sampling_numbers
        for problem_directory in sorted(
            (result_directory / f"sample_{sampling_number}").glob(
                "problem_*"
            )
        )
    ]
    vertical_limits = shared_vertical_limits(problem_directories)
    replotted = 0
    for problem_directory in problem_directories:
        replotted += int(replot(problem_directory, vertical_limits))
    return replotted


def main() -> None:
    arguments = parse_arguments()
    replotted = replot_results(
        arguments.result_directory,
        arguments.sampling_numbers,
    )

    if replotted == 0:
        raise FileNotFoundError(
            f"No plottable experiment results found in "
            f"{arguments.result_directory}"
        )
    print(f"Replotted {replotted} FEMNIST OGL result directories.")
    print("No optimization algorithm was executed.")


if __name__ == "__main__":
    main()
