"""Two-dimensional FEMNIST OGL grid search for FedDR and iFedDR."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import FEMNISTOGLNeuralNetworkExperiment as femnist
from Algorithms.FedDR import FedDR
from Algorithms.TorchUtils import statistics as torch_statistics
from Algorithms.iFedDR import iFedDR
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style
from FEMNISTOGLStepsizeTest import CandidateHistory


apply_experiment_plot_style()

# ======================================================================
# Grid-search settings -- edit parameters in this section
# ======================================================================

FEDDR_PROXIMAL_PARAMETERS = (3.0, 4.0, 5.0, 6.0, 7.0)
IFEDDR_PROXIMAL_PARAMETERS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
RELAXATION_PARAMETERS = (0.1, 0.5, 1.0, 1.5, 1.98)
ALGORITHM_NAMES = ("FedDR", "iFedDR")

CLIENTS_NUMBER = 200
SAMPLING_NUMBER = 200
ITERATION_BUDGET = 300
LOCAL_UPDATES = 30
LOCAL_LEARNING_RATE = 1e-2

IFEDDR_SIGMA_SQUARED = femnist.IFEDDR_SIGMA_SQUARED
IFEDDR_REFINEMENT_FACTOR = femnist.IFEDDR_REFINEMENT_FACTOR
IFEDDR_MAX_REFINEMENTS = femnist.IFEDDR_MAX_REFINEMENTS
IFEDDR_MAX_LOCAL_ITERATIONS = femnist.IFEDDR_MAX_LOCAL_ITERATIONS
OGL_REGULARIZATION = femnist.OGL_REGULARIZATION

DATA_SEED = femnist.DATA_SEED
ALGORITHM_SEED = femnist.ALGORITHM_SEEDS[0]
MODEL_SEED = femnist.MODEL_SEED + ALGORITHM_SEED
FIGURE_DPI = femnist.FIGURE_DPI
ERROR_FLOOR = femnist.ERROR_FLOOR

DATA_DIRECTORY = Path("")  # Fill in the directory containing the FEMNIST data.
RESULT_DIRECTORY = (
    femnist.SCRIPT_DIRECTORY
    / "Results"
    / "FEMNISTOGLFedDRiFedDRGridSearch"
    / "sample_200"
    / "local_updates_30"
)


def proximal_parameters(name: str) -> tuple[float, ...]:
    if name == "FedDR":
        return FEDDR_PROXIMAL_PARAMETERS
    if name == "iFedDR":
        return IFEDDR_PROXIMAL_PARAMETERS
    raise ValueError(f"Unknown algorithm: {name}")


def algorithm_settings(
    name: str, proximal_parameter: float, relaxation_parameter: float
):
    if name == "FedDR":
        return FedDR, dict(
            ProximalParameter=proximal_parameter,
            RelaxationParameter=relaxation_parameter,
            LocalUpdates=LOCAL_UPDATES,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "iFedDR":
        return iFedDR, dict(
            ProximalParameter=proximal_parameter,
            RelaxationParameter=relaxation_parameter,
            SigmaSquared=IFEDDR_SIGMA_SQUARED,
            InitialLocalUpdates=LOCAL_UPDATES,
            RefinementFactor=IFEDDR_REFINEMENT_FACTOR,
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            MaxLocalIterations=IFEDDR_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def candidate_settings(
    name: str,
    proximal_parameter: float,
    relaxation_parameter: float,
    problem_key: str,
) -> dict:
    _, parameters = algorithm_settings(
        name, proximal_parameter, relaxation_parameter
    )
    return {
        "format": "femnist_ogl_feddr_ifeddr_grid_candidate_v1",
        "problem_key": problem_key,
        "algorithm": name,
        "proximal_parameter": proximal_parameter,
        "relaxation_parameter": relaxation_parameter,
        "stationarity_parameter": proximal_parameter,
        "algorithm_parameters": parameters,
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "client_partitioning": "natural_writer_id",
        "iteration_budget": ITERATION_BUDGET,
        "local_updates": LOCAL_UPDATES,
        "local_learning_rate": LOCAL_LEARNING_RATE,
        "ogl_regularization": OGL_REGULARIZATION,
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "algorithm_seed": ALGORITHM_SEED,
    }


def candidate_stem(
    proximal_parameter: float, relaxation_parameter: float
) -> str:
    return f"relax_{relaxation_parameter:g}_prox_{proximal_parameter:g}"


def run_one(
    name: str,
    proximal_parameter: float,
    relaxation_parameter: float,
    initial_point: torch.Tensor,
    functions: list,
    model: torch.nn.Module,
    unpack,
    clients: list[femnist.ClientData],
) -> CandidateHistory:
    prox = femnist.make_ogl_prox(model)
    common = dict(
        InitialPoint=initial_point,
        IterationBudget=ITERATION_BUDGET,
        ProxFunction=prox,
        ClientsNumber=CLIENTS_NUMBER,
        SamplingNumber=SAMPLING_NUMBER,
        SmoothL=0.0,
        ConvexMu=0.0,
        Constant=OGL_REGULARIZATION,
        RandomSeed=ALGORITHM_SEED,
    )
    algorithm_class, settings = algorithm_settings(
        name, proximal_parameter, relaxation_parameter
    )
    initial_loss, initial_mapping = torch_statistics(
        initial_point,
        functions,
        prox,
        OGL_REGULARIZATION,
        proximal_parameter,
    )
    result = algorithm_class(**common).go_torch(
        Function=functions,
        RecordStatistics=True,
        StationarityParameter=proximal_parameter,
        **settings,
    )

    points = torch.cat((initial_point[None], result.iterate_point))
    finite = torch.isfinite(points).all(dim=1).detach().cpu().numpy()
    test_accuracy = np.full(ITERATION_BUDGET + 1, np.nan, dtype=np.float64)
    inactive = np.full(ITERATION_BUDGET + 1, np.nan, dtype=np.float64)
    for index in np.flatnonzero(finite):
        _, test_accuracy[index] = femnist._split_metrics(
            model,
            unpack,
            points[index],
            clients,
            "test",
            femnist.EVALUATION_BATCH_SIZE,
        )
        inactive[index] = prox.inactive_group_fraction(
            points[index], femnist.INACTIVE_GROUP_TOLERANCE
        )

    return CandidateHistory(
        communication=np.arange(ITERATION_BUDGET + 1),
        gradient_mapping=np.concatenate((
            [float(initial_mapping)], result.subgradient.detach().cpu().numpy()
        )),
        loss=np.concatenate((
            [float(initial_loss)], result.function_value.detach().cpu().numpy()
        )),
        test_accuracy=test_accuracy,
        cumulative_oracles=np.concatenate((
            [0], np.cumsum(result.oracle_calling.detach().cpu().numpy())
        )),
        process_time=np.concatenate((
            [0.0], result.process_time.detach().cpu().numpy()
        )),
        inactive_group_fraction=inactive,
        refinement_count=(
            result.refinement_count.detach().cpu().numpy()
            if name == "iFedDR" else None
        ),
        relative_error_satisfied=(
            np.logical_not(result.index.detach().cpu().numpy())
            if name == "iFedDR" else None
        ),
    )


def load_cached_history(
    name: str,
    proximal_parameter: float,
    relaxation_parameter: float,
    directory: Path,
    settings: dict,
) -> CandidateHistory | None:
    stem = candidate_stem(proximal_parameter, relaxation_parameter)
    settings_path = directory / f"{stem}_settings.json"
    metrics_path = directory / f"{stem}.csv"
    if not settings_path.is_file() or not metrics_path.is_file():
        return None
    with settings_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != settings:
            return None
    table = np.loadtxt(metrics_path, delimiter=",", skiprows=1, ndmin=2)
    if len(table) != ITERATION_BUDGET + 1:
        return None
    if int(table[-1, 0]) != ITERATION_BUDGET:
        return None

    refinement_count = None
    relative_error_satisfied = None
    if name == "iFedDR":
        refinement_path = directory / f"{stem}_refinements.csv"
        if not refinement_path.is_file():
            return None
        refinement_table = np.loadtxt(
            refinement_path, delimiter=",", skiprows=1, ndmin=2
        )
        if len(refinement_table) != ITERATION_BUDGET:
            return None
        refinement_count = refinement_table[:, 1].astype(np.int32)
        relative_error_satisfied = refinement_table[:, 2].astype(bool)

    return CandidateHistory(
        communication=table[:, 0].astype(np.int64),
        gradient_mapping=table[:, 1],
        loss=table[:, 2],
        test_accuracy=table[:, 3],
        cumulative_oracles=table[:, 4].astype(np.int64),
        process_time=table[:, 5],
        inactive_group_fraction=table[:, 6],
        refinement_count=refinement_count,
        relative_error_satisfied=relative_error_satisfied,
    )


def save_history(
    name: str,
    proximal_parameter: float,
    relaxation_parameter: float,
    history: CandidateHistory,
    directory: Path,
    settings: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = candidate_stem(proximal_parameter, relaxation_parameter)
    np.savetxt(
        directory / f"{stem}.csv",
        np.column_stack((
            history.communication,
            history.gradient_mapping,
            history.loss,
            history.test_accuracy,
            history.cumulative_oracles,
            history.process_time,
            history.inactive_group_fraction,
        )),
        delimiter=",",
        fmt=("%d", "%.10e", "%.10e", "%.10e", "%d", "%.10e", "%.10e"),
        header=(
            "communication,gradient_mapping_norm,regularized_train_loss,"
            "test_accuracy,cumulative_gradient_oracles,algorithm_time_seconds,"
            "inactive_group_fraction"
        ),
        comments="",
    )
    if name == "iFedDR":
        if (
            history.refinement_count is None
            or history.relative_error_satisfied is None
        ):
            raise ValueError("iFedDR history is missing refinement statistics.")
        np.savetxt(
            directory / f"{stem}_refinements.csv",
            np.column_stack((
                np.arange(1, ITERATION_BUDGET + 1),
                history.refinement_count,
                history.relative_error_satisfied.astype(np.int8),
            )),
            delimiter=",",
            fmt="%d",
            header=(
                "communication,refinement_count,"
                "relative_error_condition_satisfied"
            ),
            comments="",
        )
    with (directory / f"{stem}_settings.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)


def failure_is_cached(
    proximal_parameter: float,
    relaxation_parameter: float,
    directory: Path,
    settings: dict,
) -> bool:
    path = directory / (
        f"{candidate_stem(proximal_parameter, relaxation_parameter)}_failed.json"
    )
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload.get("settings") == settings


def save_divergence(
    proximal_parameter: float,
    relaxation_parameter: float,
    directory: Path,
    settings: dict,
    error: FloatingPointError,
) -> None:
    path = directory / (
        f"{candidate_stem(proximal_parameter, relaxation_parameter)}_failed.json"
    )
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"settings": settings, "status": "diverged", "error": str(error)},
            stream,
            indent=2,
            ensure_ascii=False,
        )


def save_relaxation_plots(
    name: str,
    relaxation_parameter: float,
    histories: dict[float, CandidateHistory],
    directory: Path,
) -> None:
    plots = (
        ("gradient_mapping", "Gradient mapping norm", True, "gradient_mapping"),
        ("loss", "Regularized training loss", True, "loss"),
        ("test_accuracy", "Test accuracy", False, "test_accuracy"),
        ("inactive_group_fraction", "Inactive group fraction", False, "inactive"),
    )
    for field, vertical_label, logarithmic, suffix in plots:
        figure = plt.figure(figsize=(8.4, 5.6))
        for proximal_parameter, history in histories.items():
            values = getattr(history, field)
            if logarithmic:
                values = np.maximum(values, ERROR_FLOOR)
                plt.semilogy(
                    history.communication, values,
                    linewidth=LINE_WIDTH, label=f"prox={proximal_parameter:g}",
                )
            else:
                plt.plot(
                    history.communication, values,
                    linewidth=LINE_WIDTH, label=f"prox={proximal_parameter:g}",
                )
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(
            f"{name}: relaxation={relaxation_parameter:g}, participation=200/200"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        figure.savefig(
            directory / f"relax_{relaxation_parameter:g}_{suffix}.png",
            dpi=FIGURE_DPI,
        )
        plt.close(figure)

    if name == "iFedDR":
        figure = plt.figure(figsize=(8.4, 5.6))
        for proximal_parameter, history in histories.items():
            if history.refinement_count is None:
                continue
            plt.plot(
                history.communication[1:],
                history.refinement_count,
                linewidth=LINE_WIDTH,
                label=f"prox={proximal_parameter:g}",
            )
        plt.xlabel("Communication rounds")
        plt.ylabel("Refinement count")
        plt.title(
            f"iFedDR: relaxation={relaxation_parameter:g}, participation=200/200"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        figure.savefig(
            directory / f"relax_{relaxation_parameter:g}_refinements.png",
            dpi=FIGURE_DPI,
        )
        plt.close(figure)


def save_heatmap(
    name: str,
    rows: list[tuple],
    value_index: int,
    label: str,
    filename: str,
    directory: Path,
) -> None:
    proximal_grid = proximal_parameters(name)
    values = np.full(
        (len(RELAXATION_PARAMETERS), len(proximal_grid)),
        np.nan,
        dtype=np.float64,
    )
    relaxation_index = {
        value: index for index, value in enumerate(RELAXATION_PARAMETERS)
    }
    proximal_index = {value: index for index, value in enumerate(proximal_grid)}
    for row in rows:
        values[relaxation_index[row[1]], proximal_index[row[2]]] = row[value_index]

    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    image = axis.imshow(values, aspect="auto", interpolation="nearest")
    axis.set_xticks(
        range(len(proximal_grid)),
        [f"{value:g}" for value in proximal_grid],
    )
    axis.set_yticks(
        range(len(RELAXATION_PARAMETERS)),
        [f"{value:g}" for value in RELAXATION_PARAMETERS],
    )
    axis.set_xlabel("Proximal parameter")
    axis.set_ylabel("Relaxation parameter")
    axis.set_title(f"{name}: final {label}")
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    figure.savefig(directory / filename, dpi=FIGURE_DPI)
    plt.close(figure)


def save_algorithm_summary(
    name: str, rows: list[tuple], directory: Path
) -> None:
    header = (
        "algorithm", "relaxation_parameter", "proximal_parameter",
        "local_learning_rate", "final_gradient_mapping_norm", "final_loss",
        "final_test_accuracy", "inactive_group_fraction",
        "cumulative_gradient_oracles", "algorithm_time_seconds",
        "total_refinements", "relative_error_success_fraction", "status",
    )
    with (directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    finite_rows = [
        row for row in rows
        if row[-1] == "completed"
        and np.isfinite((row[4], row[5], row[6], row[7])).all()
    ]
    with (directory / "selected_hyperparameters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        if finite_rows:
            writer.writerow(min(finite_rows, key=lambda row: row[4]))


def save_algorithm_heatmaps(
    name: str, rows: list[tuple], directory: Path
) -> None:
    plots = [
        (4, "gradient mapping norm", "final_gradient_mapping_heatmap.png"),
        (5, "regularized train loss", "final_loss_heatmap.png"),
        (6, "test accuracy", "final_test_accuracy_heatmap.png"),
        (7, "inactive group fraction", "final_inactive_fraction_heatmap.png"),
    ]
    if name == "iFedDR":
        plots.extend((
            (10, "total refinements", "total_refinements_heatmap.png"),
            (11, "relative-error success fraction", "relative_success_heatmap.png"),
        ))
    for value_index, label, filename in plots:
        save_heatmap(name, rows, value_index, label, filename, directory)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=list(ALGORITHM_NAMES),
    )
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--result-directory", type=Path, default=RESULT_DIRECTORY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed and divergence cache files.",
    )
    return parser.parse_args()


def run_algorithm_grid(
    name: str,
    arguments: argparse.Namespace,
    problem_key: str,
    initial_point: torch.Tensor,
    functions: list,
    model: torch.nn.Module,
    unpack,
    clients: list[femnist.ClientData],
) -> None:
    directory = arguments.result_directory / name
    directory.mkdir(parents=True, exist_ok=True)
    femnist.save_json(
        directory / "problem_settings.json",
        femnist.problem_settings(clients, SAMPLING_NUMBER),
    )
    femnist.save_clients(clients, directory / "clients.csv")
    rows: list[tuple] = []

    for relaxation_parameter in RELAXATION_PARAMETERS:
        histories: dict[float, CandidateHistory] = {}
        for proximal_parameter in proximal_parameters(name):
            settings = candidate_settings(
                name, proximal_parameter, relaxation_parameter, problem_key
            )
            print(
                f"Running {name}: prox={proximal_parameter:g}, "
                f"relaxation={relaxation_parameter:g}...",
                flush=True,
            )
            try:
                history = None if arguments.restart else load_cached_history(
                    name,
                    proximal_parameter,
                    relaxation_parameter,
                    directory,
                    settings,
                )
                cached_failure = (
                    False if arguments.restart else failure_is_cached(
                        proximal_parameter,
                        relaxation_parameter,
                        directory,
                        settings,
                    )
                )
                if history is not None:
                    print("  loaded completed candidate", flush=True)
                elif cached_failure:
                    print("  loaded divergence record", flush=True)
                    rows.append((
                        name, relaxation_parameter, proximal_parameter,
                        LOCAL_LEARNING_RATE, np.nan, np.nan, np.nan, np.nan,
                        0, 0.0, np.nan, np.nan, "diverged",
                    ))
                    save_algorithm_summary(name, rows, directory)
                    continue
                else:
                    history = run_one(
                        name,
                        proximal_parameter,
                        relaxation_parameter,
                        initial_point,
                        functions,
                        model,
                        unpack,
                        clients,
                    )
                    save_history(
                        name,
                        proximal_parameter,
                        relaxation_parameter,
                        history,
                        directory,
                        settings,
                    )

                histories[proximal_parameter] = history
                finite = np.isfinite((
                    history.gradient_mapping[-1], history.loss[-1],
                    history.test_accuracy[-1],
                    history.inactive_group_fraction[-1],
                )).all()
                total_refinements = (
                    int(history.refinement_count.sum())
                    if history.refinement_count is not None else np.nan
                )
                relative_success = (
                    float(history.relative_error_satisfied.mean())
                    if history.relative_error_satisfied is not None else np.nan
                )
                status = "completed" if finite else "diverged"
                rows.append((
                    name,
                    relaxation_parameter,
                    proximal_parameter,
                    LOCAL_LEARNING_RATE,
                    history.gradient_mapping[-1],
                    history.loss[-1],
                    history.test_accuracy[-1],
                    history.inactive_group_fraction[-1],
                    history.cumulative_oracles[-1],
                    history.process_time[-1],
                    total_refinements,
                    relative_success,
                    status,
                ))
                save_algorithm_summary(name, rows, directory)
                print(
                    f"  mapping={history.gradient_mapping[-1]:.3e}; "
                    f"loss={history.loss[-1]:.4f}; "
                    f"accuracy={history.test_accuracy[-1]:.3f}",
                    flush=True,
                )
            except FloatingPointError as error:
                save_divergence(
                    proximal_parameter,
                    relaxation_parameter,
                    directory,
                    settings,
                    error,
                )
                rows.append((
                    name, relaxation_parameter, proximal_parameter,
                    LOCAL_LEARNING_RATE, np.nan, np.nan, np.nan, np.nan,
                    0, 0.0, np.nan, np.nan, "diverged",
                ))
                save_algorithm_summary(name, rows, directory)
                print(f"  diverged: {error}", flush=True)
            except Exception:
                traceback.print_exc()
                print("  Failed; continuing...", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if histories:
            save_relaxation_plots(
                name,
                relaxation_parameter,
                histories,
                directory,
            )
    save_algorithm_heatmaps(name, rows, directory)


def main() -> None:
    arguments = parse_arguments()
    device = femnist.resolve_device(arguments.device)
    clients = femnist.load_femnist_clients(
        arguments.data_directory,
        clients_number=CLIENTS_NUMBER,
        test_fraction=femnist.TEST_FRACTION,
        data_seed=DATA_SEED,
        minimum_writer_samples=femnist.MINIMUM_WRITER_SAMPLES,
    )
    model = femnist.FEMNISTNet().to(device)
    femnist.initialize_model(model, MODEL_SEED)
    unpack = femnist.parameter_unpacker(model)
    initial_point = parameters_to_vector(model.parameters()).detach()
    functions = [
        femnist.make_scaled_client_objective(model, unpack, client)
        for client in clients
    ]
    problem = femnist.problem_settings(clients, SAMPLING_NUMBER)
    problem_key = femnist.settings_key(problem)

    candidate_counts = {
        name: len(RELAXATION_PARAMETERS) * len(proximal_parameters(name))
        for name in arguments.algorithms
    }
    print(
        f"Device={device}; algorithms={arguments.algorithms}; "
        f"candidates={candidate_counts}; clients={CLIENTS_NUMBER}; "
        f"participation={SAMPLING_NUMBER}; seed={ALGORITHM_SEED}; "
        f"rounds={ITERATION_BUDGET}; local_updates={LOCAL_UPDATES}",
        flush=True,
    )
    print("Natural writer-ID partition; no synthetic heterogeneity.", flush=True)
    for name in arguments.algorithms:
        run_algorithm_grid(
            name,
            arguments,
            problem_key,
            initial_point,
            functions,
            model,
            unpack,
            clients,
        )
    print(f"Saved results to {arguments.result_directory}", flush=True)


if __name__ == "__main__":
    main()
