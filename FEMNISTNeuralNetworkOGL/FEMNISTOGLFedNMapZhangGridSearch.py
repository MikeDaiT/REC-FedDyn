"""Two-dimensional stepsize grid search for FEMNIST FedNMap and Zhang."""

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
from Algorithms.FedNMap import FedNMap
from Algorithms.TorchUtils import statistics as torch_statistics
from Algorithms.Zhang import Zhang
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style
from FEMNISTOGLStepsizeTest import CandidateHistory


apply_experiment_plot_style()

# ======================================================================
# Grid-search settings -- edit parameters in this section
# ======================================================================

SERVER_LEARNING_RATES = (0.1, 0.5, 1.0, 1.5, 2.0)
GAMMAS = (0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
ALGORITHM_NAMES = ("FedNMap", "Zhang")

CLIENTS_NUMBER = 200
SAMPLING_NUMBER = 200
ITERATION_BUDGET = 300
LOCAL_UPDATES = 30

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
    / "FEMNISTOGLFedNMapZhangGridSearch"
    / "sample_200"
    / "local_updates_30"
)


def algorithm_settings(
    name: str, gamma: float, server_learning_rate: float
):
    local_learning_rate = gamma / LOCAL_UPDATES
    if name == "FedNMap":
        return FedNMap, dict(
            ProximalParameter=gamma * server_learning_rate,
            LocalLearningRate=local_learning_rate,
            ServerLearningRate=server_learning_rate,
            LocalUpdates=LOCAL_UPDATES,
        )
    if name == "Zhang":
        return Zhang, dict(
            LocalLearningRate=local_learning_rate,
            ServerLearningRate=server_learning_rate,
            LocalUpdates=LOCAL_UPDATES,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def stationarity_parameter(
    name: str, gamma: float, server_learning_rate: float
) -> float:
    if name in ALGORITHM_NAMES:
        return gamma * server_learning_rate
    raise ValueError(f"Unknown algorithm: {name}")


def candidate_settings(
    name: str,
    gamma: float,
    server_learning_rate: float,
    problem_key: str,
) -> dict:
    _, parameters = algorithm_settings(name, gamma, server_learning_rate)
    return {
        "format": "femnist_ogl_fednmap_zhang_grid_candidate_v2",
        "problem_key": problem_key,
        "algorithm": name,
        "gamma": gamma,
        "server_learning_rate": server_learning_rate,
        "local_learning_rate": gamma / LOCAL_UPDATES,
        "stationarity_parameter": stationarity_parameter(
            name, gamma, server_learning_rate
        ),
        "algorithm_parameters": parameters,
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "client_partitioning": "natural_writer_id",
        "iteration_budget": ITERATION_BUDGET,
        "local_updates": LOCAL_UPDATES,
        "ogl_regularization": OGL_REGULARIZATION,
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "algorithm_seed": ALGORITHM_SEED,
    }


def candidate_stem(gamma: float, server_learning_rate: float) -> str:
    return f"server_{server_learning_rate:g}_gamma_{gamma:g}"


def run_one(
    name: str,
    gamma: float,
    server_learning_rate: float,
    initial_point: torch.Tensor,
    functions: list,
    model: torch.nn.Module,
    unpack,
    clients: list[femnist.ClientData],
) -> CandidateHistory:
    prox = femnist.make_ogl_prox(model)
    mapping_parameter = stationarity_parameter(
        name, gamma, server_learning_rate
    )
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
        name, gamma, server_learning_rate
    )
    initial_loss, initial_mapping = torch_statistics(
        initial_point,
        functions,
        prox,
        OGL_REGULARIZATION,
        mapping_parameter,
    )
    result = algorithm_class(**common).go_torch(
        Function=functions,
        RecordStatistics=True,
        StationarityParameter=mapping_parameter,
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
    )


def load_cached_history(
    gamma: float,
    server_learning_rate: float,
    directory: Path,
    settings: dict,
) -> CandidateHistory | None:
    stem = candidate_stem(gamma, server_learning_rate)
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
    return CandidateHistory(
        communication=table[:, 0].astype(np.int64),
        gradient_mapping=table[:, 1],
        loss=table[:, 2],
        test_accuracy=table[:, 3],
        cumulative_oracles=table[:, 4].astype(np.int64),
        process_time=table[:, 5],
        inactive_group_fraction=table[:, 6],
    )


def save_history(
    gamma: float,
    server_learning_rate: float,
    history: CandidateHistory,
    directory: Path,
    settings: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = candidate_stem(gamma, server_learning_rate)
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
    with (directory / f"{stem}_settings.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)


def failure_is_cached(
    gamma: float,
    server_learning_rate: float,
    directory: Path,
    settings: dict,
) -> bool:
    path = directory / f"{candidate_stem(gamma, server_learning_rate)}_failed.json"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload.get("settings") == settings


def save_divergence(
    gamma: float,
    server_learning_rate: float,
    directory: Path,
    settings: dict,
    error: FloatingPointError,
) -> None:
    path = directory / f"{candidate_stem(gamma, server_learning_rate)}_failed.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"settings": settings, "status": "diverged", "error": str(error)},
            stream,
            indent=2,
            ensure_ascii=False,
        )


def save_server_rate_plots(
    name: str,
    server_learning_rate: float,
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
        for gamma, history in histories.items():
            values = getattr(history, field)
            if logarithmic:
                values = np.maximum(values, ERROR_FLOOR)
                plt.semilogy(
                    history.communication, values,
                    linewidth=LINE_WIDTH, label=f"gamma={gamma:g}",
                )
            else:
                plt.plot(
                    history.communication, values,
                    linewidth=LINE_WIDTH, label=f"gamma={gamma:g}",
                )
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(
            f"{name}: server LR={server_learning_rate:g}, participation=200/200"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        figure.savefig(
            directory / f"server_{server_learning_rate:g}_{suffix}.png",
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
    values = np.full(
        (len(SERVER_LEARNING_RATES), len(GAMMAS)), np.nan, dtype=np.float64
    )
    server_index = {value: index for index, value in enumerate(SERVER_LEARNING_RATES)}
    gamma_index = {value: index for index, value in enumerate(GAMMAS)}
    for row in rows:
        values[server_index[row[1]], gamma_index[row[2]]] = row[value_index]

    figure, axis = plt.subplots(figsize=(10.0, 5.8))
    image = axis.imshow(values, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(GAMMAS)), [f"{value:g}" for value in GAMMAS])
    axis.set_yticks(
        range(len(SERVER_LEARNING_RATES)),
        [f"{value:g}" for value in SERVER_LEARNING_RATES],
    )
    axis.set_xlabel("gamma")
    axis.set_ylabel("Server learning rate")
    axis.set_title(f"{name}: final {label}")
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    figure.savefig(directory / filename, dpi=FIGURE_DPI)
    plt.close(figure)


def save_algorithm_summary(
    name: str, rows: list[tuple], directory: Path
) -> None:
    header = (
        "algorithm", "server_learning_rate", "gamma", "local_learning_rate",
        "final_gradient_mapping_norm", "final_loss", "final_test_accuracy",
        "inactive_group_fraction", "cumulative_gradient_oracles",
        "algorithm_time_seconds", "status",
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
    for value_index, label, filename in (
        (4, "gradient mapping norm", "final_gradient_mapping_heatmap.png"),
        (5, "regularized train loss", "final_loss_heatmap.png"),
        (6, "test accuracy", "final_test_accuracy_heatmap.png"),
        (7, "inactive group fraction", "final_inactive_fraction_heatmap.png"),
    ):
        save_heatmap(
            name, rows, value_index, label, filename, directory
        )


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

    for server_learning_rate in SERVER_LEARNING_RATES:
        server_histories: dict[float, CandidateHistory] = {}
        for gamma in GAMMAS:
            local_learning_rate = gamma / LOCAL_UPDATES
            settings = candidate_settings(
                name, gamma, server_learning_rate, problem_key
            )
            print(
                f"Running {name}: server_lr={server_learning_rate:g}, "
                f"gamma={gamma:g}, local_lr={local_learning_rate:g}...",
                flush=True,
            )
            try:
                history = None if arguments.restart else load_cached_history(
                    gamma, server_learning_rate, directory, settings
                )
                cached_failure = (
                    False if arguments.restart else failure_is_cached(
                        gamma, server_learning_rate, directory, settings
                    )
                )
                if history is not None:
                    print("  loaded completed candidate", flush=True)
                elif cached_failure:
                    print("  loaded divergence record", flush=True)
                    rows.append((
                        name, server_learning_rate, gamma, local_learning_rate,
                        np.nan, np.nan, np.nan, np.nan, 0, 0.0, "diverged",
                    ))
                    save_algorithm_summary(name, rows, directory)
                    continue
                else:
                    history = run_one(
                        name,
                        gamma,
                        server_learning_rate,
                        initial_point,
                        functions,
                        model,
                        unpack,
                        clients,
                    )
                    save_history(
                        gamma, server_learning_rate, history, directory, settings
                    )

                server_histories[gamma] = history
                finite = np.isfinite((
                    history.gradient_mapping[-1], history.loss[-1],
                    history.test_accuracy[-1],
                    history.inactive_group_fraction[-1],
                )).all()
                status = "completed" if finite else "diverged"
                rows.append((
                    name,
                    server_learning_rate,
                    gamma,
                    local_learning_rate,
                    history.gradient_mapping[-1],
                    history.loss[-1],
                    history.test_accuracy[-1],
                    history.inactive_group_fraction[-1],
                    history.cumulative_oracles[-1],
                    history.process_time[-1],
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
                    gamma, server_learning_rate, directory, settings, error
                )
                rows.append((
                    name, server_learning_rate, gamma, local_learning_rate,
                    np.nan, np.nan, np.nan, np.nan, 0, 0.0, "diverged",
                ))
                save_algorithm_summary(name, rows, directory)
                print(f"  diverged: {error}", flush=True)
            except Exception:
                traceback.print_exc()
                print("  Failed; continuing...", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if server_histories:
            save_server_rate_plots(
                name,
                server_learning_rate,
                server_histories,
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

    print(
        f"Device={device}; algorithms={arguments.algorithms}; "
        f"clients={CLIENTS_NUMBER}; participation={SAMPLING_NUMBER}; "
        f"seed={ALGORITHM_SEED}; rounds={ITERATION_BUDGET}; "
        f"grid={len(SERVER_LEARNING_RATES)}x{len(GAMMAS)}",
        flush=True,
    )
    print(
        "Natural writer-ID partition; "
        "local learning rate = gamma / local updates.",
        flush=True,
    )
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
