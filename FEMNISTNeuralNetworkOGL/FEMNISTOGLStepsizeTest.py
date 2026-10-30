"""Overall-stepsize sweep for the FEMNIST-62 OGL neural experiment."""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import FEMNISTOGLNeuralNetworkExperiment as femnist
from Algorithms.REC_FedDyn import REC_FedDyn
from Algorithms.FedDR import FedDR
from Algorithms.FedNMap import FedNMap
from Algorithms.TorchUtils import statistics as torch_statistics
from Algorithms.Zhang import Zhang
from Algorithms.iFedDR import iFedDR
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

# ======================================================================
# Stepsize-test settings -- edit parameters in this section
# ======================================================================

OVERALL_STEPSIZES = (0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0)
ALGORITHM_NAMES = ("REC_FedDyn", "FedDR", "iFedDR", "FedNMap", "Zhang")

CLIENTS_NUMBER = 200
SAMPLING_NUMBER = 200
ITERATION_BUDGET = 300

LOCAL_UPDATES = 10
LOCAL_LEARNING_RATE = 1e-2
REC_FED_DYN_SIGMA = femnist.REC_FED_DYN_SIGMA
REC_FED_DYN_MAX_LOCAL_ITERATIONS = femnist.REC_FED_DYN_MAX_LOCAL_ITERATIONS
IFEDDR_SIGMA_SQUARED = femnist.IFEDDR_SIGMA_SQUARED
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
    femnist.SCRIPT_DIRECTORY / "Results" / "FEMNISTOGLStepsizeTest" / "sample_200"
)


@dataclass(slots=True)
class CandidateHistory:
    communication: np.ndarray
    gradient_mapping: np.ndarray
    loss: np.ndarray
    test_accuracy: np.ndarray
    cumulative_oracles: np.ndarray
    process_time: np.ndarray
    inactive_group_fraction: np.ndarray
    refinement_count: np.ndarray | None = None
    relative_error_satisfied: np.ndarray | None = None


def algorithm_settings(name: str, stepsize: float):
    if name == "REC_FedDyn":
        return REC_FedDyn, dict(
            PenaltyParameter=1.0 / stepsize,
            Sigma=REC_FED_DYN_SIGMA,
            MaxLocalIterations=REC_FED_DYN_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "FedDR":
        return FedDR, dict(
            ProximalParameter=stepsize,
            RelaxationParameter=femnist.FEDDR_RELAXATION,
            LocalUpdates=LOCAL_UPDATES,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "iFedDR":
        return iFedDR, dict(
            ProximalParameter=stepsize,
            RelaxationParameter=femnist.IFEDDR_RELAXATION,
            SigmaSquared=IFEDDR_SIGMA_SQUARED,
            InitialLocalUpdates=LOCAL_UPDATES,
            RefinementFactor=femnist.IFEDDR_REFINEMENT_FACTOR,
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            MaxLocalIterations=IFEDDR_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "FedNMap":
        return FedNMap, dict(
            ProximalParameter=stepsize,
            LocalLearningRate=stepsize / LOCAL_UPDATES,
            ServerLearningRate=femnist.FEDNMAP_SERVER_LEARNING_RATE,
            LocalUpdates=LOCAL_UPDATES,
        )
    if name == "Zhang":
        return Zhang, dict(
            LocalLearningRate=stepsize / LOCAL_UPDATES,
            ServerLearningRate=femnist.ZHANG_SERVER_LEARNING_RATE,
            LocalUpdates=LOCAL_UPDATES,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def candidate_settings(name: str, stepsize: float, problem_key: str) -> dict:
    _, parameters = algorithm_settings(name, stepsize)
    return {
        "format": "femnist_ogl_stepsize_candidate_v1",
        "problem_key": problem_key,
        "algorithm": name,
        "overall_stepsize": stepsize,
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


def run_one(
    name: str,
    stepsize: float,
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
    algorithm_class, settings = algorithm_settings(name, stepsize)
    initial_loss, initial_mapping = torch_statistics(
        initial_point, functions, prox, OGL_REGULARIZATION, stepsize
    )
    result = algorithm_class(**common).go_torch(
        Function=functions,
        RecordStatistics=True,
        StationarityParameter=stepsize,
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
    name: str, stepsize: float, directory: Path, settings: dict
) -> CandidateHistory | None:
    suffix = f"stepsize_{stepsize:g}"
    settings_path = directory / f"{suffix}_settings.json"
    metrics_path = directory / f"{suffix}.csv"
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
    refinement_path = directory / f"{suffix}_refinements.csv"
    if name == "iFedDR" and refinement_path.is_file():
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
    stepsize: float,
    history: CandidateHistory,
    directory: Path,
    settings: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"stepsize_{stepsize:g}"
    np.savetxt(
        directory / f"{suffix}.csv",
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
    if history.refinement_count is not None:
        np.savetxt(
            directory / f"{suffix}_refinements.csv",
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
    with (directory / f"{suffix}_settings.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)


def save_plots(
    name: str, histories: dict[float, CandidateHistory], directory: Path
) -> None:
    plots = (
        ("gradient_mapping", "Gradient mapping norm", True,
         "gradient_mapping_vs_communication.png"),
        ("loss", "Regularized training loss", True,
         "loss_decay_vs_communication.png"),
        ("test_accuracy", "Test accuracy", False,
         "test_accuracy_vs_communication.png"),
        ("inactive_group_fraction", "Inactive group fraction", False,
         "inactive_groups_vs_communication.png"),
    )
    for field, vertical_label, logarithmic, filename in plots:
        figure = plt.figure(figsize=(8.4, 5.6))
        for stepsize, history in histories.items():
            values = getattr(history, field)
            if logarithmic:
                values = np.maximum(values, ERROR_FLOOR)
                plt.semilogy(
                    history.communication,
                    values,
                    linewidth=LINE_WIDTH,
                    label=f"stepsize={stepsize:g}",
                )
            else:
                plt.plot(
                    history.communication,
                    values,
                    linewidth=LINE_WIDTH,
                    label=f"stepsize={stepsize:g}",
                )
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(f"{name}: natural FEMNIST clients, participation=200/200")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        figure.savefig(directory / filename, dpi=FIGURE_DPI)
        plt.close(figure)


def save_summary(rows: list[tuple]) -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (RESULT_DIRECTORY / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "algorithm", "overall_stepsize", "final_gradient_mapping_norm",
            "final_loss", "final_test_accuracy", "inactive_group_fraction",
            "cumulative_gradient_oracles", "algorithm_time_seconds",
            "all_values_finite",
        ))
        writer.writerows(rows)

    best_rows: dict[str, tuple] = {}
    for row in rows:
        name, _, mapping, loss, accuracy, inactive, _, _, all_finite = row
        if not all_finite or not np.isfinite(
            (mapping, loss, accuracy, inactive)
        ).all():
            continue
        if name not in best_rows or mapping < best_rows[name][2]:
            best_rows[name] = row
    with (RESULT_DIRECTORY / "selected_stepsizes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "algorithm", "selected_overall_stepsize",
            "final_gradient_mapping_norm", "final_loss",
            "final_test_accuracy", "inactive_group_fraction",
        ))
        for name in sorted(best_rows):
            row = best_rows[name]
            writer.writerow((name, row[1], row[2], row[3], row[4], row[5]))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = femnist.load_femnist_clients(
        DATA_DIRECTORY,
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
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    femnist.save_json(RESULT_DIRECTORY / "problem_settings.json", problem)
    femnist.save_clients(clients, RESULT_DIRECTORY / "clients.csv")

    summary: list[tuple] = []
    print(
        f"Device={device}; clients={CLIENTS_NUMBER}; "
        f"participation={SAMPLING_NUMBER}/{CLIENTS_NUMBER}; "
        f"seed={ALGORITHM_SEED}; rounds={ITERATION_BUDGET}",
        flush=True,
    )
    print("Natural writer-ID partition; no synthetic heterogeneity.", flush=True)

    for name in ALGORITHM_NAMES:
        directory = RESULT_DIRECTORY / name
        directory.mkdir(parents=True, exist_ok=True)
        histories: dict[float, CandidateHistory] = {}
        for stepsize in OVERALL_STEPSIZES:
            print(f"Running {name}, stepsize={stepsize:g}...", flush=True)
            try:
                settings = candidate_settings(name, stepsize, problem_key)
                history = load_cached_history(
                    name, stepsize, directory, settings
                )
                if history is None:
                    history = run_one(
                        name,
                        stepsize,
                        initial_point,
                        functions,
                        model,
                        unpack,
                        clients,
                    )
                    save_history(name, stepsize, history, directory, settings)
                else:
                    print("  loaded completed candidate", flush=True)
                histories[stepsize] = history
                all_finite = all(np.isfinite(values).all() for values in (
                    history.gradient_mapping,
                    history.loss,
                    history.test_accuracy,
                    history.inactive_group_fraction,
                ))
                summary.append((
                    name,
                    stepsize,
                    history.gradient_mapping[-1],
                    history.loss[-1],
                    history.test_accuracy[-1],
                    history.inactive_group_fraction[-1],
                    history.cumulative_oracles[-1],
                    history.process_time[-1],
                    all_finite,
                ))
                save_summary(summary)
                print(
                    f"  mapping={history.gradient_mapping[-1]:.3e}; "
                    f"loss={history.loss[-1]:.4f}; "
                    f"accuracy={history.test_accuracy[-1]:.3f}",
                    flush=True,
                )
            except Exception:
                traceback.print_exc()
                print("  Failed; continuing...", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        save_plots(name, histories, directory)

    print(f"Saved results to {RESULT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    main()
