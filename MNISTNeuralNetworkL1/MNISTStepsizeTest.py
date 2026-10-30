"""Overall-stepsize test for the federated MNIST algorithms."""

import csv
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import MNISTExperiment as mnist
from Algorithms.REC_FedDyn import REC_FedDyn
from Algorithms.FedDR import FedDR
from Algorithms.FedNMap import FedNMap
from Algorithms.ProximalOperator import l_1
from Algorithms.TorchUtils import statistics as torch_statistics
from Algorithms.Zhang import Zhang
from Algorithms.iFedDR import iFedDR
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

# ======================================================================
# Test settings -- edit parameters in this section
# ======================================================================

OVERALL_STEPSIZES = (0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 70.0, 100.0)
ALGORITHM_GROUPS = (
    ("REC_FedDyn",),
    ("FedDR", "iFedDR"),
    ("FedNMap", "Zhang"),
)

CLIENTS_NUMBER = 20
SAMPLING_NUMBER = 20
DIRICHLET_CONCENTRATION = 10.0
TRAINING_SAMPLES = 60_000
TEST_SAMPLES = 10_000
ITERATION_BUDGET = 1000

LOCAL_UPDATES = 10
LOCAL_LEARNING_RATE = 1e-2
REC_FED_DYN_MAX_LOCAL_ITERATIONS = 30
IFEDDR_MAX_LOCAL_ITERATIONS = 100
IFEDDR_SIGMA_SQUARED = 0.99
IFEDDR_MAX_REFINEMENTS = 100
L1_REGULARIZATION = 1e-4

DATA_SEED = 7
MODEL_SEED = 7
ALGORITHM_SEED = 7
FIGURE_DPI = 300

PROJECT_DIRECTORY = Path("")  # Fill in the project directory used to store results.
DATA_DIRECTORY = Path("")  # Fill in the directory containing the MNIST IDX files.
RESULT_DIRECTORY = PROJECT_DIRECTORY / "Results" / "MNISTStepsizeTest"


def algorithm_settings(name, stepsize):
    if name == "REC_FedDyn":
        return REC_FedDyn, dict(
            PenaltyParameter=1.0 / stepsize,
            Sigma=0.2,
            MaxLocalIterations=REC_FED_DYN_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "FedDR":
        return FedDR, dict(
            ProximalParameter=stepsize,
            RelaxationParameter=1.0,
            LocalUpdates=LOCAL_UPDATES,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "iFedDR":
        return iFedDR, dict(
            ProximalParameter=stepsize,
            RelaxationParameter=1.0,
            SigmaSquared=IFEDDR_SIGMA_SQUARED,
            InitialLocalUpdates=LOCAL_UPDATES,
            RefinementFactor=2.0,
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            MaxLocalIterations=IFEDDR_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=LOCAL_LEARNING_RATE,
        )
    if name == "FedNMap":
        return FedNMap, dict(
            ProximalParameter=stepsize,
            LocalLearningRate=stepsize / LOCAL_UPDATES,
            ServerLearningRate=1.0,
            LocalUpdates=LOCAL_UPDATES,
        )
    if name == "Zhang":
        return Zhang, dict(
            LocalLearningRate=stepsize / LOCAL_UPDATES,
            ServerLearningRate=1.0,
            LocalUpdates=LOCAL_UPDATES,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def run_one(name, stepsize, initial_point, functions, model, unpack, test_data):
    common = dict(
        InitialPoint=initial_point,
        IterationBudget=ITERATION_BUDGET,
        ProxFunction=l_1,
        ClientsNumber=CLIENTS_NUMBER,
        SamplingNumber=SAMPLING_NUMBER,
        SmoothL=0.0,
        ConvexMu=0.0,
        Constant=L1_REGULARIZATION,
        RandomSeed=ALGORITHM_SEED,
    )
    algorithm_class, settings = algorithm_settings(name, stepsize)
    initial_loss, initial_mapping = torch_statistics(
        initial_point, functions, l_1, L1_REGULARIZATION, stepsize,
    )
    result = algorithm_class(**common).go_torch(
        Function=functions, RecordStatistics=True, **settings
    )
    points = torch.cat((initial_point[None], result.iterate_point))
    finite_points = torch.isfinite(points).all(dim=1).cpu().numpy()
    test_accuracy = mnist.accuracy_history(
        model, unpack, points, test_data[0], test_data[1]
    )
    test_accuracy[~finite_points] = np.nan

    return mnist.History(
        communication=np.arange(ITERATION_BUDGET + 1),
        gradient_mapping=np.concatenate((
            [float(initial_mapping)], result.subgradient.cpu().numpy()
        )),
        loss=np.concatenate((
            [float(initial_loss)], result.function_value.cpu().numpy()
        )),
        test_accuracy=test_accuracy,
        cumulative_oracles=np.concatenate((
            [0], np.cumsum(result.oracle_calling.cpu().numpy())
        )),
        process_time=np.concatenate((
            [0.0], result.process_time.cpu().numpy()
        )),
        refinement_count=(
            result.refinement_count.cpu().numpy()
            if name == "iFedDR" else None
        ),
        relative_error_satisfied=(
            np.logical_not(result.index.cpu().numpy())
            if name == "iFedDR" else None
        ),
    )


def candidate_settings(name, stepsize):
    _, parameters = algorithm_settings(name, stepsize)
    return {
        "format": "mnist_stepsize_candidate_v1",
        "algorithm": name,
        "overall_stepsize": stepsize,
        "algorithm_parameters": parameters,
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "dirichlet_concentration": DIRICHLET_CONCENTRATION,
        "training_samples": TRAINING_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "iteration_budget": ITERATION_BUDGET,
        "local_updates": LOCAL_UPDATES,
        "l1_regularization": L1_REGULARIZATION,
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "algorithm_seed": ALGORITHM_SEED,
    }


def load_history(path):
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if len(table) < ITERATION_BUDGET + 1:
        return None
    table = table[:ITERATION_BUDGET + 1]
    if int(table[-1, 0]) != ITERATION_BUDGET:
        return None
    return mnist.History(
        communication=table[:, 0].astype(int),
        gradient_mapping=table[:, 1],
        loss=table[:, 2],
        test_accuracy=table[:, 3],
        cumulative_oracles=table[:, 4].astype(int),
        process_time=table[:, 5],
    )


def load_cached_history(name, stepsize, directory, settings):
    suffix = f"stepsize_{stepsize:g}"
    history_path = directory / f"{suffix}.csv"
    settings_path = directory / f"{suffix}_settings.json"
    if not history_path.exists():
        return None

    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != settings:
                return None
    elif name == "iFedDR":
        # An unlabelled iFedDR file was produced by the former implementation.
        return None

    history = load_history(history_path)
    if history is None:
        return None
    if not settings_path.exists():
        with settings_path.open("w", encoding="utf-8") as stream:
            json.dump(settings, stream, indent=2, ensure_ascii=False)
        print(f"  reused legacy candidate: {history_path}", flush=True)
    return history


def save_history(name, stepsize, history, directory, settings):
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
        )),
        delimiter=",",
        fmt=("%d", "%.10e", "%.10e", "%.10e", "%d", "%.10e"),
        header=(
            "communication,gradient_mapping_norm,loss,test_accuracy,"
            "cumulative_gradient_oracles,process_time_seconds"
        ),
        comments="",
    )
    if history.refinement_count is not None:
        condition_satisfied = history.relative_error_satisfied
        if condition_satisfied is None:
            condition_satisfied = np.ones(
                len(history.refinement_count), dtype=np.bool_
            )
        np.savetxt(
            directory / f"{suffix}_refinements.csv",
            np.column_stack((
                np.arange(1, ITERATION_BUDGET + 1),
                history.refinement_count,
                condition_satisfied.astype(np.int8),
            )),
            delimiter=",", fmt="%d",
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


def save_plots(name, histories, directory):
    plots = (
        ("gradient_mapping", "Gradient mapping norm", True,
         "gradient_mapping_vs_communication.png"),
        ("loss", "Regularized training loss", True,
         "loss_decay_vs_communication.png"),
        ("test_accuracy", "Test accuracy", False,
         "test_accuracy_vs_communication.png"),
    )
    for field, vertical_label, logarithmic, filename in plots:
        figure = plt.figure(figsize=(8.4, 5.6))
        for stepsize, history in histories.items():
            plot = plt.semilogy if logarithmic else plt.plot
            plot(
                history.communication, getattr(history, field),
                linewidth=LINE_WIDTH,
                linestyle="-" if name == "REC_FedDyn" else "--",
                label=f"stepsize={stepsize:g}",
            )
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(f"{name}, Dirichlet concentration = 10")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        figure.savefig(directory / filename, dpi=FIGURE_DPI)
        plt.close(figure)


def save_summary(rows):
    with (RESULT_DIRECTORY / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow((
            "algorithm", "overall_stepsize", "final_gradient_mapping_norm",
            "final_loss", "final_test_accuracy",
            "cumulative_gradient_oracles", "all_values_finite",
        ))
        writer.writerows(rows)

    best_rows = {}
    for row in rows:
        name, _, mapping, loss, accuracy, _, all_finite = row
        if not all_finite or not np.isfinite((mapping, loss, accuracy)).all():
            continue
        if name not in best_rows or mapping < best_rows[name][2]:
            best_rows[name] = row
    with (RESULT_DIRECTORY / "selected_stepsizes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow((
            "algorithm", "selected_overall_stepsize",
            "final_gradient_mapping_norm", "final_loss",
            "final_test_accuracy",
        ))
        for name in sorted(best_rows):
            row = best_rows[name]
            writer.writerow((name, row[1], row[2], row[3], row[4]))


def main():
    mnist.DATA_DIRECTORY = DATA_DIRECTORY
    torch.manual_seed(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = mnist.read_mnist("train", TRAINING_SAMPLES, device)
    test_data = mnist.read_mnist("t10k", TEST_SAMPLES, device)
    client_indices = mnist.dirichlet_partition(
        train_data[1], DIRICHLET_CONCENTRATION
    )
    model = mnist.ShallowNet().to(device)
    unpack = mnist.parameter_unpacker(model)
    initial_point = parameters_to_vector(model.parameters()).detach()
    functions = [
        mnist.make_client_objective(
            model, unpack, train_data[0][indices], train_data[1][indices]
        )
        for indices in client_indices
    ]

    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    mnist.save_partition(train_data[1], client_indices, RESULT_DIRECTORY)
    summary = []
    print(f"Device: {device}", flush=True)

    for group in ALGORITHM_GROUPS:
        for name in group:
            directory = RESULT_DIRECTORY / name
            directory.mkdir(parents=True, exist_ok=True)
            histories = {}
            for stepsize in OVERALL_STEPSIZES:
                print(f"Running {name}, stepsize={stepsize:g}...", flush=True)
                try:
                    settings = candidate_settings(name, stepsize)
                    history = load_cached_history(
                        name, stepsize, directory, settings
                    )
                    if history is None:
                        history = run_one(
                            name, stepsize, initial_point, functions,
                            model, unpack, test_data,
                        )
                        save_history(
                            name, stepsize, history, directory, settings
                        )
                    else:
                        print("  loaded completed candidate", flush=True)
                    histories[stepsize] = history
                    all_finite = all(np.isfinite(values).all() for values in (
                        history.gradient_mapping,
                        history.loss,
                        history.test_accuracy,
                    ))
                    summary.append((
                        name, stepsize, history.gradient_mapping[-1],
                        history.loss[-1], history.test_accuracy[-1],
                        history.cumulative_oracles[-1], all_finite,
                    ))
                    save_summary(summary)
                    print(
                        f"  mapping={history.gradient_mapping[-1]:.3e}, "
                        f"loss={history.loss[-1]:.4f}, "
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
