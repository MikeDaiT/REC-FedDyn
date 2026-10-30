"""Federated MNIST experiment with Dirichlet label heterogeneity."""

import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector

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
# Experiment settings -- edit parameters in this section
# ======================================================================

CLIENTS_NUMBER = 20
SAMPLING_NUMBER = 20
DIRICHLET_CONCENTRATIONS = (0.1, 1.0, 10.0)

TRAINING_SAMPLES = 60_000
TEST_SAMPLES = 10_000
CLASSES_NUMBER = 10
HIDDEN_UNITS = 32
L1_REGULARIZATION = 1e-4
LOCAL_UPDATES = 10

ITERATION_BUDGET = 1000

ALGORITHM_NAMES = (
    "REC_FedDyn", "FedDR", "iFedDR", "FedNMap", "Zhang",
)

DATA_SEED = 7
MODEL_SEED = 7
ALGORITHM_SEEDS = (7, 8, 9, 10, 11)
FIGURE_DPI = 300
ERROR_FLOOR = 1e-16

PRIVATE_DIRECTORY = Path("")  # Fill in the parent directory of the project.
PROJECT_DIRECTORY = PRIVATE_DIRECTORY / "DisCompADMM"
DATA_DIRECTORY = Path("")  # Fill in the directory containing the MNIST IDX files.
RESULT_DIRECTORY = (
    PROJECT_DIRECTORY / "Results" / "MNISTExperiment"
    / f"sample_{SAMPLING_NUMBER}"
)


# ======================================================================
# Algorithm parameters
# ======================================================================

REC_FED_DYN_PENALTY_PARAMETER = 1 / 20.0
REC_FED_DYN_SIGMA = 0.2
REC_FED_DYN_MAX_LOCAL_ITERATIONS = 30
REC_FED_DYN_LOCAL_LEARNING_RATE = 1e-2

FEDDR_PROXIMAL_PARAMETER = 10.0
FEDDR_RELAXATION = 1.0
FEDDR_LOCAL_UPDATES = LOCAL_UPDATES
FEDDR_LOCAL_LEARNING_RATE = 1e-2

IFEDDR_PROXIMAL_PARAMETER = 1.0
IFEDDR_RELAXATION = 1.0
IFEDDR_SIGMA_SQUARED = 0.99
IFEDDR_INITIAL_LOCAL_UPDATES = LOCAL_UPDATES
IFEDDR_REFINEMENT_FACTOR = 2.0
IFEDDR_MAX_REFINEMENTS = 100
IFEDDR_MAX_LOCAL_ITERATIONS = 100
IFEDDR_LOCAL_LEARNING_RATE = 1e-2

FEDNMAP_PROXIMAL_PARAMETER = 1.0
FEDNMAP_LOCAL_UPDATES = LOCAL_UPDATES
FEDNMAP_LOCAL_LEARNING_RATE = 1.0 / FEDNMAP_LOCAL_UPDATES
FEDNMAP_SERVER_LEARNING_RATE = 1.0

ZHANG_LOCAL_UPDATES = LOCAL_UPDATES
ZHANG_SERVER_LEARNING_RATE = 1.0
ZHANG_LOCAL_LEARNING_RATE = (
    1.0 / (ZHANG_SERVER_LEARNING_RATE * ZHANG_LOCAL_UPDATES)
)


# ======================================================================
# Model, data, and recorded history
# ======================================================================

class ShallowNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, CLASSES_NUMBER),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


@dataclass
class History:
    communication: np.ndarray
    gradient_mapping: np.ndarray
    loss: np.ndarray
    test_accuracy: np.ndarray
    cumulative_oracles: np.ndarray
    process_time: np.ndarray
    selected_clients: np.ndarray | None = None
    local_oracles: np.ndarray | None = None
    local_criterion_satisfied: np.ndarray | None = None
    refinement_count: np.ndarray | None = None
    relative_error_satisfied: np.ndarray | None = None


def read_mnist(kind: str, size: int, device: torch.device):
    image_name = f"{kind}-images-idx3-ubyte"
    label_name = f"{kind}-labels-idx1-ubyte"
    image_path = DATA_DIRECTORY / image_name
    label_path = DATA_DIRECTORY / label_name
    if image_path.is_dir():
        image_path /= image_name
    if label_path.is_dir():
        label_path /= label_name

    images = np.fromfile(image_path, dtype=np.uint8, offset=16).reshape(-1, 784)
    labels = np.fromfile(label_path, dtype=np.uint8, offset=8)
    indices = np.random.default_rng(DATA_SEED).choice(
        len(labels), size, replace=False
    )
    return (
        torch.tensor(images[indices], dtype=torch.float32, device=device) / 255.0,
        torch.tensor(labels[indices], dtype=torch.long, device=device),
    )


def dirichlet_partition(labels: torch.Tensor, concentration: float):
    labels_numpy = labels.cpu().numpy()
    rng = np.random.default_rng(DATA_SEED)

    while True:
        partitions = [[] for _ in range(CLIENTS_NUMBER)]
        for label in range(CLASSES_NUMBER):
            indices = np.flatnonzero(labels_numpy == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(
                np.full(CLIENTS_NUMBER, concentration)
            )
            counts = rng.multinomial(len(indices), proportions)
            splits = np.split(indices, np.cumsum(counts)[:-1])
            for client, split in enumerate(splits):
                partitions[client].append(split)

        client_indices = [np.concatenate(parts) for parts in partitions]
        if all(len(indices) > 0 for indices in client_indices):
            return [
                torch.tensor(
                    rng.permutation(indices), dtype=torch.long,
                    device=labels.device,
                )
                for indices in client_indices
            ]


def parameter_unpacker(model: nn.Module):
    layout = [
        (name, parameter.shape, parameter.numel())
        for name, parameter in model.named_parameters()
    ]

    def unpack(point: torch.Tensor):
        parameters, start = {}, 0
        for name, shape, size in layout:
            parameters[name] = point[start:start + size].view(shape)
            start += size
        return parameters

    return unpack


def make_client_objective(model, unpack, images, labels):
    def objective(point):
        variable = point.detach().requires_grad_(True)
        loss = F.cross_entropy(
            functional_call(model, unpack(variable), (images,)), labels
        )
        gradient = torch.autograd.grad(loss, variable)[0]
        return loss.detach(), gradient.detach()

    return objective


@torch.no_grad()
def accuracy_history(model, unpack, points, test_images, test_labels):
    return np.array([
        float((
            functional_call(model, unpack(point), (test_images,)).argmax(1)
            == test_labels
        ).float().mean())
        for point in points
    ])


# ======================================================================
# Algorithms
# ======================================================================

def algorithm_settings(name):
    if name == "REC_FedDyn":
        return REC_FedDyn, dict(
            PenaltyParameter=REC_FED_DYN_PENALTY_PARAMETER,
            Sigma=REC_FED_DYN_SIGMA,
            MaxLocalIterations=REC_FED_DYN_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=REC_FED_DYN_LOCAL_LEARNING_RATE,
        ), 1.0 / REC_FED_DYN_PENALTY_PARAMETER
    if name == "FedDR":
        return FedDR, dict(
            ProximalParameter=FEDDR_PROXIMAL_PARAMETER,
            RelaxationParameter=FEDDR_RELAXATION,
            LocalUpdates=FEDDR_LOCAL_UPDATES,
            LocalLearningRate=FEDDR_LOCAL_LEARNING_RATE,
        ), FEDDR_PROXIMAL_PARAMETER
    if name == "iFedDR":
        return iFedDR, dict(
            ProximalParameter=IFEDDR_PROXIMAL_PARAMETER,
            RelaxationParameter=IFEDDR_RELAXATION,
            SigmaSquared=IFEDDR_SIGMA_SQUARED,
            InitialLocalUpdates=IFEDDR_INITIAL_LOCAL_UPDATES,
            RefinementFactor=IFEDDR_REFINEMENT_FACTOR,
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            MaxLocalIterations=IFEDDR_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=IFEDDR_LOCAL_LEARNING_RATE,
        ), IFEDDR_PROXIMAL_PARAMETER
    if name == "FedNMap":
        return FedNMap, dict(
            ProximalParameter=FEDNMAP_PROXIMAL_PARAMETER,
            LocalLearningRate=FEDNMAP_LOCAL_LEARNING_RATE,
            ServerLearningRate=FEDNMAP_SERVER_LEARNING_RATE,
            LocalUpdates=FEDNMAP_LOCAL_UPDATES,
        ), FEDNMAP_PROXIMAL_PARAMETER
    if name == "Zhang":
        return Zhang, dict(
            LocalLearningRate=ZHANG_LOCAL_LEARNING_RATE,
            ServerLearningRate=ZHANG_SERVER_LEARNING_RATE,
            LocalUpdates=ZHANG_LOCAL_UPDATES,
        ), (
            ZHANG_LOCAL_LEARNING_RATE
            * ZHANG_SERVER_LEARNING_RATE
            * ZHANG_LOCAL_UPDATES
        )
    raise ValueError(f"Unknown algorithm: {name}")


def run_algorithm(
    name, algorithm_seed, initial_point, functions, model, unpack, test_data,
):
    common = dict(
        InitialPoint=initial_point,
        IterationBudget=ITERATION_BUDGET,
        ProxFunction=l_1,
        ClientsNumber=CLIENTS_NUMBER,
        SamplingNumber=SAMPLING_NUMBER,
        SmoothL=0.0,
        ConvexMu=0.0,
        Constant=L1_REGULARIZATION,
        RandomSeed=algorithm_seed,
    )
    algorithm_class, settings, mapping_parameter = algorithm_settings(name)
    initial_loss, initial_mapping = torch_statistics(
        initial_point, functions, l_1, L1_REGULARIZATION, mapping_parameter,
    )
    algorithm = algorithm_class(**common)
    result = algorithm.go_torch(
        Function=functions, RecordStatistics=True, **settings
    )
    points = torch.cat((initial_point[None], result.iterate_point))
    communication = np.arange(ITERATION_BUDGET + 1)
    return History(
        communication=communication,
        gradient_mapping=np.concatenate((
            [float(initial_mapping)], result.subgradient.cpu().numpy()
        )),
        loss=np.concatenate((
            [float(initial_loss)], result.function_value.cpu().numpy()
        )),
        test_accuracy=accuracy_history(
            model, unpack, points, test_data[0], test_data[1]
        ),
        cumulative_oracles=np.concatenate((
            [0], np.cumsum(result.oracle_calling.cpu().numpy())
        )),
        process_time=np.concatenate((
            [0.0], result.process_time.cpu().numpy()
        )),
        selected_clients=(
            result.selected_clients.cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
        local_oracles=(
            result.local_oracle_calling.cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
        local_criterion_satisfied=(
            result.local_criterion_satisfied.cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
        refinement_count=(
            result.refinement_count.cpu().numpy()
            if name == "iFedDR" else None
        ),
        relative_error_satisfied=(
            np.logical_not(result.index.cpu().numpy())
            if name == "iFedDR" else None
        ),
    )


# ======================================================================
# Save data and figures
# ======================================================================

def save_partition(labels, client_indices, result_directory):
    rows = []
    for client, indices in enumerate(client_indices):
        counts = torch.bincount(
            labels[indices], minlength=CLASSES_NUMBER
        ).cpu().numpy()
        rows.append(np.concatenate(([client, len(indices)], counts)))
    np.savetxt(
        result_directory / "client_label_distribution.csv",
        np.asarray(rows), delimiter=",", fmt="%d",
        header="client,total," + ",".join(
            f"label_{label}" for label in range(CLASSES_NUMBER)
        ),
        comments="",
    )


def settings_key(settings):
    encoded = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def problem_settings(concentration):
    return {
        "format": "mnist_problem_v1",
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "dirichlet_concentration": concentration,
        "training_samples": TRAINING_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "classes_number": CLASSES_NUMBER,
        "hidden_units": HIDDEN_UNITS,
        "l1_regularization": L1_REGULARIZATION,
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
    }


def run_settings(name, problem_key, algorithm_seed):
    _, parameters, mapping_parameter = algorithm_settings(name)
    return {
        "format": "mnist_algorithm_v1",
        "problem_key": problem_key,
        "algorithm": name,
        "algorithm_seed": algorithm_seed,
        "stepsize": float(mapping_parameter),
        "algorithm_parameters": parameters,
        "gradient_mapping_parameter": mapping_parameter,
    }


def algorithm_run_directory(
    problem_directory, name, algorithm_seed, settings,
):
    return (
        problem_directory / "algorithms" / name / f"seed_{algorithm_seed}"
        / f"run_{settings_key(settings)}"
    )


def save_settings(directory, settings):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved != settings:
            raise ValueError(f"Settings do not match: {path}")
        return
    with path.open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)


def save_history(history, directory):
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        directory / "metrics.csv",
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
    if history.selected_clients is not None:
        local_rounds = np.repeat(
            np.arange(1, history.selected_clients.shape[0] + 1),
            history.selected_clients.shape[1],
        )
        np.savetxt(
            directory / "local_solves.csv",
            np.column_stack((
                local_rounds,
                history.selected_clients.ravel(),
                history.local_oracles.ravel(),
                history.local_criterion_satisfied.ravel(),
            )),
            delimiter=",", fmt="%d",
            header=(
                "communication,client,gradient_oracles,criterion_satisfied"
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
            directory / "refinements.csv",
            np.column_stack((
                np.arange(1, len(history.refinement_count) + 1),
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


def load_history(path):
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return History(
        communication=table[:, 0].astype(int),
        gradient_mapping=table[:, 1],
        loss=table[:, 2],
        test_accuracy=table[:, 3],
        cumulative_oracles=table[:, 4].astype(int),
        process_time=table[:, 5],
    )


def completed_history(path):
    if not path.exists():
        return None
    history = load_history(path)
    if history.communication[-1] < ITERATION_BUDGET:
        return None
    count = ITERATION_BUDGET + 1
    return History(
        communication=history.communication[:count],
        gradient_mapping=history.gradient_mapping[:count],
        loss=history.loss[:count],
        test_accuracy=history.test_accuracy[:count],
        cumulative_oracles=history.cumulative_oracles[:count],
        process_time=history.process_time[:count],
    )


def load_cached_history(directory, settings):
    settings_path = directory / "settings.json"
    if not settings_path.exists():
        return None
    with settings_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != settings:
            return None
    return completed_history(directory / "metrics.csv")


def history_range(histories, field):
    if not histories:
        raise ValueError("At least one history is required.")
    communication = histories[0].communication
    if any(
        not np.array_equal(history.communication, communication)
        for history in histories[1:]
    ):
        raise ValueError("Seed histories do not share a communication axis.")
    values = np.stack([getattr(history, field) for history in histories])
    return (
        communication,
        np.mean(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
    )


def save_results(histories, result_directory, concentration):
    result_directory.mkdir(parents=True, exist_ok=True)
    for name, seed_histories in histories.items():
        if not seed_histories:
            continue
        columns = [seed_histories[0].communication]
        header = ["communication"]
        for field in (
            "gradient_mapping", "loss", "test_accuracy",
            "cumulative_oracles", "process_time",
        ):
            _, mean, minimum, maximum = history_range(seed_histories, field)
            columns.extend((mean, minimum, maximum))
            header.extend((f"{field}_mean", f"{field}_min", f"{field}_max"))
        np.savetxt(
            result_directory / f"{name}_mean_range.csv",
            np.column_stack(columns),
            delimiter=",", fmt="%.10e", header=",".join(header), comments="",
        )

    plots = [
        ("gradient_mapping", "Gradient mapping norm", True,
         "gradient_mapping_vs_communication.pdf"),
        ("loss", "Regularized training loss", True,
         "loss_decay_vs_communication.pdf"),
        ("test_accuracy", "Test accuracy", False,
         "test_accuracy_vs_communication.pdf"),
    ]
    for field, vertical_label, logarithmic, filename in plots:
        figure = plt.figure(figsize=(8.4, 5.6))
        for name, seed_histories in histories.items():
            if not seed_histories:
                continue
            communication, mean, minimum, maximum = history_range(
                seed_histories, field
            )
            if logarithmic:
                mean = np.maximum(mean, ERROR_FLOOR)
                minimum = np.maximum(minimum, ERROR_FLOOR)
                maximum = np.maximum(maximum, ERROR_FLOOR)
            plot = plt.semilogy if logarithmic else plt.plot
            line = plot(
                communication, mean, linewidth=LINE_WIDTH,
                linestyle="-" if name == "REC_FedDyn" else "--",
                label="REC_FedDyn" if name == "REC_FedDyn" else name,
            )
            plt.fill_between(
                communication, minimum, maximum,
                color=line[0].get_color(), alpha=0.16, linewidth=0.0,
            )
        plt.xlabel("Communication rounds")
        plt.ylabel(vertical_label)
        plt.title(
            f"Dirichlet concentration = {concentration:g}, "
            f"sampled clients = {SAMPLING_NUMBER}/{CLIENTS_NUMBER}"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        figure.savefig(result_directory / filename, dpi=FIGURE_DPI)
        plt.close(figure)


def main():
    torch.manual_seed(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = read_mnist("train", TRAINING_SAMPLES, device)
    test_data = read_mnist("t10k", TEST_SAMPLES, device)
    model = ShallowNet().to(device)
    unpack = parameter_unpacker(model)
    initial_point = parameters_to_vector(model.parameters()).detach()

    print(f"Device: {device}", flush=True)
    for concentration in DIRICHLET_CONCENTRATIONS:
        concentration_directory = RESULT_DIRECTORY / f"alpha_{concentration:g}"
        problem_parameters = problem_settings(concentration)
        problem_key = settings_key(problem_parameters)
        result_directory = (
            concentration_directory / f"problem_{problem_key}"
        )
        result_directory.mkdir(parents=True, exist_ok=True)
        save_settings(result_directory, problem_parameters)
        client_indices = dirichlet_partition(train_data[1], concentration)
        functions = [
            make_client_objective(
                model, unpack, train_data[0][indices], train_data[1][indices]
            )
            for indices in client_indices
        ]
        save_partition(train_data[1], client_indices, result_directory)

        print(f"Dirichlet concentration: {concentration:g}", flush=True)
        print(
            "Client samples: "
            f"min={min(map(len, client_indices))}, "
            f"max={max(map(len, client_indices))}",
            flush=True,
        )
        histories = {name: [] for name in ALGORITHM_NAMES}
        for algorithm_seed in ALGORITHM_SEEDS:
            print(f"Algorithm seed: {algorithm_seed}", flush=True)
            for name in ALGORITHM_NAMES:
                settings = run_settings(name, problem_key, algorithm_seed)
                run_directory = algorithm_run_directory(
                    result_directory, name, algorithm_seed, settings
                )
                history = load_cached_history(run_directory, settings)
                if history is not None:
                    print(
                        f"  loaded {name}: {ITERATION_BUDGET} rounds; "
                        "no new iterations",
                        flush=True,
                    )
                    histories[name].append(history)
                    continue

                print(f"  running {name}...", flush=True)
                try:
                    history = run_algorithm(
                        name,
                        algorithm_seed,
                        initial_point,
                        functions,
                        model,
                        unpack,
                        test_data,
                    )
                    save_settings(run_directory, settings)
                    save_history(history, run_directory)
                    histories[name].append(history)
                    print(
                        f"    communication={history.communication[-1]}, "
                        f"mapping={history.gradient_mapping[-1]:.3e}, "
                        f"loss={history.loss[-1]:.4f}, "
                        f"accuracy={history.test_accuracy[-1]:.3f}, "
                        f"oracles={history.cumulative_oracles[-1]}",
                        flush=True,
                    )
                except Exception:
                    traceback.print_exc()
                    print(
                        f"    failed {name}, seed={algorithm_seed}; continuing...",
                        flush=True,
                    )
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        for name, seed_histories in histories.items():
            if len(seed_histories) != len(ALGORITHM_SEEDS):
                print(
                    f"Warning: {name} has {len(seed_histories)}/"
                    f"{len(ALGORITHM_SEEDS)} successful seeds.",
                    flush=True,
                )
        save_results(histories, result_directory, concentration)
        print(f"Saved results to {result_directory}", flush=True)


if __name__ == "__main__":
    main()
