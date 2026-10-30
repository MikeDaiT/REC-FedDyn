"""Deterministic reproduction of the FedNMap shallow-network experiment.

The paper settings reproduced here are:

* MNIST with label-sorted, non-overlapping client partitions;
* a one-hidden-layer network with a sigmoid activation;
* full client participation;
* n in {20, 50, 100}, Q in {10, 20}, and 400 communication rounds;
* eta_a = 1 / Q, eta_s = 1, gamma = 4;
* phi(x) = nu1 * ||x||_1 + nu2 * ||x||_2^2, with
  nu1 = 0.001 and nu2 = 0.01.

Unlike the stochastic experiment in the paper, every client call in this script
uses its complete local data set, so the trajectory is deterministic for a fixed
seed.  The paper does not report the hidden width; 32 is therefore an explicit,
configurable assumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Algorithms.FedNMap import FedNMap
from Algorithms.TorchUtils import statistics as torch_statistics


ROOT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = ROOT_DIRECTORY / "Data" / "MNIST"
DEFAULT_OUTPUT_DIRECTORY = (
    ROOT_DIRECTORY / "Results" / "FedNMapShallowReproduction"
)


@dataclass(frozen=True)
class ExperimentSettings:
    clients: tuple[int, ...] = (20, 50, 100)
    local_updates: tuple[int, ...] = (10, 20)
    communication_rounds: int = 400
    hidden_units: int = 32
    training_size: int = 60_000
    gamma: float = 4.0
    server_learning_rate: float = 1.0
    nu1: float = 0.001
    nu2: float = 0.01
    l2_convention: str = "paper"
    seed: int = 2026
    progress_every: int = 10


class ShallowSigmoidNetwork(nn.Module):
    """One-hidden-layer MNIST classifier specified by the paper."""

    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(28 * 28, hidden_units)
        self.output_layer = nn.Linear(hidden_units, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(self.input_layer(images))
        return self.output_layer(hidden)


def set_determinism(seed: int) -> None:
    """Set all RNGs and deterministic PyTorch execution for one trajectory."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_elastic_net_prox(
    nu1: float,
    nu2: float,
    l2_convention: str = "paper",
) -> Callable:
    """Return the Elastic Net value/prox interface expected by ``FedNMap``.

    ``paper`` implements phi(x) = nu1 ||x||_1 + nu2 ||x||_2^2 exactly as
    printed in the paper.  ``author-code`` is provided for diagnostics and uses
    phi(x) = nu1 ||x||_1 + (nu2 / 2) ||x||_2^2.
    """

    if nu1 < 0 or nu2 < 0:
        raise ValueError("nu1 and nu2 must be nonnegative.")
    if l2_convention not in {"paper", "author-code"}:
        raise ValueError("l2_convention must be 'paper' or 'author-code'.")

    quadratic_factor = 2.0 if l2_convention == "paper" else 1.0
    value_factor = 1.0 if l2_convention == "paper" else 0.5

    def elastic_net(point, parameter, constant, only_value):
        del constant
        proximal_parameter = float(parameter)
        if proximal_parameter < 0:
            raise ValueError("The proximal parameter must be nonnegative.")

        if isinstance(point, torch.Tensor):
            if only_value:
                return (
                    nu1 * torch.linalg.vector_norm(point, ord=1)
                    + value_factor * nu2 * torch.sum(point.square())
                )
            thresholded = torch.sign(point) * torch.clamp(
                torch.abs(point) - proximal_parameter * nu1,
                min=0.0,
            )
            return thresholded / (
                1.0 + quadratic_factor * proximal_parameter * nu2
            )

        array = np.asarray(point, dtype=np.float64)
        if only_value:
            return np.float64(
                nu1 * np.sum(np.abs(array))
                + value_factor * nu2 * np.sum(array * array)
            )
        thresholded = np.sign(array) * np.maximum(
            np.abs(array) - proximal_parameter * nu1,
            0.0,
        )
        return thresholded / (
            1.0 + quadratic_factor * proximal_parameter * nu2
        )

    return elastic_net


def read_mnist_training_data(
    data_directory: Path,
    training_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the uncompressed IDX MNIST training files."""

    def resolve_idx_file(name: str) -> Path:
        candidate = data_directory / name
        if candidate.is_dir():
            candidate = candidate / name
        return candidate

    image_path = resolve_idx_file("train-images-idx3-ubyte")
    label_path = resolve_idx_file("train-labels-idx1-ubyte")
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(
            "MNIST IDX files were not found. Expected "
            f"{image_path} and {label_path}."
        )

    images_array = np.fromfile(
        image_path, dtype=np.uint8, offset=16
    ).reshape(-1, 28 * 28)
    labels_array = np.fromfile(label_path, dtype=np.uint8, offset=8)
    if len(images_array) != len(labels_array):
        raise ValueError("MNIST image and label counts do not agree.")
    if training_size <= 0 or training_size > len(labels_array):
        raise ValueError(
            f"training_size must be between 1 and {len(labels_array)}."
        )

    if training_size < len(labels_array):
        indices = np.random.default_rng(seed).choice(
            len(labels_array), training_size, replace=False
        )
        images_array = images_array[indices]
        labels_array = labels_array[indices]

    images = torch.tensor(images_array, dtype=torch.float32).div_(255.0)
    labels = torch.tensor(labels_array, dtype=torch.long)
    return images, labels


def sorted_label_partitions(
    labels: torch.Tensor,
    clients: int,
) -> list[torch.Tensor]:
    """Sort samples by label and split them equally without overlap."""

    if clients <= 0:
        raise ValueError("clients must be positive.")
    if len(labels) % clients != 0:
        raise ValueError(
            "The number of training samples must be divisible by the number "
            "of clients for equal-weight client objectives."
        )
    sorted_indices = torch.argsort(labels.detach().cpu(), stable=True)
    samples_per_client = len(labels) // clients
    return list(sorted_indices.reshape(clients, samples_per_client).unbind(0))


def parameter_unpacker(model: nn.Module) -> Callable[[torch.Tensor], dict]:
    layout = [
        (name, parameter.shape, parameter.numel())
        for name, parameter in model.named_parameters()
    ]

    def unpack(point: torch.Tensor) -> dict[str, torch.Tensor]:
        parameters: dict[str, torch.Tensor] = {}
        start = 0
        for name, shape, size in layout:
            parameters[name] = point[start:start + size].view(shape)
            start += size
        if start != point.numel():
            raise ValueError("The parameter vector has an unexpected size.")
        return parameters

    return unpack


def make_full_gradient_objective(
    model: nn.Module,
    unpack: Callable[[torch.Tensor], dict],
    images: torch.Tensor,
    labels: torch.Tensor,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic exact local empirical-risk oracle."""

    def objective(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variable = point.detach().requires_grad_(True)
        logits = functional_call(model, unpack(variable), (images,))
        loss = F.cross_entropy(logits, labels)
        gradient = torch.autograd.grad(loss, variable)[0]
        return loss.detach(), gradient.detach()

    return objective


def build_client_objectives(
    model: nn.Module,
    unpack: Callable[[torch.Tensor], dict],
    images: torch.Tensor,
    labels: torch.Tensor,
    clients: int,
) -> list[Callable]:
    objectives = []
    for indices in sorted_label_partitions(labels, clients):
        device_indices = indices.to(images.device)
        objectives.append(make_full_gradient_objective(
            model,
            unpack,
            images.index_select(0, device_indices),
            labels.index_select(0, device_indices),
        ))
    return objectives


def run_configuration(
    images: torch.Tensor,
    labels: torch.Tensor,
    settings: ExperimentSettings,
    clients: int,
    local_updates: int,
    device: torch.device,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Run one deterministic full-participation FedNMap trajectory."""

    if local_updates <= 0:
        raise ValueError("local_updates must be positive.")
    set_determinism(settings.seed)
    model = ShallowSigmoidNetwork(settings.hidden_units).to(device)
    model.eval()
    initial_z = parameters_to_vector(model.parameters()).detach().clone()
    unpack = parameter_unpacker(model)
    functions = build_client_objectives(
        model, unpack, images, labels, clients
    )
    prox = make_elastic_net_prox(
        settings.nu1, settings.nu2, settings.l2_convention
    )

    initial_x = prox(initial_z, settings.gamma, 0.0, False).detach()
    initial_value, initial_stationarity_norm = torch_statistics(
        initial_x,
        functions,
        prox,
        0.0,
        settings.gamma,
    )

    if settings.communication_rounds <= 0:
        raise ValueError("communication_rounds must be positive.")
    if settings.progress_every <= 0:
        raise ValueError("progress_every must be positive.")

    stationarity_parts = [initial_stationarity_norm.reshape(1)]
    objective_parts = [initial_value.reshape(1)]
    oracle_parts = []
    time_parts = []
    initial_state = None
    completed_rounds = 0
    while completed_rounds < settings.communication_rounds:
        chunk_rounds = min(
            settings.progress_every,
            settings.communication_rounds - completed_rounds,
        )
        optimizer = FedNMap(
            InitialPoint=initial_z,
            IterationBudget=chunk_rounds,
            ProxFunction=prox,
            ClientsNumber=clients,
            SamplingNumber=clients,
            SmoothL=0.0,
            ConvexMu=0.0,
            Constant=0.0,
            RandomSeed=settings.seed,
        )
        result = optimizer.go_torch(
            ProximalParameter=settings.gamma,
            LocalLearningRate=1.0 / local_updates,
            ServerLearningRate=settings.server_learning_rate,
            LocalUpdates=local_updates,
            Function=functions,
            StationarityParameter=settings.gamma,
            InitialState=initial_state,
            RecordStatistics=True,
        )
        initial_state = optimizer.final_state
        stationarity_parts.append(result.subgradient)
        objective_parts.append(result.function_value)
        oracle_parts.append(result.oracle_calling)
        time_parts.append(result.process_time)
        completed_rounds += chunk_rounds
        if verbose:
            print(
                f"  completed {completed_rounds}/"
                f"{settings.communication_rounds} rounds",
                flush=True,
            )

    stationarity_norm = torch.cat(stationarity_parts)
    objective = torch.cat(objective_parts)
    oracle_calling = torch.cat(oracle_parts)
    process_time = torch.cat(time_parts)
    communication = np.arange(
        settings.communication_rounds + 1, dtype=np.int64
    )
    cumulative_gradient_calls = np.concatenate((
        np.zeros(1, dtype=np.int64),
        torch.cumsum(oracle_calling, dim=0).cpu().numpy(),
    ))
    algorithm_time = np.concatenate((
        np.zeros(1, dtype=np.float64),
        process_time.detach().cpu().numpy().astype(np.float64),
    ))

    return {
        "communication": communication,
        # The paper reports the squared natural gradient mapping norm.
        "stationarity": stationarity_norm.detach().cpu().numpy() ** 2,
        "objective": objective.detach().cpu().numpy(),
        "cumulative_gradient_calls": cumulative_gradient_calls,
        "algorithm_time_seconds": algorithm_time,
    }


def save_configuration(
    output_directory: Path,
    clients: int,
    local_updates: int,
    result: dict[str, np.ndarray],
) -> Path:
    path = output_directory / f"fednmap_n{clients}_q{local_updates}.npz"
    np.savez(path, **result)
    return path


def load_configuration(path: Path, expected_rounds: int) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        result = {name: archive[name] for name in archive.files}
    expected_length = expected_rounds + 1
    if len(result.get("communication", ())) != expected_length:
        raise ValueError(f"Checkpoint {path} has an incompatible length.")
    return result


def write_csv(
    output_directory: Path,
    results: dict[tuple[int, int], dict[str, np.ndarray]],
) -> Path:
    path = output_directory / "fednmap_shallow_deterministic.csv"
    fieldnames = [
        "clients",
        "local_updates",
        "communication",
        "stationarity_squared",
        "objective",
        "cumulative_client_gradient_calls",
        "algorithm_time_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for (clients, local_updates), result in sorted(results.items()):
            for index, communication in enumerate(result["communication"]):
                writer.writerow({
                    "clients": clients,
                    "local_updates": local_updates,
                    "communication": int(communication),
                    "stationarity_squared": float(
                        result["stationarity"][index]
                    ),
                    "objective": float(result["objective"][index]),
                    "cumulative_client_gradient_calls": int(
                        result["cumulative_gradient_calls"][index]
                    ),
                    "algorithm_time_seconds": float(
                        result["algorithm_time_seconds"][index]
                    ),
                })
    return path


def plot_stationarity(
    output_directory: Path,
    settings: ExperimentSettings,
    results: dict[tuple[int, int], dict[str, np.ndarray]],
) -> Path:
    rows = len(settings.local_updates)
    columns = len(settings.clients)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.8 * rows),
        squeeze=False,
        sharex=True,
    )
    for row, local_updates in enumerate(settings.local_updates):
        for column, clients in enumerate(settings.clients):
            axis = axes[row, column]
            result = results[(clients, local_updates)]
            axis.semilogy(
                result["communication"],
                np.maximum(result["stationarity"], 1e-30),
                color="#0057b8",
                linewidth=1.4,
                label="FedNMap",
            )
            axis.set_title(f"n = {clients}, Q = {local_updates}")
            axis.grid(True, which="both", alpha=0.28)
            axis.set_xlabel("# Communications")
            if column == 0:
                axis.set_ylabel(r"$\|F_{nat}^{\gamma}(x_t)\|^2$")
            axis.legend(loc="best")

    figure.suptitle("Deterministic FedNMap on label-sorted MNIST")
    figure.tight_layout()
    path = output_directory / "fednmap_shallow_stationarity.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clients", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--local-updates", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--training-size", type=int, default=60_000)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--nu1", type=float, default=0.001)
    parser.add_argument("--nu2", type=float, default=0.01)
    parser.add_argument(
        "--l2-convention",
        choices=("paper", "author-code"),
        default="paper",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Run and report this many rounds per continuation chunk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-configuration NPZ files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a two-round CPU-friendly integration check.",
    )
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> ExperimentSettings:
    if args.smoke:
        return ExperimentSettings(
            clients=(2,),
            local_updates=(2,),
            communication_rounds=2,
            hidden_units=8,
            training_size=200,
            gamma=args.gamma,
            nu1=args.nu1,
            nu2=args.nu2,
            l2_convention=args.l2_convention,
            seed=args.seed,
            progress_every=1,
        )
    return ExperimentSettings(
        clients=tuple(args.clients),
        local_updates=tuple(args.local_updates),
        communication_rounds=args.rounds,
        hidden_units=args.hidden_units,
        training_size=args.training_size,
        gamma=args.gamma,
        nu1=args.nu1,
        nu2=args.nu2,
        l2_convention=args.l2_convention,
        seed=args.seed,
        progress_every=args.progress_every,
    )


def main() -> None:
    args = parse_args()
    settings = settings_from_args(args)
    device = torch.device("cpu") if args.smoke else resolve_device(args.device)
    output_directory = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    for clients in settings.clients:
        if settings.training_size % clients != 0:
            raise ValueError(
                f"training_size={settings.training_size} is not divisible by "
                f"clients={clients}."
            )

    images_cpu, labels_cpu = read_mnist_training_data(
        args.data_dir, settings.training_size, settings.seed
    )
    images = images_cpu.to(device)
    labels = labels_cpu.to(device)
    results: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for local_updates in settings.local_updates:
        for clients in settings.clients:
            checkpoint = (
                output_directory
                / f"fednmap_n{clients}_q{local_updates}.npz"
            )
            if args.resume and checkpoint.is_file():
                print(f"Loading {checkpoint.name}", flush=True)
                result = load_configuration(
                    checkpoint, settings.communication_rounds
                )
            else:
                print(
                    f"Running deterministic FedNMap: n={clients}, "
                    f"Q={local_updates}, rounds={settings.communication_rounds}, "
                    f"device={device}",
                    flush=True,
                )
                result = run_configuration(
                    images,
                    labels,
                    settings,
                    clients,
                    local_updates,
                    device,
                    verbose=True,
                )
                save_configuration(
                    output_directory, clients, local_updates, result
                )
            results[(clients, local_updates)] = result

    csv_path = write_csv(output_directory, results)
    figure_path = plot_stationarity(output_directory, settings, results)
    metadata = {
        "settings": asdict(settings),
        "device": str(device),
        "deterministic_full_client_gradients": True,
        "full_client_participation": True,
        "local_learning_rate_rule": "eta_a = 1 / Q",
        "paper_unspecified_assumptions": {
            "hidden_units": settings.hidden_units,
        },
        "outputs": {
            "csv": csv_path.name,
            "figure": figure_path.name,
        },
    }
    metadata_path = output_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved results to {output_directory}", flush=True)


if __name__ == "__main__":
    main()
