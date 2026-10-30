"""Reproduce the CIFAR-10 linear-probing iFedDR experiment.

The default configuration follows Figure 3 and Table 3 of the iFedDR paper:
20 clients, a Dirichlet(0.1) label split, 500 communication rounds, an
ImageNet-pretrained ResNet18 feature extractor, gamma=lambda=1,
sigma_squared=0.99, ten initial local updates, and l2 regularization 1e-5.

Only the final linear classifier is optimized.  ResNet18 is evaluated once and
its 512-dimensional features are cached, making every client objective a
deterministic convex multinomial-logistic-regression problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from Algorithms.TorchUtils import statistics as torch_statistics
from Algorithms.iFedDR import iFedDR


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = Path("")  # Fill in the extracted CIFAR-10 data directory.
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY / "Results" / "CIFAR10iFedDRLinearProbing"
)
FEATURE_CACHE_VERSION = 1
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    clients: int = 20
    classes: int = 10
    feature_dimension: int = 512
    dirichlet_concentration: float = 0.1
    communication_rounds: int = 500
    l2_regularization: float = 1e-5
    proximal_parameter: float = 1.0
    relaxation_parameter: float = 1.0
    sigma_squared: float = 0.99
    initial_local_updates: int = 10
    maximum_refinements: int = 100
    increase_initial_updates: bool = True
    seed: int = 0
    dtype: str = "float32"
    smoothness_power_iterations: int = 40


class CIFAR10BatchDataset(torch.utils.data.Dataset):
    """Read an existing CIFAR-10 Python-batch directory without downloading."""

    def __init__(self, batch_directory: Path, batch_names: tuple[str, ...]):
        arrays: list[np.ndarray] = []
        targets: list[int] = []
        for name in batch_names:
            path = batch_directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Required CIFAR-10 batch is missing: {path}")
            with path.open("rb") as stream:
                payload = pickle.load(stream, encoding="latin1")
            data = payload.get("data", payload.get(b"data"))
            labels = payload.get("labels", payload.get(b"labels"))
            if data is None or labels is None:
                raise ValueError(f"Invalid CIFAR-10 Python batch: {path}")
            data_array = np.asarray(data, dtype=np.uint8)
            if data_array.ndim != 2 or data_array.shape[1] != 3 * 32 * 32:
                raise ValueError(f"Invalid CIFAR-10 image array in {path}")
            if len(data_array) != len(labels):
                raise ValueError(f"Image and label counts disagree in {path}")
            arrays.append(data_array)
            targets.extend(int(label) for label in labels)

        self.data = np.concatenate(arrays, axis=0)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.data[index]).reshape(3, 32, 32)
        image = image.to(dtype=torch.float32).div_(255.0)
        image.sub_(self.mean).div_(self.std)
        return image, self.targets[index]


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return device


def zero_regularizer(
    point: torch.Tensor, parameter: float, constant: float, only_value: bool,
) -> torch.Tensor:
    """The paper puts l2 regularization inside every smooth client loss."""

    del parameter, constant
    if only_value:
        return point.new_zeros(())
    return point.detach().clone()


def dirichlet_partition(
    labels: torch.Tensor | np.ndarray,
    clients: int,
    concentration: float,
    seed: int,
    minimum_samples: int = 1,
) -> list[np.ndarray]:
    """Make a deterministic class-wise Dirichlet partition.

    Every training example appears in exactly one client.  The draw is retried
    only if a client receives fewer than ``minimum_samples`` examples.
    """

    if clients <= 0:
        raise ValueError("clients must be positive.")
    if concentration <= 0:
        raise ValueError("concentration must be positive.")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive.")
    labels_array = (
        labels.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    )
    if labels_array.ndim != 1 or len(labels_array) < clients * minimum_samples:
        raise ValueError("There are not enough one-dimensional labels.")

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels_array)
    for _ in range(1_000):
        pieces: list[list[np.ndarray]] = [[] for _ in range(clients)]
        for label in unique_labels:
            indices = np.flatnonzero(labels_array == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(clients, concentration))
            counts = rng.multinomial(len(indices), proportions)
            splits = np.split(indices, np.cumsum(counts)[:-1])
            for client, split in enumerate(splits):
                pieces[client].append(split)
        partitions = [
            rng.permutation(np.concatenate(client_pieces)).astype(np.int64)
            for client_pieces in pieces
        ]
        if min(map(len, partitions)) >= minimum_samples:
            return partitions
    raise RuntimeError("Could not draw a nonempty Dirichlet partition.")


def extract_cifar10_features(
    data_directory: Path,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    rebuild: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load or create frozen ImageNet-ResNet18 CIFAR-10 features.

    CIFAR-10 must already exist below ``data_directory``.  This function never
    downloads or modifies the dataset files.
    """

    if cache_path.is_file() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("version") != FEATURE_CACHE_VERSION:
            raise ValueError(
                f"Feature cache {cache_path} has an incompatible version."
            )
        return (
            payload["train_features"],
            payload["train_labels"],
            payload["test_features"],
            payload["test_labels"],
        )

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required to extract CIFAR-10 features."
        ) from error

    batch_directory = data_directory / "cifar-10-batches-py"
    train_dataset = CIFAR10BatchDataset(
        batch_directory,
        tuple(f"data_batch_{index}" for index in range(1, 6)),
    )
    test_dataset = CIFAR10BatchDataset(batch_directory, ("test_batch",))

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    def encode(dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        feature_batches, label_batches = [], []
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(
                    device, non_blocking=device.type == "cuda"
                )
                feature_batches.append(backbone(images).cpu())
                label_batches.append(labels.cpu())
        return torch.cat(feature_batches), torch.cat(label_batches)

    print("Extracting frozen ResNet18 training features...", flush=True)
    train_features, train_labels = encode(train_dataset)
    print("Extracting frozen ResNet18 test features...", flush=True)
    test_features, test_labels = encode(test_dataset)
    if train_features.shape[1] != 512 or test_features.shape[1] != 512:
        raise RuntimeError("ResNet18 did not produce 512-dimensional features.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": FEATURE_CACHE_VERSION,
        "backbone": "torchvision-resnet18-imagenet1k-v1",
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
        "input_resized": False,
        "adaptive_pooling": True,
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }, cache_path)
    return train_features, train_labels, test_features, test_labels


def unpack_linear_probe(
    point: torch.Tensor, feature_dimension: int, classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_size = feature_dimension * classes
    if point.numel() != weight_size + classes:
        raise ValueError("The linear-probe parameter vector has the wrong size.")
    return (
        point[:weight_size].reshape(classes, feature_dimension),
        point[weight_size:],
    )


def initialize_linear_probe(
    feature_dimension: int,
    classes: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Match ``torch.nn.Linear``'s default uniform initialization."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bound = 1.0 / math.sqrt(feature_dimension)
    weight = torch.empty(
        classes, feature_dimension, dtype=dtype, device="cpu"
    ).uniform_(-bound, bound, generator=generator)
    bias = torch.empty(classes, dtype=dtype, device="cpu").uniform_(
        -bound, bound, generator=generator
    )
    return torch.cat((weight.reshape(-1), bias)).to(device)


def make_linear_probe_objective(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    l2_regularization: float,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic multinomial logistic-regression objective."""

    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("features must be a matrix and labels a vector.")
    if len(features) != len(labels) or len(labels) == 0:
        raise ValueError("features and labels must have the same nonzero length.")
    if l2_regularization < 0:
        raise ValueError("l2_regularization must be nonnegative.")
    labels = labels.long()
    samples, feature_dimension = features.shape
    row_indices = torch.arange(samples, device=features.device)

    def objective(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            weight, bias = unpack_linear_probe(
                point, feature_dimension, classes
            )
            logits = features @ weight.T + bias
            value = F.cross_entropy(logits, labels)
            value = value + 0.5 * l2_regularization * torch.sum(point.square())

            residual = torch.softmax(logits, dim=1)
            residual[row_indices, labels] -= 1.0
            residual /= samples
            weight_gradient = residual.T @ features
            bias_gradient = residual.sum(dim=0)
            gradient = torch.cat((
                weight_gradient.reshape(-1), bias_gradient,
            ))
            gradient.add_(point, alpha=l2_regularization)
            return value, gradient

    return objective


def estimate_softmax_smoothness(
    features: torch.Tensor,
    l2_regularization: float,
    power_iterations: int,
) -> float:
    """Estimate a safe softmax-gradient Lipschitz upper bound.

    For augmented examples ``[feature, 1]`` the bound is
    ``0.5 * ||X||_2^2 / n + l2``.  Power iteration estimates the spectral
    norm and a small safety factor protects the accelerated local solver.
    """

    if power_iterations <= 0:
        raise ValueError("power_iterations must be positive.")
    samples, dimension = features.shape
    vector = torch.linspace(
        1.0, 2.0, dimension + 1,
        dtype=features.dtype, device=features.device,
    )
    vector /= torch.linalg.vector_norm(vector)
    with torch.no_grad():
        for _ in range(power_iterations):
            product = features @ vector[:-1] + vector[-1]
            next_vector = torch.cat((features.T @ product, product.sum()[None]))
            norm = torch.linalg.vector_norm(next_vector)
            if float(norm) == 0.0:
                return float(l2_regularization)
            vector = next_vector / norm
        product = features @ vector[:-1] + vector[-1]
        spectral_squared = torch.sum(product.square())
    return 1.01 * 0.5 * float(spectral_squared) / samples + l2_regularization


def linear_probe_accuracy(
    point: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    batch_size: int = 16_384,
) -> float:
    correct = 0
    weight, bias = unpack_linear_probe(point, features.shape[1], classes)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            predictions = (
                features[start:stop] @ weight.T + bias
            ).argmax(dim=1)
            correct += int(torch.sum(predictions == labels[start:stop]))
    return correct / len(labels)


def build_client_problem(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    settings: ExperimentSettings,
) -> tuple[
    list[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    list[np.ndarray],
    list[float],
]:
    partitions = dirichlet_partition(
        train_labels,
        settings.clients,
        settings.dirichlet_concentration,
        settings.seed,
    )
    objectives = []
    smoothness = []
    for indices in partitions:
        device_indices = torch.as_tensor(
            indices, dtype=torch.long, device=train_features.device
        )
        local_features = train_features.index_select(0, device_indices)
        local_labels = train_labels.index_select(0, device_indices)
        objectives.append(make_linear_probe_objective(
            local_features,
            local_labels,
            settings.classes,
            settings.l2_regularization,
        ))
        smoothness.append(estimate_softmax_smoothness(
            local_features,
            settings.l2_regularization,
            settings.smoothness_power_iterations,
        ))
    return objectives, partitions, smoothness


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def save_checkpoint(
    path: Path,
    settings: ExperimentSettings,
    initial_point: torch.Tensor,
    algorithm_state: dict[str, object],
    history: dict[str, list[float | int]],
    client_smoothness: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "settings": asdict(settings),
        "initial_point": initial_point.detach().cpu(),
        "algorithm_state": _to_cpu(algorithm_state),
        "history": history,
        "client_smoothness": client_smoothness,
    }, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    settings: ExperimentSettings,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    dict[str, object],
    dict[str, list[float | int]],
    list[float],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("settings") != asdict(settings):
        raise ValueError(
            f"Checkpoint settings in {path} do not match this experiment."
        )
    return (
        payload["initial_point"].to(device),
        _to_device(payload["algorithm_state"], device),
        payload["history"],
        [float(value) for value in payload["client_smoothness"]],
    )


def run_ifeddr(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    settings: ExperimentSettings,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
) -> tuple[dict[str, list[float | int]], list[np.ndarray], list[float]]:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    objectives, partitions, client_smoothness = build_client_problem(
        train_features, train_labels, settings
    )
    global_smoothness = max(client_smoothness)
    device = train_features.device
    dtype = train_features.dtype

    if resume and checkpoint_path.is_file():
        initial_point, algorithm_state, history, saved_smoothness = load_checkpoint(
            checkpoint_path, settings, device
        )
        if not np.allclose(saved_smoothness, client_smoothness, rtol=1e-5):
            raise ValueError("Checkpoint and current features have different smoothness.")
        completed_rounds = int(history["communication_round"][-1])
        print(
            f"Resuming {checkpoint_path.name} at round {completed_rounds}.",
            flush=True,
        )
    else:
        initial_point = initialize_linear_probe(
            settings.feature_dimension,
            settings.classes,
            dtype,
            device,
            settings.seed,
        )
        initial_value, initial_stationarity = torch_statistics(
            initial_point,
            objectives,
            zero_regularizer,
            0.0,
            settings.proximal_parameter,
        )
        history = {
            "communication_round": [0],
            "train_objective": [float(initial_value)],
            "stationarity_norm": [float(initial_stationarity)],
            "test_accuracy": [linear_probe_accuracy(
                initial_point, test_features, test_labels, settings.classes
            )],
            "cumulative_gradient_calls": [0],
            "cumulative_local_iterations": [0],
            "cumulative_refinements": [0],
            "algorithm_time_seconds": [0.0],
        }
        algorithm_state = None
        completed_rounds = 0

    while completed_rounds < settings.communication_rounds:
        chunk_rounds = min(
            checkpoint_every,
            settings.communication_rounds - completed_rounds,
        )
        optimizer = iFedDR(
            InitialPoint=initial_point,
            IterationBudget=chunk_rounds,
            ProxFunction=zero_regularizer,
            ClientsNumber=settings.clients,
            SamplingNumber=settings.clients,
            SmoothL=np.float64(global_smoothness),
            ConvexMu=np.float64(settings.l2_regularization),
            Constant=np.float64(0.0),
            RandomSeed=settings.seed,
        )
        result = optimizer.go_torch(
            ProximalParameter=settings.proximal_parameter,
            RelaxationParameter=settings.relaxation_parameter,
            SigmaSquared=settings.sigma_squared,
            InitialLocalUpdates=settings.initial_local_updates,
            MaxRefinements=settings.maximum_refinements,
            IncreaseInitialUpdates=settings.increase_initial_updates,
            Function=objectives,
            StatisticsFunction=objectives,
            StationarityParameter=settings.proximal_parameter,
            InitialState=algorithm_state,
            RecordStatistics=True,
        )
        algorithm_state = optimizer.final_state

        previous_calls = int(history["cumulative_gradient_calls"][-1])
        calls = previous_calls + torch.cumsum(
            result.oracle_calling, dim=0
        ).detach().cpu().numpy()
        previous_refinements = int(history["cumulative_refinements"][-1])
        additional_refinements = torch.clamp(
            result.refinement_count.to(torch.int64) - 1, min=0
        )
        refinements = previous_refinements + torch.cumsum(
            additional_refinements, dim=0
        ).detach().cpu().numpy()

        for index in range(chunk_rounds):
            history["communication_round"].append(
                int(result.communication_rounds[index])
            )
            history["train_objective"].append(
                float(result.function_value[index])
            )
            history["stationarity_norm"].append(
                float(result.subgradient[index])
            )
            history["test_accuracy"].append(linear_probe_accuracy(
                result.iterate_point[index],
                test_features,
                test_labels,
                settings.classes,
            ))
            history["cumulative_gradient_calls"].append(int(calls[index]))
            history["cumulative_local_iterations"].append(
                int(result.local_iterations[index])
            )
            history["cumulative_refinements"].append(int(refinements[index]))
            history["algorithm_time_seconds"].append(
                float(result.process_time[index])
            )

        completed_rounds += chunk_rounds
        save_checkpoint(
            checkpoint_path,
            settings,
            initial_point,
            algorithm_state,
            history,
            client_smoothness,
        )
        print(
            f"Round {completed_rounds}/{settings.communication_rounds}: "
            f"test_accuracy={history['test_accuracy'][-1]:.4f}, "
            f"objective={history['train_objective'][-1]:.6e}, "
            f"refinements={history['cumulative_refinements'][-1]}",
            flush=True,
        )

    return history, partitions, client_smoothness


def save_partition(
    path: Path,
    labels: torch.Tensor,
    partitions: list[np.ndarray],
    classes: int,
) -> None:
    labels_array = labels.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "client", "samples", *[f"class_{index}" for index in range(classes)]
        ])
        for client, indices in enumerate(partitions):
            counts = np.bincount(labels_array[indices], minlength=classes)
            writer.writerow([client, len(indices), *counts.tolist()])


def save_history(path: Path, history: dict[str, list[float | int]]) -> None:
    fields = list(history)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(zip(*(history[field] for field in fields)))


def plot_paper_figure(
    path: Path, history: dict[str, list[float | int]]
) -> None:
    rounds = np.asarray(history["communication_round"])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        rounds, history["test_accuracy"], color="#d62728", linewidth=2,
        label="iFedDR",
    )
    axes[0].set_xlabel("Number of communication rounds k")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("CIFAR10")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        rounds,
        history["cumulative_refinements"],
        color="#d62728",
        linewidth=2,
        label="iFedDR",
    )
    axes[1].set_xlabel("Number of communication rounds k")
    axes[1].set_ylabel("Total number of refinements")
    axes[1].set_title("CIFAR10")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(
    path: Path, history: dict[str, list[float | int]]
) -> None:
    rounds = np.asarray(history["communication_round"])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogy(
        rounds,
        np.maximum(history["train_objective"], 1e-30),
        linewidth=1.8,
    )
    axes[0].set_xlabel("Communication rounds")
    axes[0].set_ylabel("Regularized training objective")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].semilogy(
        rounds,
        np.maximum(history["stationarity_norm"], 1e-30),
        linewidth=1.8,
    )
    axes[1].set_xlabel("Communication rounds")
    axes[1].set_ylabel("Gradient norm")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def synthetic_features(
    settings: ExperimentSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(settings.seed)
    centers = torch.randn(
        settings.classes, settings.feature_dimension, generator=generator
    )

    def make(samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.arange(samples) % settings.classes
        noise = 0.25 * torch.randn(
            samples, settings.feature_dimension, generator=generator
        )
        return centers[labels] + noise, labels

    return (*make(60), *make(30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use a tiny synthetic convex problem without downloading CIFAR-10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if args.feature_batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid feature extraction settings.")

    settings = ExperimentSettings(
        communication_rounds=args.rounds,
        seed=args.seed,
        dtype=args.dtype,
    )
    device = torch.device("cpu") if args.smoke else resolve_device(args.device)
    if args.smoke:
        settings = replace(
            settings,
            clients=2,
            classes=3,
            feature_dimension=8,
            dirichlet_concentration=1.0,
            communication_rounds=min(args.rounds, 2),
            initial_local_updates=2,
            smoothness_power_iterations=10,
        )
    set_determinism(settings.seed)
    dtype = resolve_dtype(settings.dtype)
    output_directory = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        train_features, train_labels, test_features, test_labels = (
            synthetic_features(settings)
        )
    else:
        cache_path = args.feature_cache or (
            args.data_dir / "cifar10_resnet18_imagenet1k_v1_features.pt"
        )
        train_features, train_labels, test_features, test_labels = (
            extract_cifar10_features(
                args.data_dir,
                cache_path,
                device,
                args.feature_batch_size,
                args.workers,
                args.rebuild_features,
            )
        )

    train_features = train_features.to(device=device, dtype=dtype)
    test_features = test_features.to(device=device, dtype=dtype)
    train_labels = train_labels.to(device=device, dtype=torch.long)
    test_labels = test_labels.to(device=device, dtype=torch.long)
    if train_features.shape[1] != settings.feature_dimension:
        raise ValueError("The feature cache does not match the experiment settings.")

    print(
        f"Running iFedDR linear probing on {device}: "
        f"clients={settings.clients}, rounds={settings.communication_rounds}, "
        f"alpha={settings.dirichlet_concentration}, "
        f"sigma^2={settings.sigma_squared}",
        flush=True,
    )
    checkpoint_path = output_directory / "ifeddr_checkpoint.pt"
    history, partitions, client_smoothness = run_ifeddr(
        train_features,
        train_labels,
        test_features,
        test_labels,
        settings,
        checkpoint_path,
        args.checkpoint_every,
        args.resume,
    )

    history_path = output_directory / "ifeddr_history.csv"
    partition_path = output_directory / "client_label_distribution.csv"
    figure_path = output_directory / "figure3_ifeddr_linear_probing.png"
    diagnostic_path = output_directory / "ifeddr_diagnostics.png"
    save_history(history_path, history)
    save_partition(partition_path, train_labels, partitions, settings.classes)
    plot_paper_figure(figure_path, history)
    plot_diagnostics(diagnostic_path, history)
    metadata = {
        "settings": asdict(settings),
        "device": str(device),
        "paper_configuration": {
            "backbone": "ImageNet-pretrained torchvision ResNet18",
            "backbone_frozen": True,
            "adaptive_pooling": True,
            "input_resized": False,
            "full_client_participation": True,
            "deterministic_full_client_gradients": True,
            "client_objective_weighting": "uniform across clients",
        },
        "implementation_notes": {
            "partition": (
                "deterministic class-wise Dirichlet draw; it matches the paper's "
                "distributional setup but not an unpublished exact index draw"
            ),
            "smoothness": {
                "per_client_estimates": client_smoothness,
                "algorithm_value": max(client_smoothness),
                "rule": "1.01 * 0.5 * ||[X,1]||_2^2 / n + l2",
            },
        },
        "outputs": {
            "history": history_path.name,
            "partition": partition_path.name,
            "paper_figure": figure_path.name,
            "diagnostics": diagnostic_path.name,
            "checkpoint": checkpoint_path.name,
        },
    }
    with (output_directory / "metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Saved reproduction to {output_directory}", flush=True)


if __name__ == "__main__":
    main()
