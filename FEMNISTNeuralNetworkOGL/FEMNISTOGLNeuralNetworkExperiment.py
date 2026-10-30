"""FEMNIST-62 neural-network experiment with spatial overlapping group lasso.

Each writer is one naturally non-IID client.  The experiment selects a fixed
set of writers, merges any pre-existing LEAF train/test shards, and makes a
reproducible within-writer train/test split.  Local empirical means are scaled
by ``N * n_i / sum_j n_j`` so all five optimizers retain the theoretical
``(1/N) sum_i f_i + g`` form while the resulting smooth loss weights every
training example equally.

The overlapping-group proximal operator lives in
``Algorithms.ProximalOperator`` and uses matrix-free dual FISTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector

from Algorithms.REC_FedDyn import REC_FedDyn
from Algorithms.FedDR import FedDR
from Algorithms.FedNMap import FedNMap
from Algorithms.ProximalOperator import OverlappingGroupLasso
from Algorithms.Zhang import Zhang
from Algorithms.iFedDR import iFedDR
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

# ======================================================================
# Experiment settings -- edit parameters in this section
# ======================================================================

CLIENTS_NUMBER = 200
SAMPLING_NUMBERS = (200, 100, 50)
CLASSES_NUMBER = 62
TEST_FRACTION = 0.20
MINIMUM_WRITER_SAMPLES = 10

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
HIDDEN_UNITS = 128
ACTIVATION = "softplus"  # Change to "relu" for the ReLU experiment.
SOFTPLUS_BETA = 1.0

ITERATION_BUDGET = 200
LOCAL_UPDATES = 30
GRADIENT_BATCH_SIZE = 256
EVALUATION_BATCH_SIZE = 1024
EVALUATION_INTERVAL = 1

OGL_REGULARIZATION = 0.10
OGL_PATCH_SIZE = 4
OGL_STRIDE = 2
OGL_MEMBERSHIP_NORMALIZATION = True
OGL_PROXIMAL_ACCURACY = 1e-6
OGL_PROXIMAL_MAX_ITERATIONS = 200
OGL_PROXIMAL_WARM_START = False
INACTIVE_GROUP_TOLERANCE = 1e-6

DATA_SEED = 7
MODEL_SEED = 123
ALGORITHM_SEEDS = (7, 8, 9, 10, 11)
ALGORITHM_NAMES = (
    "REC_FedDyn",
    "FedDR",
    "iFedDR",
    "FedNMap",
    "Zhang",
)
PLOT_LABELS = {"REC_FedDyn": "REC_FedDyn"}

FIGURE_DPI = 300
ERROR_FLOOR = 1e-16

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = Path("")  # Fill in the directory containing the FEMNIST data.
RESULT_DIRECTORY = SCRIPT_DIRECTORY / "Results" / "FEMNISTOGLNeuralNetwork"


# ======================================================================
# Algorithm parameters -- kept here deliberately for convenient tuning
# ======================================================================

REC_FED_DYN_PENALTY_PARAMETER = 1.0 / 10.0
REC_FED_DYN_SIGMA = 0.20
REC_FED_DYN_MAX_LOCAL_ITERATIONS = 30
REC_FED_DYN_LOCAL_LEARNING_RATE = 1e-2

FEDDR_PROXIMAL_PARAMETER = 5.0
FEDDR_RELAXATION = 1.0
FEDDR_LOCAL_UPDATES = LOCAL_UPDATES
FEDDR_LOCAL_LEARNING_RATE = 1e-2

IFEDDR_PROXIMAL_PARAMETER = 0.1
IFEDDR_RELAXATION = 1.0
IFEDDR_SIGMA_SQUARED = 0.99
IFEDDR_INITIAL_LOCAL_UPDATES = LOCAL_UPDATES
IFEDDR_REFINEMENT_FACTOR = 2.0
IFEDDR_MAX_REFINEMENTS = 100
IFEDDR_MAX_LOCAL_ITERATIONS = 100
IFEDDR_LOCAL_LEARNING_RATE = 1e-2

FEDNMAP_PROXIMAL_PARAMETER = 0.1
FEDNMAP_LOCAL_UPDATES = LOCAL_UPDATES
FEDNMAP_LOCAL_LEARNING_RATE = 0.1 / FEDNMAP_LOCAL_UPDATES
FEDNMAP_SERVER_LEARNING_RATE = 1.0

ZHANG_LOCAL_UPDATES = LOCAL_UPDATES
ZHANG_SERVER_LEARNING_RATE = 1.0
ZHANG_LOCAL_LEARNING_RATE = (
    0.1 / ZHANG_LOCAL_UPDATES
)


# ======================================================================
# Data and model
# ======================================================================

@dataclass(slots=True)
class ClientData:
    writer: str
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    objective_scale: float

    @property
    def train_samples(self) -> int:
        return int(self.train_labels.numel())

    @property
    def test_samples(self) -> int:
        return int(self.test_labels.numel())


@dataclass(slots=True)
class History:
    communication: np.ndarray
    loss: np.ndarray
    train_cross_entropy: np.ndarray
    ogl_penalty: np.ndarray
    train_accuracy: np.ndarray
    test_accuracy: np.ndarray
    cumulative_oracles: np.ndarray
    process_time: np.ndarray
    inactive_group_fraction: np.ndarray
    refinement_count: np.ndarray | None = None
    relative_error_satisfied: np.ndarray | None = None
    selected_clients: np.ndarray | None = None
    local_oracles: np.ndarray | None = None
    local_criterion_satisfied: np.ndarray | None = None


class FEMNISTNet(nn.Module):
    """The 784 -> 128 -> 62 classifier used by the experiment."""

    def __init__(self) -> None:
        super().__init__()
        if ACTIVATION == "softplus":
            activation: nn.Module = nn.Softplus(beta=SOFTPLUS_BETA)
        elif ACTIVATION == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError("ACTIVATION must be 'softplus' or 'relu'.")
        self.layers = nn.Sequential(
            nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, HIDDEN_UNITS),
            activation,
            nn.Linear(HIDDEN_UNITS, CLASSES_NUMBER),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def initialize_model(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def parameter_unpacker(model: nn.Module):
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
            raise ValueError("The flat point does not match the network layout.")
        return parameters

    return unpack


def make_ogl_prox(model: nn.Module) -> OverlappingGroupLasso:
    first_name, first_parameter = next(iter(model.named_parameters()))
    expected_shape = (HIDDEN_UNITS, IMAGE_HEIGHT * IMAGE_WIDTH)
    if first_name != "layers.0.weight" or tuple(first_parameter.shape) != expected_shape:
        raise ValueError(
            "The OGL parameter layout expects layers.0.weight to be first and "
            f"have shape {expected_shape}; found {first_name} "
            f"with shape {tuple(first_parameter.shape)}."
        )
    return OverlappingGroupLasso(
        weight_start=0,
        hidden_units=HIDDEN_UNITS,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        patch_size=OGL_PATCH_SIZE,
        stride=OGL_STRIDE,
        accuracy=OGL_PROXIMAL_ACCURACY,
        maximum_iterations=OGL_PROXIMAL_MAX_ITERATIONS,
        membership_normalization=OGL_MEMBERSHIP_NORMALIZATION,
        warm_start=OGL_PROXIMAL_WARM_START,
    )


def _payload_users(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    users = payload.get("user_data")
    if not isinstance(users, dict):
        raise ValueError(f"{path} does not contain a user_data dictionary.")
    return users


def discover_femnist_json_files(directory: Path) -> list[Path]:
    """Locate raw or already-split LEAF JSON files.

    Existing train/test files are intentionally merged later because this
    experiment creates its own within-writer split.
    """

    directory = Path(directory)
    flat_pairs = (
        (directory / "mytrain.json", directory / "mytest.json"),
        (directory / "data" / "mytrain.json", directory / "data" / "mytest.json"),
    )
    for train_path, test_path in flat_pairs:
        if train_path.is_file() and test_path.is_file():
            return [train_path, test_path]

    for raw_directory in (
        directory / "all_data",
        directory / "data" / "all_data",
    ):
        files = sorted(raw_directory.glob("*.json"))
        if files:
            return files

    for base in (directory, directory / "data"):
        train_directory = base / "train"
        test_directory = base / "test"
        train_files = sorted(train_directory.glob("*.json"))
        test_files = sorted(test_directory.glob("*.json"))
        if train_files and test_files:
            return train_files + test_files

    raise FileNotFoundError(
        "Could not find LEAF FEMNIST JSON files. Expected all_data/*.json, "
        "train/*.json plus test/*.json, or mytrain.json plus mytest.json "
        f"below {directory}."
    )


def discover_femnist_files(directory: Path) -> tuple[str, list[Path]]:
    """Locate the mounted Flower Parquet file or legacy LEAF JSON files."""

    directory = Path(directory)
    parquet_candidates = (
        directory / "train-00000-of-00001.parquet",
        directory / "data" / "train-00000-of-00001.parquet",
    )
    for path in parquet_candidates:
        if path.is_file():
            return "parquet", [path]

    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted((directory / "data").glob("*.parquet"))
    if parquet_files:
        return "parquet", parquet_files

    try:
        return "json", discover_femnist_json_files(directory)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Could not find FEMNIST data. Expected "
            "train-00000-of-00001.parquet or LEAF JSON files below "
            f"{directory}."
        ) from error


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "Reading the mounted FEMNIST Parquet file requires pyarrow."
        ) from error
    return parquet


def _parquet_writer_counts(files: list[Path]) -> dict[str, int]:
    parquet = _import_pyarrow_parquet()
    counts: Counter[str] = Counter()
    for path in files:
        parquet_file = parquet.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        required = {"writer_id", "image", "character"}
        missing = required - schema_names
        if missing:
            raise ValueError(
                f"{path} is missing FEMNIST columns: {sorted(missing)}."
            )
        for batch in parquet_file.iter_batches(columns=["writer_id"]):
            counts.update(str(writer) for writer in batch.column(0).to_pylist())
    return dict(counts)


def _decode_parquet_image(value: Any, parquet_path: Path) -> torch.Tensor:
    """Decode Hugging Face's Image feature into one flattened float tensor."""

    if isinstance(value, dict):
        encoded = value.get("bytes")
        image_path = value.get("path")
        if encoded is not None:
            source: Any = io.BytesIO(encoded)
        elif image_path:
            candidate = Path(image_path)
            source = candidate if candidate.is_absolute() else parquet_path.parent / candidate
        else:
            raise ValueError("A Parquet image has neither bytes nor a path.")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(value))
    else:
        array = np.asarray(value)
        if array.size != IMAGE_HEIGHT * IMAGE_WIDTH:
            raise ValueError(f"Unexpected FEMNIST image shape {array.shape}.")
        image = torch.as_tensor(array, dtype=torch.float32).reshape(-1)
        return image / 255.0 if float(image.max()) > 1.0 else image

    with Image.open(source) as pil_image:
        array = np.asarray(pil_image.convert("L"), dtype=np.float32).copy()
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected FEMNIST image shape {array.shape}.")
    return torch.from_numpy(array.reshape(-1)).div_(255.0)


def _load_selected_parquet_records(
    files: list[Path], selected_writers: list[str]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    parquet = _import_pyarrow_parquet()
    selected = set(selected_writers)
    images: dict[str, list[torch.Tensor]] = {writer: [] for writer in selected}
    labels: dict[str, list[int]] = {writer: [] for writer in selected}

    for path in files:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            columns=["writer_id", "image", "character"]
        ):
            payload = batch.to_pydict()
            for writer_value, image_value, label_value in zip(
                payload["writer_id"], payload["image"], payload["character"]
            ):
                writer = str(writer_value)
                if writer not in selected:
                    continue
                images[writer].append(_decode_parquet_image(image_value, path))
                labels[writer].append(int(label_value))

    records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for writer in selected_writers:
        if not images[writer]:
            raise ValueError(f"Selected writer {writer!r} was not loaded.")
        records[writer] = (
            torch.stack(images[writer]).contiguous(),
            torch.tensor(labels[writer], dtype=torch.long),
        )
    return records


def _writer_counts(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        for writer, values in _payload_users(path).items():
            labels = values.get("y")
            if not isinstance(labels, list):
                raise ValueError(f"Writer {writer!r} in {path} has invalid labels.")
            counts[writer] = counts.get(writer, 0) + len(labels)
    return counts


def _load_selected_records(
    files: list[Path], selected_writers: list[str]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    selected = set(selected_writers)
    image_chunks: dict[str, list[torch.Tensor]] = {writer: [] for writer in selected}
    label_chunks: dict[str, list[torch.Tensor]] = {writer: [] for writer in selected}
    for path in files:
        for writer, values in _payload_users(path).items():
            if writer not in selected:
                continue
            if "x" not in values or "y" not in values:
                raise ValueError(f"Writer {writer!r} in {path} is missing x or y.")
            images = torch.as_tensor(values["x"], dtype=torch.float32)
            labels = torch.as_tensor(values["y"], dtype=torch.long).reshape(-1)
            images = images.reshape(images.shape[0], -1)
            if images.shape != (labels.numel(), IMAGE_HEIGHT * IMAGE_WIDTH):
                raise ValueError(
                    f"Writer {writer!r} has images {tuple(images.shape)} and "
                    f"{labels.numel()} labels."
                )
            if images.numel() and float(images.max()) > 1.0:
                images = images / 255.0
            image_chunks[writer].append(images.contiguous())
            label_chunks[writer].append(labels.contiguous())

    records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for writer in selected_writers:
        if not image_chunks[writer]:
            raise ValueError(f"Selected writer {writer!r} was not loaded.")
        images = torch.cat(image_chunks[writer])
        labels = torch.cat(label_chunks[writer])
        records[writer] = (images, labels)
    return records


def load_femnist_clients(
    directory: Path,
    clients_number: int = CLIENTS_NUMBER,
    test_fraction: float = TEST_FRACTION,
    data_seed: int = DATA_SEED,
    minimum_writer_samples: int = MINIMUM_WRITER_SAMPLES,
) -> list[ClientData]:
    if clients_number <= 0 or minimum_writer_samples < 2:
        raise ValueError("clients_number must be positive and minimum samples at least two.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one.")

    source_format, files = discover_femnist_files(directory)
    counts = (
        _parquet_writer_counts(files)
        if source_format == "parquet"
        else _writer_counts(files)
    )
    eligible = sorted(
        writer for writer, count in counts.items()
        if count >= minimum_writer_samples
    )
    if len(eligible) < clients_number:
        raise ValueError(
            f"Only {len(eligible)} writers have at least "
            f"{minimum_writer_samples} samples; {clients_number} are required."
        )

    selection_rng = np.random.default_rng(data_seed)
    selected_writers = sorted(
        selection_rng.choice(eligible, size=clients_number, replace=False).tolist()
    )
    records = (
        _load_selected_parquet_records(files, selected_writers)
        if source_format == "parquet"
        else _load_selected_records(files, selected_writers)
    )

    split_records: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for client_index, writer in enumerate(selected_writers):
        images, labels = records[writer]
        minimum, maximum = int(labels.min()), int(labels.max())
        if minimum < 0 or maximum >= CLASSES_NUMBER:
            raise ValueError(
                f"Writer {writer!r} has labels [{minimum}, {maximum}], but "
                f"the experiment requires labels in [0, {CLASSES_NUMBER - 1}]."
            )
        rng = np.random.default_rng(data_seed + 104729 * (client_index + 1))
        permutation = torch.as_tensor(rng.permutation(labels.numel()), dtype=torch.long)
        test_count = int(round(test_fraction * labels.numel()))
        test_count = min(max(test_count, 1), labels.numel() - 1)
        test_indices = permutation[:test_count]
        train_indices = permutation[test_count:]
        split_records.append((
            writer,
            images[train_indices].contiguous(),
            labels[train_indices].contiguous(),
            images[test_indices].contiguous(),
            labels[test_indices].contiguous(),
        ))

    total_training_samples = sum(record[2].numel() for record in split_records)
    clients = [
        ClientData(
            writer=writer,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            objective_scale=(
                clients_number * train_labels.numel() / total_training_samples
            ),
        )
        for writer, train_images, train_labels, test_images, test_labels
        in split_records
    ]
    return clients


def make_scaled_client_objective(
    model: nn.Module,
    unpack,
    client: ClientData,
    gradient_batch_size: int = GRADIENT_BATCH_SIZE,
):
    """Return the exact scaled local mean loss and gradient."""

    if gradient_batch_size <= 0:
        raise ValueError("gradient_batch_size must be positive.")
    sample_count = client.train_samples
    scale = float(client.objective_scale) / sample_count

    def objective(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_loss = point.new_zeros(())
        total_gradient = torch.zeros_like(point)
        for start in range(0, sample_count, gradient_batch_size):
            stop = min(start + gradient_batch_size, sample_count)
            images = client.train_images[start:stop].to(point.device)
            labels = client.train_labels[start:stop].to(point.device)
            variable = point.detach().requires_grad_(True)
            logits = functional_call(model, unpack(variable), (images,))
            batch_loss = F.cross_entropy(logits, labels, reduction="sum")
            batch_gradient = torch.autograd.grad(batch_loss, variable)[0]
            total_loss += batch_loss.detach()
            total_gradient += batch_gradient.detach()
        return total_loss * scale, total_gradient * scale

    return objective


@torch.no_grad()
def _split_metrics(
    model: nn.Module,
    unpack,
    point: torch.Tensor,
    clients: list[ClientData],
    split: str,
    evaluation_batch_size: int,
) -> tuple[float, float]:
    parameters = unpack(point)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for client in clients:
        if split == "train":
            images, labels = client.train_images, client.train_labels
        elif split == "test":
            images, labels = client.test_images, client.test_labels
        else:
            raise ValueError("split must be 'train' or 'test'.")
        for start in range(0, labels.numel(), evaluation_batch_size):
            stop = min(start + evaluation_batch_size, labels.numel())
            batch_images = images[start:stop].to(point.device)
            batch_labels = labels[start:stop].to(point.device)
            logits = functional_call(model, parameters, (batch_images,))
            total_loss += float(F.cross_entropy(
                logits, batch_labels, reduction="sum"
            ))
            total_correct += int((logits.argmax(1) == batch_labels).sum())
            total_samples += batch_labels.numel()
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_point(
    model: nn.Module,
    unpack,
    point: torch.Tensor,
    clients: list[ClientData],
    prox: OverlappingGroupLasso,
) -> tuple[float, float, float, float, float, float]:
    train_ce, train_accuracy = _split_metrics(
        model, unpack, point, clients, "train", EVALUATION_BATCH_SIZE
    )
    _, test_accuracy = _split_metrics(
        model, unpack, point, clients, "test", EVALUATION_BATCH_SIZE
    )
    penalty = float(prox(point, 1.0, OGL_REGULARIZATION, True))
    inactive = prox.inactive_group_fraction(
        point, INACTIVE_GROUP_TOLERANCE
    )
    return (
        train_ce + penalty,
        train_ce,
        penalty,
        train_accuracy,
        test_accuracy,
        inactive,
    )


# ======================================================================
# Algorithms
# ======================================================================

def algorithm_settings(name: str):
    if name == "REC_FedDyn":
        return REC_FedDyn, dict(
            PenaltyParameter=REC_FED_DYN_PENALTY_PARAMETER,
            Sigma=REC_FED_DYN_SIGMA,
            MaxLocalIterations=REC_FED_DYN_MAX_LOCAL_ITERATIONS,
            LocalLearningRate=REC_FED_DYN_LOCAL_LEARNING_RATE,
        )
    if name == "FedDR":
        return FedDR, dict(
            ProximalParameter=FEDDR_PROXIMAL_PARAMETER,
            RelaxationParameter=FEDDR_RELAXATION,
            LocalUpdates=FEDDR_LOCAL_UPDATES,
            LocalLearningRate=FEDDR_LOCAL_LEARNING_RATE,
        )
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
        )
    if name == "FedNMap":
        return FedNMap, dict(
            ProximalParameter=FEDNMAP_PROXIMAL_PARAMETER,
            LocalLearningRate=FEDNMAP_LOCAL_LEARNING_RATE,
            ServerLearningRate=FEDNMAP_SERVER_LEARNING_RATE,
            LocalUpdates=FEDNMAP_LOCAL_UPDATES,
        )
    if name == "Zhang":
        return Zhang, dict(
            LocalLearningRate=ZHANG_LOCAL_LEARNING_RATE,
            ServerLearningRate=ZHANG_SERVER_LEARNING_RATE,
            LocalUpdates=ZHANG_LOCAL_UPDATES,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def _evaluation_rounds() -> np.ndarray:
    rounds = list(range(0, ITERATION_BUDGET + 1, EVALUATION_INTERVAL))
    if rounds[-1] != ITERATION_BUDGET:
        rounds.append(ITERATION_BUDGET)
    return np.asarray(rounds, dtype=np.int64)


def run_algorithm(
    name: str,
    algorithm_seed: int,
    sampling_number: int,
    initial_point: torch.Tensor,
    functions: list,
    model: nn.Module,
    unpack,
    clients: list[ClientData],
) -> tuple[History, dict[str, float | int]]:
    prox = make_ogl_prox(model)
    common = dict(
        InitialPoint=initial_point,
        IterationBudget=ITERATION_BUDGET,
        ProxFunction=prox,
        ClientsNumber=CLIENTS_NUMBER,
        SamplingNumber=sampling_number,
        SmoothL=0.0,
        ConvexMu=0.0,
        Constant=OGL_REGULARIZATION,
        RandomSeed=algorithm_seed,
    )
    algorithm_class, settings = algorithm_settings(name)
    algorithm = algorithm_class(**common)
    result = algorithm.go_torch(
        Function=functions,
        RecordStatistics=False,
        **settings,
    )

    all_points = torch.cat((initial_point[None], result.iterate_point))
    communication = _evaluation_rounds()
    evaluated = [
        evaluate_point(model, unpack, all_points[round_index], clients, prox)
        for round_index in communication
    ]
    values = np.asarray(evaluated, dtype=np.float64)
    cumulative_oracles = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(result.oracle_calling.detach().cpu().numpy()),
    ))
    process_time = np.concatenate((
        np.zeros(1, dtype=np.float64),
        result.process_time.detach().cpu().numpy(),
    ))

    refinement_count = None
    relative_error_satisfied = None
    if name == "iFedDR":
        refinement_count = np.concatenate((
            np.zeros(1, dtype=np.int32),
            result.refinement_count.detach().cpu().numpy(),
        ))[communication]
        relative_error_satisfied = np.concatenate((
            np.ones(1, dtype=np.bool_),
            np.logical_not(result.index.detach().cpu().numpy()),
        ))[communication]

    history = History(
        communication=communication,
        loss=values[:, 0],
        train_cross_entropy=values[:, 1],
        ogl_penalty=values[:, 2],
        train_accuracy=values[:, 3],
        test_accuracy=values[:, 4],
        cumulative_oracles=cumulative_oracles[communication],
        process_time=process_time[communication],
        inactive_group_fraction=values[:, 5],
        refinement_count=refinement_count,
        relative_error_satisfied=relative_error_satisfied,
        selected_clients=(
            result.selected_clients.detach().cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
        local_oracles=(
            result.local_oracle_calling.detach().cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
        local_criterion_satisfied=(
            result.local_criterion_satisfied.detach().cpu().numpy()
            if name == "REC_FedDyn" else None
        ),
    )
    proximal_summary = {
        "proximal_calls": prox.proximal_calls,
        "total_dual_iterations": prox.total_dual_iterations,
        "last_dual_iterations": prox.last_iterations,
        "last_duality_gap": prox.last_gap,
    }
    return history, proximal_summary


# ======================================================================
# Persistence and plots
# ======================================================================

def settings_key(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def problem_settings(
    clients: list[ClientData], sampling_number: int
) -> dict[str, Any]:
    return {
        "format": "femnist_ogl_problem_v1",
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": sampling_number,
        "client_partitioning": "natural_writer_id",
        "classes_number": CLASSES_NUMBER,
        "test_fraction": TEST_FRACTION,
        "minimum_writer_samples": MINIMUM_WRITER_SAMPLES,
        "train_samples": sum(client.train_samples for client in clients),
        "test_samples": sum(client.test_samples for client in clients),
        "writers": [client.writer for client in clients],
        "hidden_units": HIDDEN_UNITS,
        "activation": ACTIVATION,
        "softplus_beta": SOFTPLUS_BETA,
        "local_objective_scaling": "N*n_i/sum_j(n_j)",
        "aggregation": "uniform_client_average",
        "ogl_regularization": OGL_REGULARIZATION,
        "ogl_patch_size": OGL_PATCH_SIZE,
        "ogl_stride": OGL_STRIDE,
        "ogl_membership_normalization": OGL_MEMBERSHIP_NORMALIZATION,
        "data_seed": DATA_SEED,
    }


def run_settings(
    name: str,
    problem_key: str,
    algorithm_seed: int,
    model_seed: int,
    sampling_number: int,
) -> dict[str, Any]:
    _, parameters = algorithm_settings(name)
    return {
        "format": "femnist_ogl_algorithm_v1",
        "problem_key": problem_key,
        "algorithm": name,
        "algorithm_seed": algorithm_seed,
        "model_seed": model_seed,
        "sampling_number": sampling_number,
        "iteration_budget": ITERATION_BUDGET,
        "gradient_batch_size": GRADIENT_BATCH_SIZE,
        "evaluation_interval": EVALUATION_INTERVAL,
        "algorithm_parameters": parameters,
        "proximal_accuracy": OGL_PROXIMAL_ACCURACY,
        "proximal_max_iterations": OGL_PROXIMAL_MAX_ITERATIONS,
        "proximal_warm_start": OGL_PROXIMAL_WARM_START,
        "inactive_group_tolerance": INACTIVE_GROUP_TOLERANCE,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def save_clients(clients: list[ClientData], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "client_index", "writer", "train_samples", "test_samples",
            "objective_scale",
        ))
        for index, client in enumerate(clients):
            writer.writerow((
                index,
                client.writer,
                client.train_samples,
                client.test_samples,
                f"{client.objective_scale:.12e}",
            ))


def save_history(history: History, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    refinements = (
        history.refinement_count
        if history.refinement_count is not None
        else np.full(history.communication.shape, -1, dtype=np.int32)
    )
    condition = (
        history.relative_error_satisfied.astype(np.int8)
        if history.relative_error_satisfied is not None
        else np.full(history.communication.shape, -1, dtype=np.int8)
    )
    np.savetxt(
        directory / "metrics.csv",
        np.column_stack((
            history.communication,
            history.loss,
            history.train_cross_entropy,
            history.ogl_penalty,
            history.train_accuracy,
            history.test_accuracy,
            history.cumulative_oracles,
            history.process_time,
            history.inactive_group_fraction,
            refinements,
            condition,
        )),
        delimiter=",",
        fmt=(
            "%d", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e",
            "%d", "%.10e", "%.10e", "%d", "%d",
        ),
        header=(
            "communication,regularized_train_loss,train_cross_entropy,"
            "ogl_penalty,train_accuracy,test_accuracy,"
            "cumulative_gradient_oracles,algorithm_time_seconds,"
            "inactive_group_fraction,refinement_count,"
            "relative_error_condition_satisfied"
        ),
        comments="",
    )
    if history.selected_clients is not None:
        rounds = np.repeat(
            np.arange(1, history.selected_clients.shape[0] + 1),
            history.selected_clients.shape[1],
        )
        np.savetxt(
            directory / "local_solves.csv",
            np.column_stack((
                rounds,
                history.selected_clients.ravel(),
                history.local_oracles.ravel(),
                history.local_criterion_satisfied.ravel(),
            )),
            delimiter=",",
            fmt="%d",
            header="communication,client,gradient_oracles,criterion_satisfied",
            comments="",
        )


def load_history(path: Path) -> History:
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    refinements = table[:, 9].astype(np.int32)
    condition = table[:, 10].astype(np.int8)
    has_refinements = bool(np.any(refinements >= 0))
    return History(
        communication=table[:, 0].astype(np.int64),
        loss=table[:, 1],
        train_cross_entropy=table[:, 2],
        ogl_penalty=table[:, 3],
        train_accuracy=table[:, 4],
        test_accuracy=table[:, 5],
        cumulative_oracles=table[:, 6].astype(np.int64),
        process_time=table[:, 7],
        inactive_group_fraction=table[:, 8],
        refinement_count=refinements if has_refinements else None,
        relative_error_satisfied=(condition.astype(bool) if has_refinements else None),
    )


def complete_cached_history(
    directory: Path, settings: dict[str, Any]
) -> History | None:
    settings_path = directory / "settings.json"
    metrics_path = directory / "metrics.csv"
    if not settings_path.exists() or not metrics_path.exists():
        return None
    with settings_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != settings:
            return None
    history = load_history(metrics_path)
    if history.communication[-1] != ITERATION_BUDGET:
        return None
    return history


def history_mean_std(
    histories: list[History], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    communication = histories[0].communication
    if any(
        not np.array_equal(history.communication, communication)
        for history in histories[1:]
    ):
        raise ValueError("Seed histories do not share a communication axis.")
    values = np.stack([getattr(history, field) for history in histories])
    return communication, values.mean(axis=0), values.std(axis=0)


def save_aggregate_tables(
    histories: dict[str, list[History]], result_directory: Path
) -> None:
    for name, seed_histories in histories.items():
        if not seed_histories:
            continue
        columns: list[np.ndarray] = [seed_histories[0].communication]
        header = ["communication"]
        for field in (
            "loss",
            "train_accuracy",
            "test_accuracy",
            "cumulative_oracles",
            "process_time",
            "inactive_group_fraction",
        ):
            _, mean, standard_deviation = history_mean_std(seed_histories, field)
            columns.extend((mean, standard_deviation))
            header.extend((f"{field}_mean", f"{field}_std"))
        np.savetxt(
            result_directory / f"{name}_mean_std.csv",
            np.column_stack(columns),
            delimiter=",",
            fmt="%.10e",
            header=",".join(header),
            comments="",
        )


def _plot_with_band(
    histories: dict[str, list[History]],
    field: str,
    vertical_label: str,
    path: Path,
    sampling_number: int,
    logarithmic: bool = False,
) -> None:
    figure = plt.figure(figsize=(8.4, 5.6))
    for name, seed_histories in histories.items():
        if not seed_histories:
            continue
        communication, mean, standard_deviation = history_mean_std(
            seed_histories, field
        )
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
    plt.xlabel("Communication rounds")
    plt.ylabel(vertical_label)
    plt.title(
        f"FEMNIST-62 OGL: sampled clients = "
        f"{sampling_number}/{CLIENTS_NUMBER}"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_plots(
    histories: dict[str, list[History]],
    result_directory: Path,
    sampling_number: int,
) -> None:
    _plot_with_band(
        histories,
        "loss",
        "Regularized training loss",
        result_directory / "loss_decay_vs_communication.pdf",
        sampling_number,
        logarithmic=True,
    )
    _plot_with_band(
        histories,
        "test_accuracy",
        "Test accuracy",
        result_directory / "test_accuracy_vs_communication.pdf",
        sampling_number,
    )
    _plot_with_band(
        histories,
        "train_accuracy",
        "Training accuracy",
        result_directory / "train_accuracy_vs_communication.pdf",
        sampling_number,
    )
    _plot_with_band(
        histories,
        "inactive_group_fraction",
        "Inactive group fraction",
        result_directory / "inactive_groups_vs_communication.pdf",
        sampling_number,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--result-directory", type=Path, default=RESULT_DIRECTORY)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed cached runs and run every algorithm again.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def run_sampling_experiment(
    arguments: argparse.Namespace,
    device: torch.device,
    clients: list[ClientData],
    sampling_number: int,
) -> None:
    problem = problem_settings(clients, sampling_number)
    problem_key = settings_key(problem)
    result_directory = (
        arguments.result_directory
        / f"sample_{sampling_number}"
        / f"problem_{problem_key}"
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    save_json(result_directory / "problem_settings.json", problem)
    save_clients(clients, result_directory / "clients.csv")

    print(
        f"\nSampling case={sampling_number}/{CLIENTS_NUMBER}; "
        f"results={result_directory}",
        flush=True,
    )

    histories: dict[str, list[History]] = {name: [] for name in ALGORITHM_NAMES}
    for experiment_seed in ALGORITHM_SEEDS:
        current_model_seed = MODEL_SEED + experiment_seed
        model = FEMNISTNet().to(device)
        initialize_model(model, current_model_seed)
        unpack = parameter_unpacker(model)
        initial_point = parameters_to_vector(model.parameters()).detach()
        functions = [
            make_scaled_client_objective(model, unpack, client)
            for client in clients
        ]

        print(
            f"Experiment seed={experiment_seed}; model seed={current_model_seed}",
            flush=True,
        )
        for name in ALGORITHM_NAMES:
            settings = run_settings(
                name,
                problem_key,
                experiment_seed,
                current_model_seed,
                sampling_number,
            )
            run_directory = (
                result_directory / "algorithms" / name
                / f"seed_{experiment_seed}" / f"run_{settings_key(settings)}"
            )
            cached = None if arguments.restart else complete_cached_history(
                run_directory, settings
            )
            if cached is not None:
                histories[name].append(cached)
                print(f"  loaded {name}", flush=True)
                continue

            print(f"  running {name}...", flush=True)
            try:
                history, proximal_summary = run_algorithm(
                    name,
                    experiment_seed,
                    sampling_number,
                    initial_point,
                    functions,
                    model,
                    unpack,
                    clients,
                )
                save_json(run_directory / "settings.json", settings)
                save_json(
                    run_directory / "proximal_summary.json", proximal_summary
                )
                save_history(history, run_directory)
                histories[name].append(history)
                print(
                    f"    loss={history.loss[-1]:.6f}; "
                    f"test_acc={history.test_accuracy[-1]:.4f}; "
                    f"inactive={history.inactive_group_fraction[-1]:.3%}; "
                    f"oracles={history.cumulative_oracles[-1]}; "
                    f"time={history.process_time[-1]:.2f}s",
                    flush=True,
                )
            except Exception:
                traceback.print_exc()
                print(
                    f"    failed {name}, seed={experiment_seed}; continuing...",
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
    save_aggregate_tables(histories, result_directory)
    save_plots(histories, result_directory, sampling_number)
    print(f"Saved results to {result_directory}", flush=True)


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    if any(
        sampling_number < 1 or sampling_number > CLIENTS_NUMBER
        for sampling_number in SAMPLING_NUMBERS
    ):
        raise ValueError(
            "Every sampling number must lie between 1 and CLIENTS_NUMBER."
        )

    clients = load_femnist_clients(arguments.data_directory)
    observed_labels = sorted({
        int(label)
        for client in clients
        for labels in (client.train_labels, client.test_labels)
        for label in torch.unique(labels).tolist()
    })
    if observed_labels != list(range(CLASSES_NUMBER)):
        print(
            "Warning: selected writers do not contain all 62 labels; "
            f"observed labels={observed_labels}",
            flush=True,
        )

    print(
        f"Device={device}; writers={len(clients)}; "
        f"train_samples={sum(c.train_samples for c in clients)}; "
        f"test_samples={sum(c.test_samples for c in clients)}; "
        f"classes={CLASSES_NUMBER}; activation={ACTIVATION}; "
        f"sampling_cases={SAMPLING_NUMBERS}",
        flush=True,
    )
    print(
        "Natural writer-ID partition (no synthetic heterogeneity); "
        "local scaling: a_i=N*n_i/sum(n_i); server aggregation: uniform; "
        f"OGL patches={OGL_PATCH_SIZE}x{OGL_PATCH_SIZE}, "
        f"stride={OGL_STRIDE}, lambda={OGL_REGULARIZATION:g}",
        flush=True,
    )

    for sampling_number in SAMPLING_NUMBERS:
        run_sampling_experiment(arguments, device, clients, sampling_number)

    from FEMNISTOGLExperimentReplot import replot_results

    replot_results(arguments.result_directory, list(SAMPLING_NUMBERS))


if __name__ == "__main__":
    main()
