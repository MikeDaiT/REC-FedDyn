"""Run the FedDR FEMNIST l1-composite full-gradient experiment.

The data/model/algorithm parameters follow the authors' Figure 5 script, while
the local solver deliberately uses the unchanged ``FedDR.py`` placed next to
this file. ``gradient_batch_size`` only controls memory: gradients from every
client sample are accumulated before a single local update is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector

from FedDR import FedDR
from ProximalOperator import l_1

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


# Figure 5 parameters from Supplement C.1 of the FedDR paper.
CLIENTS_NUMBER = 200
SAMPLING_NUMBER = 50
CLASSES_NUMBER = 26
HIDDEN_UNITS = 128
ITERATION_BUDGET = 200
LOCAL_UPDATES = 50
PROXIMAL_PARAMETER = 500.0
RELAXATION_PARAMETER = 1.95
L1_REGULARIZATION = 0.01
# User-selected full-gradient step-size sweep with eta fixed at the paper's 500.
LEARNING_RATES = (0.005, 0.01, 0.03, 0.05, 0.08)

MODEL_SEED = 123
ALGORITHM_SEED = 0
GRADIENT_BATCH_SIZE = 256
CHECKPOINT_ROUNDS = 10
FIGURE_DPI = 300

VALIDATION_DIRECTORY = SCRIPT_DIRECTORY
DATA_DIRECTORY = VALIDATION_DIRECTORY / "FEMNIST"
RESULT_DIRECTORY = (
    VALIDATION_DIRECTORY / "Results" / "FEMNISTFedDRComposite"
    / "eta500_local50_author_aggregation_metrics"
)
GOOGLE_DRIVE_FILE_ID = "1tCEcJgRJ8NdRo11UJZR6WSKMNdmox4GC"


@dataclass(slots=True)
class ClientData:
    """One writer's flattened FEMNIST examples, kept on CPU."""

    user: str
    images: torch.Tensor
    labels: torch.Tensor


@dataclass(slots=True)
class FederatedData:
    clients: list[str]
    train: list[ClientData]
    test: list[ClientData]


@dataclass(slots=True)
class ExperimentHistory:
    communication_round: list[int]
    train_loss: list[float]
    train_cross_entropy: list[float]
    l1_penalty: list[float]
    train_accuracy: list[float]
    test_accuracy: list[float]
    server_max_abs: list[float]
    server_nonzero_fraction: list[float]
    cumulative_gradient_oracles: list[int]
    algorithm_time_seconds: list[float]

    @classmethod
    def empty(cls) -> "ExperimentHistory":
        return cls([], [], [], [], [], [], [], [], [], [])


class FEMNISTNet(nn.Module):
    """The authors' 784 -> 128 -> 26 fully connected classifier."""

    def __init__(self, classes_number: int = CLASSES_NUMBER) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, classes_number),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def initialize_like_tensorflow(model: nn.Module, seed: int = MODEL_SEED) -> None:
    """Use Glorot-uniform weights and zero biases, as tf.layers.dense does."""

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
            raise ValueError("The flattened point does not match the model.")
        return parameters

    return unpack


def _find_leaf_sources(directory: Path) -> tuple[Path, Path]:
    candidates = (
        directory,
        directory / "data",
        directory / "FEMNIST" / "data",
        directory / "FEMNIST",
    )
    for candidate in candidates:
        flat_train = candidate / "mytrain.json"
        flat_test = candidate / "mytest.json"
        if flat_train.is_file() and flat_test.is_file():
            return flat_train, flat_test
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate / "train", candidate / "test"
    raise FileNotFoundError(
        "Could not find FEMNIST files mytrain.json and mytest.json, or LEAF "
        f"folders train and test, below {directory}. This exact "
        "writer-partitioned dataset is not installed by torchvision. Download "
        "the authors' ZIP from Google Drive file "
        f"{GOOGLE_DRIVE_FILE_ID} and pass its location with --data-directory."
    )


def _read_leaf_split(source: Path) -> dict[str, dict[str, Any]]:
    files = [source] if source.is_file() else sorted(source.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No LEAF JSON shards found at {source}.")

    users: dict[str, dict[str, Any]] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            shard = json.load(stream)
        shard_users = shard.get("user_data")
        if not isinstance(shard_users, dict):
            raise ValueError(f"{path} does not contain a user_data dictionary.")
        overlap = users.keys() & shard_users.keys()
        if overlap:
            duplicate = sorted(overlap)[0]
            raise ValueError(f"Duplicate user {duplicate!r} in {path}.")
        users.update(shard_users)
    return users


def _client_data(user: str, values: dict[str, Any]) -> ClientData:
    if "x" not in values or "y" not in values:
        raise ValueError(f"User {user!r} is missing x or y.")
    images = torch.as_tensor(values["x"], dtype=torch.float32)
    labels = torch.as_tensor(values["y"], dtype=torch.long)
    if images.ndim < 2:
        raise ValueError(f"User {user!r} images must include a sample dimension.")
    images = images.reshape(images.shape[0], -1).contiguous()
    labels = labels.reshape(-1).contiguous()
    if images.shape != (labels.numel(), 28 * 28):
        raise ValueError(
            f"User {user!r} has image shape {tuple(images.shape)} and "
            f"{labels.numel()} labels; expected N x 784."
        )
    if labels.numel() == 0:
        raise ValueError(f"User {user!r} has no examples.")
    if not torch.isfinite(images).all():
        raise ValueError(f"User {user!r} contains non-finite image values.")
    return ClientData(user, images, labels)


def load_femnist(
    directory: Path,
    expected_clients: int = CLIENTS_NUMBER,
    expected_classes: int = CLASSES_NUMBER,
    require_every_class: bool = False,
) -> FederatedData:
    """Load the authors' 200-writer data for their 26-output classifier.

    The released JSON files contain labels 0--9 even though the authors' model
    has 26 outputs.  Missing output classes are therefore valid by default.
    The range check still rejects the incompatible 62-class FEMNIST release.
    """

    train_source, test_source = _find_leaf_sources(Path(directory))
    train_users = _read_leaf_split(train_source)
    test_users = _read_leaf_split(test_source)
    if set(train_users) != set(test_users):
        train_only = len(set(train_users) - set(test_users))
        test_only = len(set(test_users) - set(train_users))
        raise ValueError(
            "Training and test client IDs differ "
            f"(train-only={train_only}, test-only={test_only})."
        )
    clients = sorted(train_users)
    if len(clients) != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} writer clients, found {len(clients)}. "
            "The 62-class TensorFlow Federated/LEAF FEMNIST release is not the "
            "dataset used by this FedDR experiment."
        )

    train = [_client_data(user, train_users[user]) for user in clients]
    test = [_client_data(user, test_users[user]) for user in clients]
    all_labels = torch.cat([client.labels for client in train + test])
    minimum, maximum = int(all_labels.min()), int(all_labels.max())
    if minimum < 0 or maximum >= expected_classes:
        raise ValueError(
            f"Expected labels in [0, {expected_classes - 1}], found "
            f"[{minimum}, {maximum}]."
        )
    if require_every_class:
        observed = set(map(int, torch.unique(all_labels).tolist()))
        expected = set(range(expected_classes))
        if observed != expected:
            raise ValueError(
                "The dataset does not contain exactly the expected label set; "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}."
            )
    return FederatedData(clients, train, test)


def make_full_gradient_objective(
    model: nn.Module,
    unpack,
    client: ClientData,
    gradient_batch_size: int = GRADIENT_BATCH_SIZE,
):
    """Return the exact client mean loss/gradient using memory-only chunks."""

    if gradient_batch_size <= 0:
        raise ValueError("gradient_batch_size must be positive.")
    sample_count = client.labels.numel()

    def objective(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_loss = point.new_zeros(())
        total_gradient = torch.zeros_like(point)
        for start in range(0, sample_count, gradient_batch_size):
            stop = min(start + gradient_batch_size, sample_count)
            images = client.images[start:stop].to(point.device)
            labels = client.labels[start:stop].to(point.device)
            variable = point.detach().requires_grad_(True)
            logits = functional_call(model, unpack(variable), (images,))
            batch_loss = F.cross_entropy(logits, labels, reduction="sum")
            batch_gradient = torch.autograd.grad(batch_loss, variable)[0]
            total_loss += batch_loss.detach()
            total_gradient += batch_gradient.detach()
        scale = 1.0 / sample_count
        return total_loss * scale, total_gradient * scale

    return objective


@torch.no_grad()
def evaluate_point(
    model: nn.Module,
    unpack,
    point: torch.Tensor,
    train: list[ClientData],
    test: list[ClientData],
    evaluation_batch_size: int,
) -> tuple[float, float, float, float, float]:
    """Compute the authors' sample-weighted train/test metrics."""

    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive.")
    if not bool(torch.isfinite(point).all()):
        raise FloatingPointError("The server model contains NaN or Inf values.")
    parameters = unpack(point)

    def split_metrics(split: list[ClientData]) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for client in split:
            count = client.labels.numel()
            for start in range(0, count, evaluation_batch_size):
                stop = min(start + evaluation_batch_size, count)
                images = client.images[start:stop].to(point.device)
                labels = client.labels[start:stop].to(point.device)
                logits = functional_call(model, parameters, (images,))
                total_loss += float(
                    F.cross_entropy(logits, labels, reduction="sum")
                )
                total_correct += int((logits.argmax(1) == labels).sum())
                total_samples += labels.numel()
        return total_loss / total_samples, total_correct / total_samples

    train_cross_entropy, train_accuracy = split_metrics(train)
    _, test_accuracy = split_metrics(test)
    regularizer = float(l_1(point, 1.0, L1_REGULARIZATION, True))
    return (
        train_cross_entropy + regularizer,
        train_cross_entropy,
        regularizer,
        train_accuracy,
        test_accuracy,
    )


@torch.no_grad()
def evaluate_client_models(
    model: nn.Module,
    unpack,
    local_points: torch.Tensor,
    server_point: torch.Tensor,
    train: list[ClientData],
    test: list[ClientData],
    evaluation_batch_size: int,
) -> tuple[float, float, float, float, float]:
    """Match the released code's sample-weighted client-local metrics."""

    if local_points.shape[0] != len(train) or len(train) != len(test):
        raise ValueError("Expected one local model per train/test client.")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive.")
    if not bool(torch.isfinite(local_points).all()) or not bool(
        torch.isfinite(server_point).all()
    ):
        raise FloatingPointError("A local or server model contains NaN or Inf.")

    def split_metrics(split: list[ClientData]) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for point, client in zip(local_points, split):
            parameters = unpack(point)
            count = client.labels.numel()
            for start in range(0, count, evaluation_batch_size):
                stop = min(start + evaluation_batch_size, count)
                images = client.images[start:stop].to(point.device)
                labels = client.labels[start:stop].to(point.device)
                logits = functional_call(model, parameters, (images,))
                total_loss += float(
                    F.cross_entropy(logits, labels, reduction="sum")
                )
                total_correct += int((logits.argmax(1) == labels).sum())
                total_samples += labels.numel()
        return total_loss / total_samples, total_correct / total_samples

    train_cross_entropy, train_accuracy = split_metrics(train)
    _, test_accuracy = split_metrics(test)
    regularizer = float(
        l_1(server_point, 1.0, L1_REGULARIZATION, True)
    )
    return (
        train_cross_entropy + regularizer,
        train_cross_entropy,
        regularizer,
        train_accuracy,
        test_accuracy,
    )


@torch.no_grad()
def parameter_diagnostics(point: torch.Tensor) -> tuple[float, float]:
    """Return maximum magnitude and exact nonzero fraction."""

    return (
        float(point.abs().max()),
        float(torch.count_nonzero(point)) / point.numel(),
    )


def _state_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_device(item, device) for item in value)
    return value


def _state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def _rate_name(learning_rate: float) -> str:
    return f"lr_{learning_rate:g}".replace(".", "p")


def experiment_signature(
    learning_rate: float,
    rounds: int,
    gradient_batch_size: int,
) -> dict[str, Any]:
    return {
        "algorithm": "FedDR.FedDR",
        "dataset": "FedDR authors' 200-writer FEMNIST JSON",
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "classes_number": CLASSES_NUMBER,
        "hidden_units": HIDDEN_UNITS,
        "rounds": rounds,
        "local_updates": LOCAL_UPDATES,
        "local_solver": "exact_full_gradient_descent",
        "learning_rate": learning_rate,
        "proximal_parameter": PROXIMAL_PARAMETER,
        "relaxation_parameter": RELAXATION_PARAMETER,
        "l1_regularization": L1_REGULARIZATION,
        "gradient_batch_size": gradient_batch_size,
        "model_seed": MODEL_SEED,
        "algorithm_seed": ALGORITHM_SEED,
        "aggregation": "sample_count_weighted_all_reflection_caches",
        "evaluation_point": "client_local_models",
        "regularizer_point": "post_prox_server_point",
    }


def save_history(history: ExperimentHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "communication_round",
            "regularized_train_loss",
            "train_cross_entropy",
            "l1_penalty",
            "train_accuracy",
            "test_accuracy",
            "server_max_abs",
            "server_nonzero_fraction",
            "cumulative_gradient_oracles",
            "algorithm_time_seconds",
        ))
        writer.writerows(zip(
            history.communication_round,
            history.train_loss,
            history.train_cross_entropy,
            history.l1_penalty,
            history.train_accuracy,
            history.test_accuracy,
            history.server_max_abs,
            history.server_nonzero_fraction,
            history.cumulative_gradient_oracles,
            history.algorithm_time_seconds,
        ))


def _save_checkpoint(
    path: Path,
    signature: dict[str, Any],
    history: ExperimentHistory,
    state: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "signature": signature,
        "history": asdict(history),
        "state": _state_to_cpu(state),
    }, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    signature: dict[str, Any],
    device: torch.device,
) -> tuple[ExperimentHistory, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("signature") != signature:
        raise ValueError(
            f"Checkpoint settings do not match the requested run: {path}. "
            "Use --restart to begin again."
        )
    history = ExperimentHistory(**checkpoint["history"])
    state = _state_to_device(checkpoint["state"], device)
    return history, state


def run_learning_rate(
    learning_rate: float,
    rounds: int,
    checkpoint_rounds: int,
    gradient_batch_size: int,
    evaluation_batch_size: int,
    device: torch.device,
    data: FederatedData,
    model: FEMNISTNet,
    unpack,
    initial_point: torch.Tensor,
    result_directory: Path,
    restart: bool,
) -> ExperimentHistory:
    """Run one learning rate with author-style aggregation and metrics."""

    if rounds <= 0 or checkpoint_rounds <= 0:
        raise ValueError("rounds and checkpoint_rounds must be positive.")
    run_name = _rate_name(learning_rate)
    checkpoint_path = result_directory / f"{run_name}.pt"
    csv_path = result_directory / f"{run_name}.csv"
    signature = experiment_signature(learning_rate, rounds, gradient_batch_size)
    if restart and checkpoint_path.exists():
        checkpoint_path.unlink()

    if checkpoint_path.exists():
        history, state = _load_checkpoint(checkpoint_path, signature, device)
        completed_rounds = history.communication_round[-1]
        print(f"Resuming {run_name} at round {completed_rounds}.", flush=True)
    else:
        (
            train_loss,
            train_cross_entropy,
            l1_penalty,
            train_accuracy,
            test_accuracy,
        ) = evaluate_point(
            model, unpack, initial_point, data.train, data.test,
            evaluation_batch_size,
        )
        server_max_abs, server_nonzero_fraction = parameter_diagnostics(
            initial_point
        )
        history = ExperimentHistory(
            communication_round=[0],
            train_loss=[train_loss],
            train_cross_entropy=[train_cross_entropy],
            l1_penalty=[l1_penalty],
            train_accuracy=[train_accuracy],
            test_accuracy=[test_accuracy],
            server_max_abs=[server_max_abs],
            server_nonzero_fraction=[server_nonzero_fraction],
            cumulative_gradient_oracles=[0],
            algorithm_time_seconds=[0.0],
        )
        state = None
        completed_rounds = 0
        save_history(history, csv_path)

    objectives = [
        make_full_gradient_objective(
            model, unpack, client, gradient_batch_size
        )
        for client in data.train
    ]
    client_weights = torch.tensor(
        [client.labels.numel() for client in data.train],
        dtype=initial_point.dtype,
        device=device,
    )
    while completed_rounds < rounds:
        algorithm = FedDR(
            InitialPoint=initial_point,
            IterationBudget=1,
            ProxFunction=l_1,
            ClientsNumber=CLIENTS_NUMBER,
            SamplingNumber=SAMPLING_NUMBER,
            SmoothL=0.0,
            ConvexMu=0.0,
            Constant=L1_REGULARIZATION,
            RandomSeed=ALGORITHM_SEED,
        )
        result = algorithm.go_torch(
            ProximalParameter=PROXIMAL_PARAMETER,
            RelaxationParameter=RELAXATION_PARAMETER,
            LocalUpdates=LOCAL_UPDATES,
            LocalLearningRate=learning_rate,
            Function=objectives,
            ClientWeights=client_weights,
            InitialState=state,
            RecordStatistics=False,
        )
        state = algorithm.final_state
        server_point = state["server_point"]
        (
            train_loss,
            train_cross_entropy,
            l1_penalty,
            train_accuracy,
            test_accuracy,
        ) = evaluate_client_models(
            model,
            unpack,
            state["local_x"],
            server_point,
            data.train,
            data.test,
            evaluation_batch_size,
        )
        server_max_abs, server_nonzero_fraction = parameter_diagnostics(
            server_point
        )
        completed_rounds += 1
        cumulative = (
            history.cumulative_gradient_oracles[-1]
            + int(result.oracle_calling[0].detach().cpu())
        )
        history.communication_round.append(completed_rounds)
        history.train_loss.append(train_loss)
        history.train_cross_entropy.append(train_cross_entropy)
        history.l1_penalty.append(l1_penalty)
        history.train_accuracy.append(train_accuracy)
        history.test_accuracy.append(test_accuracy)
        history.server_max_abs.append(server_max_abs)
        history.server_nonzero_fraction.append(server_nonzero_fraction)
        history.cumulative_gradient_oracles.append(cumulative)
        history.algorithm_time_seconds.append(
            float(result.process_time[0].detach().cpu())
        )

        save_history(history, csv_path)
        if completed_rounds % checkpoint_rounds == 0 or completed_rounds == rounds:
            _save_checkpoint(checkpoint_path, signature, history, state)
            print(
                f"{run_name}: round={completed_rounds}/{rounds}, "
                f"loss={history.train_loss[-1]:.6f}, "
                f"ce={history.train_cross_entropy[-1]:.6f}, "
                f"l1={history.l1_penalty[-1]:.6f}, "
                f"train_acc={history.train_accuracy[-1]:.4f}, "
                f"test_acc={history.test_accuracy[-1]:.4f}, "
                f"max_abs={history.server_max_abs[-1]:.3e}, "
                f"nonzero={history.server_nonzero_fraction[-1]:.3%}",
                flush=True,
            )
    return history


def save_figure(
    histories: dict[float, ExperimentHistory],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    panels = (
        ("train_loss", "Regularized training loss"),
        ("train_accuracy", "Training accuracy"),
        ("test_accuracy", "Test accuracy"),
    )
    for learning_rate, history in histories.items():
        for axis, (field, label) in zip(axes, panels):
            axis.plot(
                history.communication_round,
                getattr(history, field),
                linewidth=1.8,
                label=f"lr={learning_rate:g}",
            )
            axis.set_xlabel("Communication rounds")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
    axes[-1].legend()
    figure.suptitle(
        "FedDR on FEMNIST: author aggregation and client-local metrics"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--result-directory", type=Path, default=RESULT_DIRECTORY)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=LEARNING_RATES
    )
    parser.add_argument("--rounds", type=int, default=ITERATION_BUDGET)
    parser.add_argument(
        "--checkpoint-rounds", type=int, default=CHECKPOINT_ROUNDS
    )
    parser.add_argument(
        "--gradient-batch-size", type=int, default=GRADIENT_BATCH_SIZE,
        help=(
            "Memory chunk for exact gradient accumulation; this does not "
            "perform mini-batch SGD."
        ),
    )
    parser.add_argument(
        "--evaluation-batch-size", type=int, default=1024
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Discard same-setting checkpoints and rerun each learning rate.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    data = load_femnist(arguments.data_directory)
    arguments.result_directory.mkdir(parents=True, exist_ok=True)

    model = FEMNISTNet().to(device)
    initialize_like_tensorflow(model)
    unpack = parameter_unpacker(model)
    initial_point = parameters_to_vector(model.parameters()).detach()
    print(
        f"Device={device}; clients={len(data.clients)}; "
        f"train_samples={sum(c.labels.numel() for c in data.train)}; "
        f"test_samples={sum(c.labels.numel() for c in data.test)}",
        flush=True,
    )
    observed_labels = sorted({
        int(label)
        for client in data.train + data.test
        for label in torch.unique(client.labels).tolist()
    })
    print(
        f"Observed labels={observed_labels}; model outputs={CLASSES_NUMBER}. "
        "This matches the authors' released files and 26-output model.",
        flush=True,
    )
    print(
        f"Local solver: {LOCAL_UPDATES} exact full-gradient steps per client solve; "
        f"gradient memory chunk={arguments.gradient_batch_size}.",
        flush=True,
    )
    print(
        "Aggregation: sample-count weighted reflected caches; metrics: "
        "client-local models with the server L1 penalty.",
        flush=True,
    )

    histories: dict[float, ExperimentHistory] = {}
    for learning_rate in arguments.learning_rates:
        histories[learning_rate] = run_learning_rate(
            learning_rate=learning_rate,
            rounds=arguments.rounds,
            checkpoint_rounds=arguments.checkpoint_rounds,
            gradient_batch_size=arguments.gradient_batch_size,
            evaluation_batch_size=arguments.evaluation_batch_size,
            device=device,
            data=data,
            model=model,
            unpack=unpack,
            initial_point=initial_point,
            result_directory=arguments.result_directory,
            restart=arguments.restart,
        )
        save_figure(
            histories,
            arguments.result_directory / "feddr_femnist_l1_lr_sweep.png",
        )
    print(f"Results saved to {arguments.result_directory}", flush=True)


if __name__ == "__main__":
    main()
