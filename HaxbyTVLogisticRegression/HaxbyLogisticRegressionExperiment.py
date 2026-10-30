"""Haxby face-versus-house logistic regression with 3-D TV regularization.

This is intentionally a plain experiment script.  Edit the settings below,
then run ``python HaxbyLogisticRegressionExperiment.py``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Algorithms.REC_FedDyn_Correct import REC_FedDyn
from Algorithms.FMRI import (
    FMRIAlgorithmResults,
    FMRIModel,
    FMRIObjectiveFunction,
    FMRIValueFunction,
)
from Algorithms.FedDR import FedDR
from Algorithms.ProximalOperator import TV
from Algorithms.iFedDR import iFedDR
from Data.Haxby.load_haxby import iter_run_clients, load_haxby
from HaxbyExperimentOutput import save_aggregate_results, save_results


# ============================================================================
# Experiment settings -- edit parameters only in this block
# ============================================================================

CLIENTS_NUMBER = 59
SAMPLES_PER_CLIENT = 18
SAMPLING_NUMBER = 30
ITERATION_BUDGET = 15000                    # cumulative communication rounds
PROGRESS_INTERVAL = 1000                   # console report interval
CONTINUE_PREVIOUS_RUN = True

TV_REGULARIZATION = 1e-3
TV_PROXIMAL_ACCURACY = 1e-5
TV_PROXIMAL_MAX_ITERATIONS = 120 

LOCAL_UPDATES = 10                        # FedDR and iFedDR initial work
STEP_SIZE_FACTOR = 300.0
ALGORITHM_SEEDS = (
    (123,)
    if SAMPLING_NUMBER == CLIENTS_NUMBER
    else (123, 124, 125, 126, 127)
)
EXPERIMENT_ALGORITHMS = ("REC_FedDyn", "FedDR", "iFedDR")

REC_FED_DYN_SIGMA = 0.9
REC_FED_DYN_MAX_LOCAL_ITERATIONS = 500       # relative-error solver safety cap

FEDDR_RELAXATION = 1.0

IFEDDR_RELAXATION = 1.0
IFEDDR_SIGMA_SQUARED = 0.99
IFEDDR_LOCAL_UPDATES = 20
IFEDDR_REFINEMENT_FACTOR = 2.0
IFEDDR_MAX_REFINEMENTS = 10
IFEDDR_INCREASE_INITIAL_UPDATES = False
IFEDDR_CONTINUE_AFTER_MAX_REFINEMENTS = True
# Set an integer to stop iFedDR at the first 1000-round checkpoint reaching
# the limit. ``None`` runs the full communication-round budget.
IFEDDR_GRADIENT_ORACLE_BUDGET: int | None = 10000000

REFERENCE_ACCEPTED_RESIDUAL = 2.12e-3
FIGURE_DPI = 300

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "Data" / "Haxby" / "processed"
RESULT_DIRECTORY = PROJECT_ROOT / "Results" / "HaxbyTVLogisticRegression"
REFERENCE_FILE = (
    RESULT_DIRECTORY
    / "reference_cache"
    / "haxby_tv_logistic_seed2026_1caca2ff63cafb0f.npz"
)
SMOOTHNESS_FILE = RESULT_DIRECTORY / "local_smoothness.npz"


# ============================================================================
# Objective, TV proximal operator, and recorded history
# ============================================================================

@dataclass
class History:
    rounds: np.ndarray
    communications: np.ndarray
    function_values: np.ndarray
    test_accuracy: np.ndarray
    oracle_per_round: np.ndarray
    process_time: np.ndarray
    final_model: FMRIModel | None
    initialization_oracles: int = 0
    unsatisfied_rounds: int | None = 0
    refinements: int | None = 0
    truncated_from_round: int | None = None


def make_objective(images: np.ndarray, labels: np.ndarray) -> FMRIObjectiveFunction:
    """Return average logistic loss and gradient for one collection of scans."""

    labels = np.asarray(labels, dtype=np.float64)

    def objective(model: FMRIModel) -> tuple[np.float64, FMRIModel]:
        # Images stay float32 on disk; parameters and returned gradients are float64.
        omega32 = np.asarray(model.omega, dtype=images.dtype)
        scores = np.asarray(
            np.tensordot(images, omega32, axes=((1, 2, 3), (0, 1, 2))),
            dtype=np.float64,
        ) + model.intercept
        margins = labels * scores
        value = np.mean(np.logaddexp(0.0, -margins))
        score_gradient = -labels * np.exp(-np.logaddexp(0.0, margins))
        omega_gradient = np.tensordot(
            np.asarray(score_gradient, dtype=images.dtype), images, axes=(0, 0)
        ) / labels.size
        return np.float64(value), FMRIModel(
            np.asarray(omega_gradient, dtype=np.float64),
            np.float64(np.mean(score_gradient)),
        )

    return objective


def make_value_function(images: np.ndarray, labels: np.ndarray) -> FMRIValueFunction:
    """Return logistic loss only; no coefficient gradient is formed."""

    labels = np.asarray(labels, dtype=np.float64)

    def value(model: FMRIModel) -> np.float64:
        scores = np.asarray(
            np.tensordot(
                images,
                np.asarray(model.omega, dtype=images.dtype),
                axes=((1, 2, 3), (0, 1, 2)),
            ),
            dtype=np.float64,
        ) + model.intercept
        margins = labels * scores
        return np.float64(np.mean(np.logaddexp(0.0, -margins)))

    return value


def tv_proximal_operator(
    omega: np.ndarray,
    parameter: np.float64,
    constant: np.float64,
    only_value: bool,
) -> np.float64 | np.ndarray:
    """Use the repository TV operator with the experiment's fixed accuracy."""

    return TV(
        omega,
        parameter,
        constant,
        only_value,
        omega.shape,
        accuracy=np.float64(TV_PROXIMAL_ACCURACY),
        maximumIteration=TV_PROXIMAL_MAX_ITERATIONS,
    )


def full_objective(model: FMRIModel, smooth_value_function: FMRIValueFunction) -> float:
    smooth_value = smooth_value_function(model)
    tv_value = tv_proximal_operator(
        model.omega, np.float64(1.0), np.float64(TV_REGULARIZATION), True
    )
    return float(smooth_value + tv_value)


def accuracy(model: FMRIModel, images: np.ndarray, labels: np.ndarray) -> np.float64:
    scores = np.asarray(
        np.tensordot(
            images,
            np.asarray(model.omega, dtype=images.dtype),
            axes=((1, 2, 3), (0, 1, 2)),
        ),
        dtype=np.float64,
    ) + model.intercept
    predictions = np.where(scores >= 0.0, 1, -1)
    return np.float64(np.mean(predictions == np.asarray(labels)))


# ============================================================================
# Simple checkpoint helpers
# ============================================================================

def initial_history(
    model: FMRIModel,
    function_value: float,
    test_accuracy: float,
) -> History:
    return History(
        rounds=np.array([0], dtype=np.int64),
        communications=np.array([0], dtype=np.int64),
        function_values=np.array([function_value], dtype=np.float64),
        test_accuracy=np.array([test_accuracy], dtype=np.float64),
        oracle_per_round=np.array([0], dtype=np.int64),
        process_time=np.array([0.0], dtype=np.float64),
        final_model=model.copy(),
    )


def load_checkpoint(
    path: Path,
    settings: dict,
    model: FMRIModel,
    function_value: float,
    test_accuracy: float,
) -> tuple[dict | None, History, int]:
    if not CONTINUE_PREVIOUS_RUN or not path.is_file():
        return None, initial_history(model, function_value, test_accuracy), 0

    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if saved["settings"] != settings:
        raise ValueError(f"Checkpoint settings do not match its key: {path}")

    completed = int(saved["completed_rounds"])
    history = saved["history"]
    if completed > ITERATION_BUDGET:
        history = truncate_history(history, ITERATION_BUDGET, completed)
        print(
            f"using the first {ITERATION_BUDGET} rounds from "
            f"a {completed}-round checkpoint: {path}"
        )
        # Scalar histories can be reused exactly.  The saved optimizer state
        # and final model belong to ``completed``, not to the requested prefix.
        return None, history, ITERATION_BUDGET
    return saved["algorithm_state"], history, completed


def truncate_history(
    history: History,
    iteration_budget: int,
    completed_rounds: int | None = None,
) -> History:
    """Return an exact scalar-history prefix without mislabelling its model."""

    if iteration_budget < 0:
        raise ValueError("iteration_budget must be nonnegative.")
    keep = int(np.searchsorted(history.rounds, iteration_budget, side="right"))
    if keep == 0 or int(history.rounds[keep - 1]) != iteration_budget:
        raise ValueError(
            f"History does not contain requested round {iteration_budget}."
        )
    source_round = (
        int(history.rounds[-1])
        if completed_rounds is None
        else int(completed_rounds)
    )
    if source_round <= iteration_budget:
        return history
    return History(
        rounds=np.asarray(history.rounds[:keep]).copy(),
        communications=np.asarray(history.communications[:keep]).copy(),
        function_values=np.asarray(history.function_values[:keep]).copy(),
        test_accuracy=np.asarray(history.test_accuracy[:keep]).copy(),
        oracle_per_round=np.asarray(history.oracle_per_round[:keep]).copy(),
        process_time=np.asarray(history.process_time[:keep]).copy(),
        final_model=None,
        initialization_oracles=history.initialization_oracles,
        # Only final totals were saved in the old checkpoint, so exact prefix
        # totals for these two diagnostics cannot be reconstructed.
        unsatisfied_rounds=None,
        refinements=None,
        truncated_from_round=source_round,
    )


def append_segment(
    history: History,
    segment: FMRIAlgorithmResults,
    completed_rounds: int,
    initialization_oracles: int = 0,
) -> History:
    if history.final_model is None:
        raise ValueError("Cannot continue from a history without a cutoff model.")
    added = segment.function_value.size
    rounds = np.arange(
        completed_rounds + 1, completed_rounds + added + 1, dtype=np.int64
    )
    return History(
        rounds=np.concatenate((history.rounds, rounds)),
        communications=np.concatenate(
            (history.communications, np.asarray(segment.communication_rounds))
        ),
        function_values=np.concatenate(
            (history.function_values, segment.function_value)
        ),
        test_accuracy=np.concatenate((history.test_accuracy, segment.test_accuracy)),
        oracle_per_round=np.concatenate(
            (history.oracle_per_round, segment.oracle_calling)
        ),
        process_time=np.concatenate((history.process_time, segment.process_time)),
        final_model=segment.final_point.copy(),
        initialization_oracles=(
            history.initialization_oracles + int(initialization_oracles)
        ),
        unsatisfied_rounds=(
            int(history.unsatisfied_rounds)
            + int(np.count_nonzero(segment.index))
        ),
        refinements=(
            int(history.refinements) + int(np.sum(segment.refinement_count))
        ),
        truncated_from_round=None,
    )


def save_checkpoint(
    path: Path,
    settings: dict,
    state: dict,
    history: History,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(
            {
                "settings": settings,
                "completed_rounds": int(history.rounds[-1]),
                "algorithm_state": state,
                "history": history,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def segment_budget(completed_rounds: int) -> int:
    return min(PROGRESS_INTERVAL, ITERATION_BUDGET - completed_rounds)


def ifeddr_segment_budget(completed_rounds: int) -> int:
    """End the next iFedDR segment at a gradient-budget check boundary."""

    rounds_to_check = PROGRESS_INTERVAL - completed_rounds % PROGRESS_INTERVAL
    return min(rounds_to_check, ITERATION_BUDGET - completed_rounds)


def oracle_budget_cutoff_round(
    history: History,
    oracle_budget: int | None,
) -> int | None:
    """Return the first 1000-round checkpoint reaching ``oracle_budget``."""

    if oracle_budget is None:
        return None
    if oracle_budget <= 0:
        raise ValueError("IFEDDR_GRADIENT_ORACLE_BUDGET must be positive or None.")
    cumulative = history.initialization_oracles + np.cumsum(
        history.oracle_per_round, dtype=np.int64
    )
    checkpoint = (
        (history.rounds > 0)
        & (history.rounds % PROGRESS_INTERVAL == 0)
        & (cumulative >= oracle_budget)
    )
    reached = np.flatnonzero(checkpoint)
    if reached.size == 0:
        return None
    return int(history.rounds[int(reached[0])])


def history_for_oracle_budget(
    history: History,
    oracle_budget: int | None,
) -> tuple[History, int | None]:
    """Return the reportable prefix while leaving the checkpoint untouched."""

    cutoff_round = oracle_budget_cutoff_round(history, oracle_budget)
    if cutoff_round is None:
        return history, None
    return truncate_history(history, cutoff_round), cutoff_round


def print_progress(
    algorithm: str,
    history: History,
    reference_value: float,
) -> None:
    total_oracles = history.initialization_oracles + int(
        np.sum(history.oracle_per_round)
    )
    print(
        f"{algorithm:9s} round={int(history.rounds[-1])}, "
        f"communication={int(history.communications[-1])}, "
        f"error={history.function_values[-1] - reference_value:.8e}, "
        f"accuracy={history.test_accuracy[-1]:.4f}, "
        f"oracle_total={total_oracles}"
    )


ALGORITHM_IMPLEMENTATIONS = {
    "REC_FedDyn": "fmri_tensor_rec_feddyn_v1",
    "FedDR": "fmri_tensor_feddr",
    "iFedDR": "fmri_tensor_ifeddr_corrected_zeta_scaling",
    "FedDyn": "fmri_tensor_feddyn_v1",
}


def short_key(settings: dict) -> str:
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def problem_settings(full_settings: dict) -> dict:
    """Settings that define the outer distributed optimization problem."""
    names = (
        "reference_key",
        "volume_shape",
        "clients_number",
        "samples_per_client",
        "sampling_number",
        "tv_regularization",
        "smoothness",
    )
    return {
        "format": "haxby_tv_logistic_problem_v1",
        **{name: full_settings[name] for name in names},
    }


def algorithm_settings(
    algorithm: str,
    problem_key: str,
    full_settings: dict,
) -> dict:
    """Settings that affect one algorithm under a fixed problem."""
    shared = {
        "format": "haxby_tv_logistic_algorithm_v1",
        "algorithm": algorithm,
        "implementation": ALGORITHM_IMPLEMENTATIONS[algorithm],
        "problem_key": problem_key,
        "algorithm_seed": full_settings["algorithm_seed"],
        "tv_proximal_accuracy": full_settings["tv_proximal_accuracy"],
        "tv_proximal_max_iterations": full_settings[
            "tv_proximal_max_iterations"
        ],
        "step_size_factor": full_settings["step_size_factor"],
        "local_solver": full_settings["local_solver"],
    }
    if algorithm == "REC_FedDyn":
        shared.update(
            penalty=full_settings["penalty"],
            rec_feddyn_sigma=full_settings["rec_feddyn_sigma"],
            rec_feddyn_max_local_iterations=full_settings[
                "rec_feddyn_max_local_iterations"
            ],
        )
    elif algorithm == "FedDR":
        shared.update(
            step_size=full_settings["step_size"],
            local_updates=full_settings["local_updates"],
            feddr_relaxation=full_settings["feddr_relaxation"],
        )
    elif algorithm == "iFedDR":
        shared.update(
            step_size=full_settings["step_size"],
            local_updates=full_settings["ifeddr_local_updates"],
            ifeddr_relaxation=full_settings["ifeddr_relaxation"],
            ifeddr_sigma_squared=full_settings["ifeddr_sigma_squared"],
            ifeddr_refinement_factor=full_settings[
                "ifeddr_refinement_factor"
            ],
            ifeddr_max_refinements=full_settings["ifeddr_max_refinements"],
            ifeddr_increase_initial_updates=full_settings[
                "ifeddr_increase_initial_updates"
            ],
            ifeddr_continue_after_max_refinements=full_settings[
                "ifeddr_continue_after_max_refinements"
            ],
        )
    elif algorithm == "FedDyn":
        shared.update(
            penalty=full_settings["penalty"],
            local_updates=full_settings["local_updates"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return shared


def checkpoint_path(
    run_directory: Path,
    algorithm: str,
    settings: dict,
) -> Path:
    return (
        run_directory
        / "checkpoints"
        / algorithm
        / f"{short_key(settings)}.pkl"
    )


def problem_run_directory(problem_key: str) -> Path:
    """Use one run directory for one fixed distributed problem."""

    return RESULT_DIRECTORY / "runs" / f"run_{problem_key}"


def save_problem_settings(run_directory: Path, settings: dict) -> None:
    """Write the outer problem description without replacing a mismatch."""
    path = run_directory / "problem_settings.json"
    run_directory.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved != settings:
            raise ValueError(f"Problem settings do not match {path}.")
        return
    with path.open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2, ensure_ascii=False)


def reuse_matching_checkpoint(
    runs_directory: Path,
    run_directory: Path,
    algorithm: str,
    settings: dict,
    current_problem_settings: dict | None = None,
) -> Path:
    """Copy the best compatible algorithm checkpoint from another run."""
    destination = checkpoint_path(run_directory, algorithm, settings)
    if not CONTINUE_PREVIOUS_RUN or destination.is_file():
        return destination

    key = short_key(settings)
    candidates = []
    pattern = f"run_*/checkpoints/{algorithm}/{key}.pkl"
    for source in runs_directory.glob(pattern):
        with source.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["settings"] != settings:
            raise ValueError(f"Checkpoint key collision or invalid settings: {source}")
        completed = int(saved["completed_rounds"])
        candidates.append((completed, source, saved, False))

    # Older Haxby runs stored the algorithm seed in the problem identity.  It
    # is algorithm-side randomness, so current runs store it in each algorithm
    # identity instead.  Normalize that one schema difference to retain
    # compatible seed-123 work without accepting the former iFedDR settings.
    if current_problem_settings is not None:
        broad_pattern = f"run_*/checkpoints/{algorithm}/*.pkl"
        exact_sources = {candidate[1] for candidate in candidates}
        for source in runs_directory.glob(broad_pattern):
            if source in exact_sources:
                continue
            problem_path = source.parents[2] / "problem_settings.json"
            if not problem_path.is_file():
                continue
            with problem_path.open("r", encoding="utf-8") as stream:
                old_problem_settings = json.load(stream)
            old_algorithm_seed = old_problem_settings.pop("algorithm_seed", None)
            if old_problem_settings != current_problem_settings:
                continue
            with source.open("rb") as stream:
                saved = pickle.load(stream)
            normalized_settings = dict(saved["settings"])
            normalized_settings["problem_key"] = settings["problem_key"]
            normalized_settings.setdefault("algorithm_seed", old_algorithm_seed)
            if normalized_settings != settings:
                continue
            completed = int(saved["completed_rounds"])
            converted = dict(saved)
            converted["settings"] = settings
            candidates.append((completed, source, converted, True))

    if candidates:
        # Prefer the shortest checkpoint that already covers the requested
        # prefix.  If none does, continue from the longest available prefix.
        def candidate_rank(candidate):
            completed = candidate[0]
            if completed >= ITERATION_BUDGET:
                return 2, -completed
            return 1, completed

        completed, source, saved, converted = max(candidates, key=candidate_rank)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if converted:
            with destination.open("wb") as stream:
                pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            shutil.copy2(source, destination)
        print(
            f"reused {algorithm} checkpoint ({completed} rounds)\n"
            f"  from {source}\n"
            f"    to {destination}"
        )
    return destination


# ============================================================================
# Experiment
# ============================================================================

def run_algorithm_seed(algorithm_seed: int):
    data = load_haxby(DATA_DIRECTORY, mmap_mode="r")
    clients = list(iter_run_clients(data, "train"))
    volume_shape = tuple(int(size) for size in data["train_X"].shape[1:])
    if len(clients) != CLIENTS_NUMBER:
        raise ValueError(f"Expected {CLIENTS_NUMBER} training clients.")
    if {client_y.size for _, _, client_y, _ in clients} != {SAMPLES_PER_CLIENT}:
        raise ValueError(f"Expected {SAMPLES_PER_CLIENT} samples per client.")

    local_objectives = [
        make_objective(client_x, client_y) for _, client_x, client_y, _ in clients
    ]
    global_value = make_value_function(data["train_X"], data["train_y"])

    if not SMOOTHNESS_FILE.is_file():
        raise FileNotFoundError(f"Missing smoothness cache: {SMOOTHNESS_FILE}")
    with np.load(SMOOTHNESS_FILE) as cached:
        expected_shape = tuple(int(value) for value in cached["training_shape"])
        if expected_shape != tuple(data["train_X"].shape):
            raise ValueError("The smoothness cache belongs to different training data.")
        smoothness = float(np.max(cached["local_bounds"]))

    if not REFERENCE_FILE.is_file():
        raise FileNotFoundError(f"Missing cached reference: {REFERENCE_FILE}")
    with np.load(REFERENCE_FILE) as cached:
        reference_model = FMRIModel(
            np.asarray(cached["omega"], dtype=np.float64),
            np.float64(cached["intercept"]),
        )
        reference_value = float(cached["objective_value"])
        reference_residual = float(cached["residual"])
        reference_key = str(cached["problem_key"].item())
        reference_iterations = int(cached["iterations"])
    if reference_model.omega.shape != volume_shape:
        raise ValueError("The cached reference uses a different voxel grid.")
    if reference_residual > REFERENCE_ACCEPTED_RESIDUAL:
        raise ValueError(
            f"Reference residual {reference_residual:.6e} exceeds the accepted "
            f"value {REFERENCE_ACCEPTED_RESIDUAL:.6e}."
        )

    step_size = STEP_SIZE_FACTOR / smoothness
    penalty = 1.0 / step_size
    initial_model = FMRIModel.zeros(volume_shape)
    initial_value = full_objective(initial_model, global_value)
    initial_accuracy = float(
        accuracy(initial_model, data["test_X"], data["test_y"])
    )

    full_settings = {
        "implementation": "simple_fmri_accelerated",
        "reference_key": reference_key,
        "volume_shape": list(volume_shape),
        "clients_number": CLIENTS_NUMBER,
        "samples_per_client": SAMPLES_PER_CLIENT,
        "sampling_number": SAMPLING_NUMBER,
        "tv_regularization": TV_REGULARIZATION,
        "tv_proximal_accuracy": TV_PROXIMAL_ACCURACY,
        "tv_proximal_max_iterations": TV_PROXIMAL_MAX_ITERATIONS,
        "local_updates": LOCAL_UPDATES,
        "step_size_factor": STEP_SIZE_FACTOR,
        "smoothness": smoothness,
        "step_size": step_size,
        "penalty": penalty,
        "algorithm_seed": algorithm_seed,
        "rec_feddyn_sigma": REC_FED_DYN_SIGMA,
        "rec_feddyn_max_local_iterations": REC_FED_DYN_MAX_LOCAL_ITERATIONS,
        "feddr_relaxation": FEDDR_RELAXATION,
        "ifeddr_relaxation": IFEDDR_RELAXATION,
        "ifeddr_sigma_squared": IFEDDR_SIGMA_SQUARED,
        "ifeddr_local_updates": IFEDDR_LOCAL_UPDATES,
        "ifeddr_refinement_factor": IFEDDR_REFINEMENT_FACTOR,
        "ifeddr_max_refinements": IFEDDR_MAX_REFINEMENTS,
        "ifeddr_increase_initial_updates": IFEDDR_INCREASE_INITIAL_UPDATES,
        "ifeddr_continue_after_max_refinements": (
            IFEDDR_CONTINUE_AFTER_MAX_REFINEMENTS
        ),
        "local_solver": "Nesterov accelerated gradient for all four algorithms",
    }
    outer_settings = problem_settings(full_settings)
    problem_key = short_key(outer_settings)
    algorithm_settings_by_name = {
        name: algorithm_settings(name, problem_key, full_settings)
        for name in EXPERIMENT_ALGORITHMS
    }
    algorithm_keys = {
        name: short_key(settings)
        for name, settings in algorithm_settings_by_name.items()
    }
    run_directory = problem_run_directory(problem_key)
    save_problem_settings(run_directory, outer_settings)
    output_settings = {
        "problem_key": problem_key,
        "problem_settings": outer_settings,
        "algorithm_keys": algorithm_keys,
        "algorithm_settings": algorithm_settings_by_name,
        "iteration_budget": ITERATION_BUDGET,
        "ifeddr_gradient_oracle_budget": IFEDDR_GRADIENT_ORACLE_BUDGET,
        "ifeddr_oracle_check_interval": PROGRESS_INTERVAL,
    }

    def test_accuracy(model: FMRIModel) -> np.float64:
        return accuracy(model, data["test_X"], data["test_y"])

    print(f"Haxby TV-logistic experiment, algorithm seed={algorithm_seed}")
    print(f"clients={CLIENTS_NUMBER}, sampled={SAMPLING_NUMBER}, rounds={ITERATION_BUDGET}")
    print(f"LOCAL_UPDATES={LOCAL_UPDATES} (accelerated local gradients)")
    print(f"step_size={step_size:.8e}, penalty={penalty:.8e}")
    print(
        f"reference={reference_value:.12e}, residual={reference_residual:.6e}, "
        f"iterations={reference_iterations}"
    )
    print(f"run directory: {run_directory}")

    saved_runs = {}
    for name in EXPERIMENT_ALGORITHMS:
        current_settings = algorithm_settings_by_name[name]
        checkpoint = reuse_matching_checkpoint(
            RESULT_DIRECTORY / "runs",
            run_directory,
            name,
            current_settings,
            outer_settings,
        )
        state, history, completed = load_checkpoint(
            checkpoint,
            current_settings,
            initial_model,
            initial_value,
            initial_accuracy,
        )
        saved_runs[name] = (
            checkpoint,
            current_settings,
            state,
            history,
            completed,
        )
        print(f"{name:9s} checkpoint rounds={completed}: {checkpoint}")

    histories: dict[str, History] = {}

    # ------------------------------------------------------------------
    # REC_FedDyn: accelerated local solver with relative-error stopping.
    # ------------------------------------------------------------------
    checkpoint, current_settings, state, history, completed = saved_runs["REC_FedDyn"]
    while completed < ITERATION_BUDGET:
        optimizer = REC_FedDyn(
            InitialPoint=history.final_model.omega,
            IterationBudget=segment_budget(completed),
            ProxFunction=tv_proximal_operator,
            ClientsNumber=CLIENTS_NUMBER,
            SamplingNumber=SAMPLING_NUMBER,
            SmoothL=np.float64(smoothness),
            ConvexMu=np.float64(0.0),
            Constant=np.float64(TV_REGULARIZATION),
            RandomSeed=algorithm_seed,
        )
        segment = optimizer.go_fmri(
            PenaltyParameter=np.float64(penalty),
            Sigma=np.float64(REC_FED_DYN_SIGMA),
            MaxLocalIterations=REC_FED_DYN_MAX_LOCAL_ITERATIONS,
            Function=local_objectives,
            StatisticsFunction=global_value,
            AccuracyFunction=test_accuracy,
            InitialIntercept=history.final_model.intercept,
            InitialState=state,
        )
        history = append_segment(history, segment, completed)
        save_checkpoint(checkpoint, current_settings, optimizer.final_state, history)
        state = optimizer.final_state
        completed = int(history.rounds[-1])
        print_progress("REC_FedDyn", history, reference_value)
    histories["REC_FedDyn"] = history

    # ------------------------------------------------------------------
    # FedDR: exactly LOCAL_UPDATES accelerated-gradient steps per client.
    # ------------------------------------------------------------------
    checkpoint, current_settings, state, history, completed = saved_runs["FedDR"]
    while completed < ITERATION_BUDGET:
        optimizer = FedDR(
            InitialPoint=history.final_model.omega,
            IterationBudget=segment_budget(completed),
            ProxFunction=tv_proximal_operator,
            ClientsNumber=CLIENTS_NUMBER,
            SamplingNumber=SAMPLING_NUMBER,
            SmoothL=np.float64(smoothness),
            ConvexMu=np.float64(0.0),
            Constant=np.float64(TV_REGULARIZATION),
            RandomSeed=algorithm_seed,
        )
        segment = optimizer.go_fmri(
            ProximalParameter=np.float64(step_size),
            RelaxationParameter=np.float64(FEDDR_RELAXATION),
            LocalUpdates=LOCAL_UPDATES,
            Function=local_objectives,
            StatisticsFunction=global_value,
            AccuracyFunction=test_accuracy,
            InitialIntercept=history.final_model.intercept,
            InitialState=state,
        )
        history = append_segment(
            history,
            segment,
            completed,
            initialization_oracles=optimizer.initialization_oracle_calling,
        )
        save_checkpoint(checkpoint, current_settings, optimizer.final_state, history)
        state = optimizer.final_state
        completed = int(history.rounds[-1])
        print_progress("FedDR", history, reference_value)
    histories["FedDR"] = history

    # ------------------------------------------------------------------
    # iFedDR: accelerated local steps with the explicit refinement rule.
    # ------------------------------------------------------------------
    checkpoint, current_settings, state, history, completed = saved_runs["iFedDR"]
    report_history, cutoff_round = history_for_oracle_budget(
        history, IFEDDR_GRADIENT_ORACLE_BUDGET
    )
    while completed < ITERATION_BUDGET and cutoff_round is None:
        optimizer = iFedDR(
            InitialPoint=history.final_model.omega,
            IterationBudget=ifeddr_segment_budget(completed),
            ProxFunction=tv_proximal_operator,
            ClientsNumber=CLIENTS_NUMBER,
            SamplingNumber=SAMPLING_NUMBER,
            SmoothL=np.float64(smoothness),
            ConvexMu=np.float64(0.0),
            Constant=np.float64(TV_REGULARIZATION),
            RandomSeed=algorithm_seed,
        )
        segment = optimizer.go_fmri(
            ProximalParameter=np.float64(step_size),
            RelaxationParameter=np.float64(IFEDDR_RELAXATION),
            SigmaSquared=np.float64(IFEDDR_SIGMA_SQUARED),
            InitialLocalUpdates=IFEDDR_LOCAL_UPDATES,
            RefinementFactor=np.float64(IFEDDR_REFINEMENT_FACTOR),
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            IncreaseInitialUpdates=IFEDDR_INCREASE_INITIAL_UPDATES,
            ContinueAfterMaxRefinements=(
                IFEDDR_CONTINUE_AFTER_MAX_REFINEMENTS
            ),
            Function=local_objectives,
            StatisticsFunction=global_value,
            AccuracyFunction=test_accuracy,
            InitialIntercept=history.final_model.intercept,
            InitialState=state,
        )
        history = append_segment(history, segment, completed)
        save_checkpoint(checkpoint, current_settings, optimizer.final_state, history)
        state = optimizer.final_state
        completed = int(history.rounds[-1])
        print_progress("iFedDR", history, reference_value)
        report_history, cutoff_round = history_for_oracle_budget(
            history, IFEDDR_GRADIENT_ORACLE_BUDGET
        )
    if cutoff_round is not None:
        cutoff_oracles = report_history.initialization_oracles + int(
            np.sum(report_history.oracle_per_round)
        )
        print(
            "iFedDR stopped for reporting at "
            f"round={cutoff_round}, oracle_total={cutoff_oracles}, "
            f"oracle_budget={IFEDDR_GRADIENT_ORACLE_BUDGET}"
        )
    else:
        report_history = history
    histories["iFedDR"] = report_history

    reference_information = {
        "file": str(REFERENCE_FILE),
        "problem_key": reference_key,
        "objective_value": reference_value,
        "iterations": reference_iterations,
        "residual": reference_residual,
        "accepted_residual": REFERENCE_ACCEPTED_RESIDUAL,
        "used_without_replacement": True,
    }
    if SAMPLING_NUMBER == CLIENTS_NUMBER:
        save_results(
            run_directory,
            output_settings,
            histories,
            reference_model,
            reference_value,
            reference_information,
            DATA_DIRECTORY,
            FIGURE_DPI,
            save_plots=True,
        )

    for name, history in histories.items():
        total_oracles = history.initialization_oracles + int(
            np.sum(history.oracle_per_round)
        )
        print(
            f"{name:9s}: F={history.function_values[-1]:.8e}, "
            f"accuracy={history.test_accuracy[-1]:.4f}, "
            f"oracle total={total_oracles}"
        )
    if SAMPLING_NUMBER == CLIENTS_NUMBER:
        print(f"Results saved to {run_directory}")
    else:
        print(f"Seed checkpoints saved to {run_directory}")
    return (
        histories,
        output_settings,
        reference_model,
        reference_value,
        reference_information,
    )


def main() -> None:
    histories_by_algorithm = {
        name: [] for name in EXPERIMENT_ALGORITHMS
    }
    settings_by_seed = {}
    reference_model = None
    reference_value = None
    reference_information = None

    for algorithm_seed in ALGORITHM_SEEDS:
        (
            seed_histories,
            seed_settings,
            current_reference_model,
            current_reference_value,
            current_reference_information,
        ) = run_algorithm_seed(algorithm_seed)
        settings_by_seed[str(algorithm_seed)] = seed_settings
        for name, history in seed_histories.items():
            histories_by_algorithm[name].append(history)
        if reference_model is None:
            reference_model = current_reference_model
            reference_value = current_reference_value
            reference_information = current_reference_information

    if SAMPLING_NUMBER == CLIENTS_NUMBER:
        print("Full-participation seed-123 plots saved with the run results.")
        return

    aggregate_settings = {
        "format": "haxby_tv_logistic_five_seed_aggregate_v1",
        "algorithm_seeds": list(ALGORITHM_SEEDS),
        "seed_settings": settings_by_seed,
        "iteration_budget": ITERATION_BUDGET,
    }
    aggregate_directory = (
        RESULT_DIRECTORY / "aggregates"
        / f"aggregate_{short_key(aggregate_settings)}"
    )
    save_aggregate_results(
        aggregate_directory,
        aggregate_settings,
        histories_by_algorithm,
        reference_model,
        reference_value,
        reference_information,
        DATA_DIRECTORY,
        FIGURE_DPI,
    )
    print(f"Five-seed aggregate saved to {aggregate_directory}")


if __name__ == "__main__":
    main()
