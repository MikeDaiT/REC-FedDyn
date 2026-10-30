"""Strongly convex distributed TV-inpainting experiment on BSDS300."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Algorithms.Base_Optimizer import FloatArray
from Algorithms.REC_FedDyn_Correct import REC_FedDyn
from Algorithms.FedDR import FedDR
from Algorithms.FedDyn import FedDyn
from Algorithms.ProximalOperator import TV
from Algorithms.iFedDR import iFedDR
from ExperimentPlotStyle import LINE_WIDTH, apply_experiment_plot_style


apply_experiment_plot_style()

ObjectiveFunction = Callable[[FloatArray], tuple[np.float64, FloatArray]]


# ======================================================================
# Experiment parameters
# ======================================================================

CLIENTS_NUMBER = 20
SAMPLING_NUMBER = 5

IMAGE_IDS = (
    "102061",  # castle
    "108005",  # tiger
    "167083",  # mountain and lake
    "208001",  # mushroom
    "241004",  # rocks and landscape
    "86016",  # garden and gravel
)
HETEROGENEITY_LEVELS = (1, 2, 3)
ALGORITHM_NAMES = ("REC_FedDyn", "FedDyn", "FedDR", "iFedDR")

SCRATCH_NUMBER = 8
SCRATCH_WIDTH = 6
SCRATCH_LENGTH = 50
SCRATCH_BOUNDARY_MARGIN = 10 

RIDGE_MU = 0.001
TV_REGULARIZATION = 0.02

# Cumulative target.  For example, changing 300 to 500 continues for 200
# rounds when a 300-round checkpoint exists.
ITERATION_BUDGET = 500
CONTINUE_PREVIOUS_RUN = True

DATA_SEED = 2026
ALGORITHM_SEEDS = (123, 124, 125, 126, 127)
# The grid-search utilities intentionally use one seed; keep their established
# seed while the comparison experiment above uses every seed in ALGORITHM_SEEDS.
ALGORITHM_SEED = ALGORITHM_SEEDS[0]
ERROR_FLOOR = 1e-16

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "Data" / "BSDS300" / "images" / "test"
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "Results" / "BSDS300StronglyConvexTVInpainting"
)


# ======================================================================
# Algorithm parameters
# ======================================================================

OVERALL_STEPSIZE = 1.0
PENALTY_PARAMETER = 1.0 / OVERALL_STEPSIZE
RELATIVE_ERROR = 0.1

FEDDR_PROXIMAL_PARAMETER = OVERALL_STEPSIZE
FEDDR_RELAXATION_PARAMETER = 1.0
FEDDR_LOCAL_UPDATES = 10

IFEDDR_PROXIMAL_PARAMETER = OVERALL_STEPSIZE
IFEDDR_RELAXATION_PARAMETER = 1.0
IFEDDR_SIGMA_SQUARED = 0.99
IFEDDR_INITIAL_LOCAL_UPDATES = 5
IFEDDR_REFINEMENT_FACTOR = 2.0
IFEDDR_MAX_REFINEMENTS = 100
IFEDDR_INCREASE_INITIAL_UPDATES = True

TV_PROXIMAL_ACCURACY = 1e-5
TV_PROXIMAL_MAX_ITERATIONS = 120    

REFERENCE_TOLERANCE = 2e-6
REFERENCE_MAX_ITERATIONS = 3500
REFERENCE_CHECK_INTERVAL = 50
REFERENCE_TV_ACCURACY = 1e-8  
REFERENCE_TV_MAX_ITERATIONS = 600

FIGURE_DPI = 300


# ======================================================================
# Image, missing scratches, and client data
# ======================================================================


def split_into_client_blocks(
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Split a landscape image 4x5 and a portrait image 5x4."""

    height, width = shape
    rows, columns = (4, 5) if width >= height else (5, 4)
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    column_edges = np.linspace(0, width, columns + 1, dtype=int)
    return [
        (
            int(row_edges[row]),
            int(row_edges[row + 1]),
            int(column_edges[column]),
            int(column_edges[column + 1]),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def make_scratch_path(turns: int, rng: random.Random) -> list[tuple[int, int]]:
    """Make a 4-neighbour path with zero, one, or two right-angle turns."""

    total_steps = SCRATCH_LENGTH - SCRATCH_WIDTH
    while True:
        cuts = sorted(rng.sample(range(1, total_steps), turns))
        segment_steps = np.diff([0, *cuts, total_steps])
        if min(segment_steps) >= 10:
            break

    direction = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    path = [(0, 0)]
    for segment, steps in enumerate(segment_steps):
        if segment > 0:
            row_direction, column_direction = direction
            direction = rng.choice(
                [
                    (-column_direction, row_direction),
                    (column_direction, -row_direction),
                ]
            )
        for _ in range(int(steps)):
            row, column = path[-1]
            path.append((row + direction[0], column + direction[1]))
    return path


def choose_missing_scratches(
    shape: tuple[int, int], seed: int
) -> list[list[tuple[int, int]]]:
    """Place eight separated scratches with the configured width and length."""

    height, width = shape
    rng = random.Random(seed)
    turn_numbers = [0, 0, 1, 1, 1, 1, 2, 2]
    rng.shuffle(turn_numbers)
    occupied = np.zeros(shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    half_width = SCRATCH_WIDTH // 2
    offsets = range(-half_width, SCRATCH_WIDTH - half_width)

    for turns in turn_numbers:
        for _ in range(1_000):
            path = make_scratch_path(turns, rng)
            relative_pixels = {
                (row + row_offset, column + column_offset)
                for row, column in path
                for row_offset in offsets
                for column_offset in offsets
            }
            rows = [row for row, _ in relative_pixels]
            columns = [column for _, column in relative_pixels]
            row_shift = rng.randint(
                SCRATCH_BOUNDARY_MARGIN - min(rows),
                height - SCRATCH_BOUNDARY_MARGIN - 1 - max(rows),
            )
            column_shift = rng.randint(
                SCRATCH_BOUNDARY_MARGIN - min(columns),
                width - SCRATCH_BOUNDARY_MARGIN - 1 - max(columns),
            )
            pixels = [
                (row + row_shift, column + column_shift)
                for row, column in relative_pixels
            ]

            row_min = min(row for row, _ in pixels) - SCRATCH_WIDTH
            row_max = max(row for row, _ in pixels) + SCRATCH_WIDTH + 1
            column_min = min(column for _, column in pixels) - SCRATCH_WIDTH
            column_max = max(column for _, column in pixels) + SCRATCH_WIDTH + 1
            if np.any(
                occupied[
                    max(0, row_min) : min(height, row_max),
                    max(0, column_min) : min(width, column_max),
                ]
            ):
                continue

            row_indices, column_indices = zip(*pixels)
            occupied[row_indices, column_indices] = True
            components.append(pixels)
            break
        else:
            raise RuntimeError("Could not place the missing scratches.")

    return components


def load_image(image_id: str):
    image_path = DATA_DIRECTORY / f"{image_id}.jpg"
    with Image.open(image_path) as source:
        truth = np.asarray(source.convert("L"), dtype=np.float64) / 255.0

    client_blocks = split_into_client_blocks(truth.shape)
    scratch_components = choose_missing_scratches(
        truth.shape, DATA_SEED + int(image_id)
    )
    missing_mask = np.zeros(truth.shape, dtype=bool)
    for component in scratch_components:
        row_indices, column_indices = zip(*component)
        missing_mask[row_indices, column_indices] = True

    observed_mask = ~missing_mask
    corrupted = truth.copy()
    corrupted[missing_mask] = 0.0
    return (
        truth,
        corrupted,
        observed_mask,
        missing_mask,
        client_blocks,
        scratch_components,
    )


def client_block_assignments(maximum_level: int, seed: int) -> list[list[int]]:
    """Give every base block exact k-fold coverage at level k."""

    rng = random.Random(seed)
    base_order = list(range(CLIENTS_NUMBER))
    rng.shuffle(base_order)
    offsets = [0] + rng.sample(range(1, CLIENTS_NUMBER), maximum_level - 1)
    layers = [
        [
            base_order[(client + offset) % CLIENTS_NUMBER]
            for client in range(CLIENTS_NUMBER)
        ]
        for offset in offsets
    ]
    return [
        [layers[level][client] for level in range(maximum_level)]
        for client in range(CLIENTS_NUMBER)
    ]


def make_client_masks(
    shape: tuple[int, int],
    observed_mask: np.ndarray,
    client_blocks: list[tuple[int, int, int, int]],
    assignments: list[list[int]],
    heterogeneity_level: int,
) -> np.ndarray:
    masks = np.zeros((CLIENTS_NUMBER,) + shape, dtype=bool)
    for client in range(CLIENTS_NUMBER):
        for block_index in assignments[client][:heterogeneity_level]:
            row_start, row_end, column_start, column_end = client_blocks[block_index]
            masks[client, row_start:row_end, column_start:column_end] = True
        masks[client] &= observed_mask
    return masks


def make_local_objectives(
    truth: FloatArray,
    client_masks: np.ndarray,
    heterogeneity_level: int,
) -> list[ObjectiveFunction]:
    scale = CLIENTS_NUMBER / heterogeneity_level
    local_functions = []

    for mask in client_masks:

        def local_objective(point: FloatArray, mask=mask):
            residual = mask * (point - truth)
            value = 0.5 * scale * np.sum(residual**2)
            value += 0.5 * RIDGE_MU * np.sum(point**2)
            gradient = scale * residual + RIDGE_MU * point
            return np.float64(value), gradient

        local_functions.append(local_objective)

    return local_functions


# ======================================================================
# TV proximal operators and centralized reference point
# ======================================================================


def tv_proximal(
    point: FloatArray,
    parameter: np.float64,
    constant: np.float64,
    only_value: bool,
):
    return TV(
        point,
        parameter,
        constant,
        only_value,
        accuracy=np.float64(TV_PROXIMAL_ACCURACY),
        maximumIteration=TV_PROXIMAL_MAX_ITERATIONS,
    )


def reference_residual(
    point: FloatArray,
    truth: FloatArray,
    observed_mask: np.ndarray,
) -> float:
    step = 1.0 / (1.0 + RIDGE_MU)
    gradient = observed_mask * (point - truth) + RIDGE_MU * point
    proximal_point = TV(
        point - step * gradient,
        np.float64(step),
        np.float64(TV_REGULARIZATION),
        False,
        accuracy=np.float64(REFERENCE_TV_ACCURACY),
        maximumIteration=REFERENCE_TV_MAX_ITERATIONS,
    )
    return float(np.linalg.norm(point - proximal_point) / step)


def compute_reference_solution(
    truth: FloatArray,
    corrupted: FloatArray,
    observed_mask: np.ndarray,
    image_directory: Path,
) -> tuple[FloatArray, int, float]:
    """Compute and checkpoint the centralized FISTA reference point."""

    checkpoint_path = image_directory / "reference_checkpoint.npz"
    point = corrupted.copy()
    accelerated_point = point.copy()
    momentum = 1.0
    start_iteration = 0
    residual = np.inf

    if CONTINUE_PREVIOUS_RUN and checkpoint_path.exists():
        with np.load(checkpoint_path) as saved:
            point = saved["point"].copy()
            accelerated_point = saved["accelerated_point"].copy()
            momentum = float(saved["momentum"])
            start_iteration = int(saved["iteration"])
            residual = float(saved["residual"])
        if residual <= REFERENCE_TOLERANCE or start_iteration == REFERENCE_MAX_ITERATIONS:
            return point, start_iteration, residual

    step = 1.0 / (1.0 + RIDGE_MU)
    for iteration in range(start_iteration + 1, REFERENCE_MAX_ITERATIONS + 1):
        gradient = (
            observed_mask * (accelerated_point - truth) + RIDGE_MU * accelerated_point
        )
        next_point = TV(
            accelerated_point - step * gradient,
            np.float64(step),
            np.float64(TV_REGULARIZATION),
            False,
            accuracy=np.float64(REFERENCE_TV_ACCURACY),
            maximumIteration=REFERENCE_TV_MAX_ITERATIONS,
        )
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated_point = next_point + ((momentum - 1.0) / next_momentum) * (
            next_point - point
        )
        point = np.asarray(next_point, dtype=np.float64)
        momentum = next_momentum

        if iteration % REFERENCE_CHECK_INTERVAL == 0:
            residual = reference_residual(point, truth, observed_mask)
            np.savez_compressed(
                checkpoint_path,
                point=point,
                accelerated_point=accelerated_point,
                momentum=momentum,
                iteration=iteration,
                residual=residual,
            )
            print(
                f"  FISTA iteration {iteration}: residual={residual:.3e}",
                flush=True,
            )
            if residual <= REFERENCE_TOLERANCE or iteration == REFERENCE_MAX_ITERATIONS:
                return point, iteration, residual
    


# ======================================================================
# Run one distributed algorithm and save its last state
# ======================================================================


def reconstruction_metrics(
    point: FloatArray,
    reference_point: FloatArray,
    truth: FloatArray,
    missing_mask: np.ndarray,
) -> tuple[float, float]:
    error = float(np.sum((point - reference_point) ** 2))
    mse = float(np.mean((point[missing_mask] - truth[missing_mask]) ** 2))
    psnr = 120.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)
    return error, float(psnr)


def initial_history(
    initial_point: FloatArray,
    reference_point: FloatArray,
    truth: FloatArray,
    missing_mask: np.ndarray,
) -> dict[str, object]:
    error, psnr = reconstruction_metrics(
        initial_point, reference_point, truth, missing_mask
    )
    return {
        "round": np.array([0], dtype=int),
        "communication": np.array([0], dtype=int),
        "error": np.array([error]),
        "psnr": np.array([psnr]),
        "oracle": np.array([0], dtype=int),
        "time": np.array([0.0]),
        "point": initial_point.copy(),
    }


def run_algorithm(
    name: str,
    initial_point: FloatArray,
    reference_point: FloatArray,
    truth: FloatArray,
    missing_mask: np.ndarray,
    client_masks: np.ndarray,
    local_functions: list[ObjectiveFunction],
    heterogeneity_level: int,
    algorithm_seed: int,
    algorithm_directory: Path,
    settings: dict[str, object],
) -> dict[str, object]:
    algorithm_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = algorithm_directory / "checkpoint.pkl"
    state = None
    history = initial_history(initial_point, reference_point, truth, missing_mask)

    if CONTINUE_PREVIOUS_RUN and checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("settings") != settings:
            raise ValueError(f"Checkpoint settings do not match: {checkpoint_path}")
        state = saved["state"]
        history = saved["history"]

    completed_rounds = int(history["round"][-1])
    additional_rounds = ITERATION_BUDGET - completed_rounds
    if additional_rounds <= 0:
        print(
            f"    loaded {name} checkpoint: {completed_rounds} rounds; "
            "no new iterations"
        )
        return history

    local_smoothness = CLIENTS_NUMBER / heterogeneity_level + RIDGE_MU
    algorithm_class = {
        "REC_FedDyn": REC_FedDyn,
        "FedDyn": FedDyn,
        "FedDR": FedDR,
        "iFedDR": iFedDR,
    }[name]
    algorithm = algorithm_class(
        InitialPoint=history["point"],
        IterationBudget=additional_rounds,
        ProxFunction=tv_proximal,
        ClientsNumber=CLIENTS_NUMBER,
        SamplingNumber=SAMPLING_NUMBER,
        SmoothL=np.float64(local_smoothness),
        ConvexMu=np.float64(RIDGE_MU),
        Constant=np.float64(TV_REGULARIZATION),
        RandomSeed=algorithm_seed,
    )

    if name == "REC_FedDyn":
        result = algorithm.go(
            PenaltyParameter=np.float64(PENALTY_PARAMETER),
            Sigma=np.float64(RELATIVE_ERROR),
            Function=local_functions,
            InitialState=state,
            RecordStatistics=False,
        )
    elif name == "FedDyn":
        scale = CLIENTS_NUMBER / heterogeneity_level
        result = algorithm.go_exact_diagonal(
            PenaltyParameter=np.float64(PENALTY_PARAMETER),
            Function=local_functions,
            LocalHessianDiagonal=scale * client_masks + RIDGE_MU,
            LocalLinearTerm=scale * client_masks * truth,
            InitialState=state,
            RecordStatistics=False,
        )
    elif name == "FedDR":
        result = algorithm.go(
            ProximalParameter=np.float64(FEDDR_PROXIMAL_PARAMETER),
            RelaxationParameter=np.float64(FEDDR_RELAXATION_PARAMETER),
            LocalUpdates=FEDDR_LOCAL_UPDATES,
            Function=local_functions,
            StationarityParameter=np.float64(1.0 / (1.0 + RIDGE_MU)),
            InitialState=state,
            RecordStatistics=False,
        )
    else:
        result = algorithm.go(
            ProximalParameter=np.float64(IFEDDR_PROXIMAL_PARAMETER),
            RelaxationParameter=np.float64(IFEDDR_RELAXATION_PARAMETER),
            SigmaSquared=np.float64(IFEDDR_SIGMA_SQUARED),
            InitialLocalUpdates=IFEDDR_INITIAL_LOCAL_UPDATES,
            Function=local_functions,
            RefinementFactor=np.float64(IFEDDR_REFINEMENT_FACTOR),
            MaxRefinements=IFEDDR_MAX_REFINEMENTS,
            IncreaseInitialUpdates=IFEDDR_INCREASE_INITIAL_UPDATES,
            StationarityParameter=np.float64(1.0 / (1.0 + RIDGE_MU)),
            InitialState=state,
            RecordStatistics=False,
        )

    errors = []
    psnr_values = []
    for point in result.iterate_point:
        error, psnr = reconstruction_metrics(
            point, reference_point, truth, missing_mask
        )
        errors.append(error)
        psnr_values.append(psnr)

    new_rounds = np.arange(completed_rounds + 1, ITERATION_BUDGET + 1, dtype=int)
    if name == "iFedDR":
        new_communications = result.communication_rounds
    else:
        new_communications = new_rounds

    history["round"] = np.concatenate((history["round"], new_rounds))
    history["communication"] = np.concatenate(
        (history["communication"], new_communications)
    )
    history["error"] = np.concatenate((history["error"], errors))
    history["psnr"] = np.concatenate((history["psnr"], psnr_values))
    history["oracle"] = np.concatenate(
        (
            history["oracle"],
            int(history["oracle"][-1]) + np.cumsum(result.oracle_calling, dtype=int),
        )
    )
    history["time"] = np.concatenate((history["time"], result.process_time))
    history["point"] = result.iterate_point[-1].copy()

    # Save once after this run.  The user is responsible for keeping the same
    # parameters when CONTINUE_PREVIOUS_RUN is True.
    with checkpoint_path.open("wb") as stream:
        pickle.dump(
            {
                "settings": settings,
                "state": algorithm.final_state,
                "history": history,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    table = np.column_stack(
        (
            history["round"],
            history["communication"],
            history["error"],
            history["psnr"],
            history["oracle"],
            history["time"],
        )
    )
    np.savetxt(
        algorithm_directory / "metrics.csv",
        table,
        delimiter=",",
        header="round,communication,error_squared,missing_psnr,oracle_calls,time_seconds",
        comments="",
    )
    return history


# ======================================================================
# Figures
# ======================================================================


def history_range(
    histories: list[dict[str, object]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a common communication axis and mean/min/max seed statistics."""

    if not histories:
        raise ValueError("At least one history is required.")
    communication = np.asarray(histories[0]["communication"])
    values = []
    for history in histories:
        current_communication = np.asarray(history["communication"])
        if not np.array_equal(current_communication, communication):
            raise ValueError("Seed histories do not share a communication axis.")
        values.append(np.asarray(history[key], dtype=np.float64))
    stacked = np.stack(values)
    return (
        communication,
        np.mean(stacked, axis=0),
        np.min(stacked, axis=0),
        np.max(stacked, axis=0),
    )


def plot_curve(
    histories: dict[str, list[dict[str, object]]],
    key: str,
    ylabel: str,
    output_path: Path,
    logarithmic: bool = False,
) -> None:
    colors = {
        "REC_FedDyn": "#1f77b4",
        "FedDyn": "#d62728",
        "FedDR": "#2ca02c",
        "iFedDR": "#9467bd",
    }

    line_styles = {
        "REC_FedDyn": "-",
        "FedDyn": "--",
        "FedDR": "--",
        "iFedDR": "--",
    }

    figure, axis = plt.subplots(figsize=(6.2, 4.4))

    for name, seed_histories in histories.items():
        communication, mean_values, minimum_values, maximum_values = history_range(
            seed_histories, key
        )

        if logarithmic:
            mean_values = np.maximum(mean_values, ERROR_FLOOR)
            minimum_values = np.maximum(minimum_values, ERROR_FLOOR)
            maximum_values = np.maximum(maximum_values, ERROR_FLOOR)
            axis.semilogy(
                communication,
                mean_values,
                label="REC_FedDyn" if name == "REC_FedDyn" else name,
                color=colors[name],
                linestyle=line_styles[name],
                linewidth=LINE_WIDTH,
            )
        else:
            axis.plot(
                communication,
                mean_values,
                label="REC_FedDyn" if name == "REC_FedDyn" else name,
                color=colors[name],
                linestyle=line_styles[name],
                linewidth=LINE_WIDTH,
            )
        axis.fill_between(
            communication,
            minimum_values,
            maximum_values,
            color=colors[name],
            alpha=0.16,
            linewidth=0.0,
        )

    axis.set_xlabel("Communication rounds")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", linestyle=":", alpha=0.55)
    if key == "error":
        legend_location = "lower left" if SAMPLING_NUMBER == 5 else "upper right"
    else:
        legend_location = "best"
    axis.legend(loc=legend_location)

    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_results(
    histories: dict[str, list[dict[str, object]]],
    truth: FloatArray,
    corrupted: FloatArray,
    missing_mask: np.ndarray,
    reference_point: FloatArray,
    experiment_directory: Path,
) -> None:
    plot_curve(
        histories,
        "error",
        r"Squared iterate error $\|x^r-x^\star\|_F^2$",
        experiment_directory / "iterate_error_vs_communication.pdf",
        logarithmic=True,
    )
    plot_curve(
        histories,
        "psnr",
        "Missing-region PSNR (dB)",
        experiment_directory / "reconstruction_quality_vs_communication.pdf",
    )

    images = {
        "ground_truth": truth,
        "observed_image": corrupted,
        "fista_reference": reference_point,
        **{
            f"{name.lower()}_reconstruction": np.mean(
                np.stack([history["point"] for history in seed_histories]),
                axis=0,
            )
            for name, seed_histories in histories.items()
        },
    }
    height, width = truth.shape
    figure_width = 5.0
    figure_height = figure_width * height / width
    for filename, image in images.items():
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        axis.contour(
            missing_mask.astype(int),
            levels=[0.5],
            colors=["red"],
            linewidths=0.8,
        )
        axis.axis("off")
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        figure.savefig(
            experiment_directory / f"{filename}.pdf",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(figure)


# ======================================================================
# Main experiment
# ======================================================================


def settings_key(settings: dict[str, object]) -> str:
    encoded = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def reference_settings() -> dict[str, object]:
    return {
        "data_seed": DATA_SEED,
        "scratch_number": SCRATCH_NUMBER,
        "scratch_width": SCRATCH_WIDTH,
        "scratch_length": SCRATCH_LENGTH,
        "scratch_boundary_margin": SCRATCH_BOUNDARY_MARGIN,
        "ridge_mu": RIDGE_MU,
        "tv_regularization": TV_REGULARIZATION,
        "reference_tv_accuracy": REFERENCE_TV_ACCURACY,
        "reference_tv_max_iterations": REFERENCE_TV_MAX_ITERATIONS,
    }


def problem_settings(reference_key: str) -> dict[str, object]:
    return {
        "format": "bsds300_tv_inpainting_problem_v1",
        "reference_key": reference_key,
        "clients_number": CLIENTS_NUMBER,
        "sampling_number": SAMPLING_NUMBER,
        "data_seed": DATA_SEED,
        "ridge_mu": RIDGE_MU,
        "tv_regularization": TV_REGULARIZATION,
    }


def algorithm_settings(
    name: str,
    problem_key: str,
    image_id: str,
    heterogeneity_level: int,
    algorithm_seed: int = ALGORITHM_SEEDS[0],
) -> dict[str, object]:
    settings = {
        "format": "bsds300_tv_inpainting_algorithm_v1",
        "algorithm": name,
        "problem_key": problem_key,
        "image_id": image_id,
        "heterogeneity_level": heterogeneity_level,
        "algorithm_seed": algorithm_seed,
        "tv_proximal_accuracy": TV_PROXIMAL_ACCURACY,
        "tv_proximal_max_iterations": TV_PROXIMAL_MAX_ITERATIONS,
    }
    if name == "REC_FedDyn":
        settings.update(
            stepsize=OVERALL_STEPSIZE,
            penalty_parameter=PENALTY_PARAMETER,
            relative_error=RELATIVE_ERROR,
        )
    elif name == "FedDyn":
        settings.update(
            stepsize=OVERALL_STEPSIZE,
            penalty_parameter=PENALTY_PARAMETER,
            subproblem_solver="exact_diagonal",
        )
    elif name == "FedDR":
        settings.update(
            proximal_parameter=FEDDR_PROXIMAL_PARAMETER,
            relaxation_parameter=FEDDR_RELAXATION_PARAMETER,
            local_updates=FEDDR_LOCAL_UPDATES,
        )
    elif name == "iFedDR":
        settings.update(
            proximal_parameter=IFEDDR_PROXIMAL_PARAMETER,
            relaxation_parameter=IFEDDR_RELAXATION_PARAMETER,
            sigma_squared=IFEDDR_SIGMA_SQUARED,
            initial_local_updates=IFEDDR_INITIAL_LOCAL_UPDATES,
            refinement_factor=IFEDDR_REFINEMENT_FACTOR,
            max_refinements=IFEDDR_MAX_REFINEMENTS,
            increase_initial_updates=IFEDDR_INCREASE_INITIAL_UPDATES,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    return settings


def algorithm_run_directory(
    problem_directory: Path,
    image_id: str,
    heterogeneity_level: int,
    name: str,
    settings: dict[str, object],
) -> Path:
    return (
        problem_directory
        / image_id
        / f"k_{heterogeneity_level}"
        / "algorithms"
        / name
        / f"run_{settings_key(settings)}"
    )


def save_settings(directory: Path, settings: dict[str, object]) -> None:
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


def compatible_reference_keys(reference_parameters: dict[str, object]) -> set[str]:
    """Accept old reference folders that differ only by obsolete metadata."""
    keys = {settings_key(reference_parameters)}
    references_directory = OUTPUT_DIRECTORY / "references"
    if not references_directory.exists():
        return keys

    for directory in references_directory.glob("reference_*"):
        settings_path = directory / "settings.json"
        if not settings_path.exists():
            continue
        with settings_path.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        try:
            normalized = {key: saved[key] for key in reference_parameters}
        except KeyError:
            continue
        if normalized == reference_parameters:
            keys.add(directory.name.removeprefix("reference_"))
    return keys


def reuse_legacy_reference(
    reference_directory: Path,
    image_id: str,
    compatible_keys: set[str],
) -> None:
    destination = reference_directory / "reference_checkpoint.npz"
    if not CONTINUE_PREVIOUS_RUN or destination.exists():
        return

    candidates = []
    for key in compatible_keys:
        source = (
            OUTPUT_DIRECTORY
            / "references"
            / f"reference_{key}"
            / image_id
            / "reference_checkpoint.npz"
        )
        if not source.exists() or source == destination:
            continue
        with np.load(source) as saved:
            candidates.append((int(saved["iteration"]), source))

    if candidates:
        iteration, source = max(candidates, key=lambda item: item[0])
        reference_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"  reused reference checkpoint ({iteration} iterations): {source}")


def legacy_problem_matches(
    saved: dict[str, object],
    current: dict[str, object],
    reference_keys: set[str],
) -> bool:
    if saved.get("reference_key") not in reference_keys:
        return False
    names = (
        "clients_number",
        "sampling_number",
        "data_seed",
        "ridge_mu",
        "tv_regularization",
    )
    return all(saved.get(name) == current[name] for name in names)


def legacy_algorithm_settings(
    name: str,
    saved: dict[str, object],
    problem_key: str,
    image_id: str,
    heterogeneity_level: int,
) -> dict[str, object] | None:
    try:
        settings = {
            "format": "bsds300_tv_inpainting_algorithm_v1",
            "algorithm": name,
            "problem_key": problem_key,
            "image_id": image_id,
            "heterogeneity_level": heterogeneity_level,
            "algorithm_seed": saved["algorithm_seed"],
            "tv_proximal_accuracy": saved["tv_proximal_accuracy"],
            "tv_proximal_max_iterations": saved["tv_proximal_max_iterations"],
        }
        if name == "REC_FedDyn":
            stepsize = saved["overall_stepsize"]
            settings.update(
                stepsize=stepsize,
                penalty_parameter=1.0 / stepsize,
                relative_error=saved["relative_error"],
            )
        elif name == "FedDyn":
            stepsize = saved["overall_stepsize"]
            settings.update(
                stepsize=stepsize,
                penalty_parameter=1.0 / stepsize,
                subproblem_solver="exact_diagonal",
            )
        elif name == "FedDR":
            settings.update(
                proximal_parameter=saved["feddr_proximal_parameter"],
                relaxation_parameter=saved["feddr_relaxation_parameter"],
                local_updates=saved["feddr_local_updates"],
            )
        elif name == "iFedDR":
            settings.update(
                proximal_parameter=saved["ifeddr_proximal_parameter"],
                relaxation_parameter=saved["ifeddr_relaxation_parameter"],
                sigma=saved["ifeddr_sigma"],
                initial_local_updates=saved["ifeddr_initial_local_updates"],
                refinement_factor=saved["ifeddr_refinement_factor"],
                max_refinements=saved["ifeddr_max_refinements"],
                increase_initial_updates=saved[
                    "ifeddr_increase_initial_updates"
                ],
            )
        else:
            raise ValueError(f"Unknown algorithm: {name}")
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    return settings


def reuse_legacy_comparison_checkpoint(
    name: str,
    image_id: str,
    heterogeneity_level: int,
    problem_key: str,
    problem_parameters: dict[str, object],
    current_settings: dict[str, object],
    algorithm_directory: Path,
    reference_keys: set[str],
) -> None:
    """Copy a compatible old comparison checkpoint; never reuse REC_FedDyn."""
    destination = algorithm_directory / "checkpoint.pkl"
    if (
        name == "REC_FedDyn"
        or not CONTINUE_PREVIOUS_RUN
        or destination.exists()
    ):
        return

    candidates = []
    pattern = f"run_*/{image_id}/k_{heterogeneity_level}/{name}_checkpoint.pkl"
    for source in (OUTPUT_DIRECTORY / "runs").glob(pattern):
        with source.open("rb") as stream:
            checkpoint = pickle.load(stream)
        saved_settings = checkpoint.get("settings", {})
        if not legacy_problem_matches(
            saved_settings, problem_parameters, reference_keys
        ):
            continue
        converted_settings = legacy_algorithm_settings(
            name,
            saved_settings,
            problem_key,
            image_id,
            heterogeneity_level,
        )
        if converted_settings != current_settings:
            continue
        completed_rounds = int(checkpoint["history"]["round"][-1])
        candidates.append((completed_rounds, source, checkpoint))

    if not candidates:
        return

    completed_rounds, source, checkpoint = max(
        candidates, key=lambda item: item[0]
    )
    algorithm_directory.mkdir(parents=True, exist_ok=True)
    converted = dict(checkpoint)
    converted["settings"] = current_settings
    with destination.open("wb") as stream:
        pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)

    source_metrics = source.with_name(f"{name}_metrics.csv")
    if source_metrics.exists():
        shutil.copy2(source_metrics, algorithm_directory / "metrics.csv")
    print(
        f"    reused legacy {name} checkpoint "
        f"({completed_rounds} rounds): {source}"
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    reference_parameters = reference_settings()
    reference_key = settings_key(reference_parameters)
    reference_root = OUTPUT_DIRECTORY / "references" / f"reference_{reference_key}"
    save_settings(reference_root, reference_parameters)
    reference_keys = compatible_reference_keys(reference_parameters)

    problem_parameters = problem_settings(reference_key)
    problem_key = settings_key(problem_parameters)
    problem_directory = OUTPUT_DIRECTORY / "runs" / f"problem_{problem_key}"
    save_settings(problem_directory, problem_parameters)
    print(f"Problem directory: {problem_directory}")

    for image_id in IMAGE_IDS:
        (
            truth,
            corrupted,
            observed_mask,
            missing_mask,
            client_blocks,
            scratch_components,
        ) = load_image(image_id)

        reference_directory = reference_root / image_id
        reference_directory.mkdir(parents=True, exist_ok=True)
        reuse_legacy_reference(reference_directory, image_id, reference_keys)
        image_directory = problem_directory / image_id
        image_directory.mkdir(parents=True, exist_ok=True)

        print("=" * 72)
        print(f"Image {image_id}")
        print(f"Missing pixels: {int(np.sum(missing_mask))}")

        reference_point, reference_iterations, reference_error = (
            compute_reference_solution(
                truth,
                corrupted,
                observed_mask,
                reference_directory,
            )
        )
        print(
            f"Reference: {reference_iterations} iterations, "
            f"residual={reference_error:.3e}"
        )

        assignments = client_block_assignments(
            max(HETEROGENEITY_LEVELS), DATA_SEED + int(image_id) + 1
        )

        for heterogeneity_level in HETEROGENEITY_LEVELS:
            experiment_directory = image_directory / f"k_{heterogeneity_level}"
            experiment_directory.mkdir(parents=True, exist_ok=True)

            client_masks = make_client_masks(
                truth.shape,
                observed_mask,
                client_blocks,
                assignments,
                heterogeneity_level,
            )
            local_functions = make_local_objectives(
                truth, client_masks, heterogeneity_level
            )

            histories = {name: [] for name in ALGORITHM_NAMES}
            for algorithm_seed in ALGORITHM_SEEDS:
                print(f"  k={heterogeneity_level}, algorithm seed={algorithm_seed}")
                for name in ALGORITHM_NAMES:
                    current_settings = algorithm_settings(
                        name,
                        problem_key,
                        image_id,
                        heterogeneity_level,
                        algorithm_seed,
                    )
                    algorithm_directory = algorithm_run_directory(
                        problem_directory,
                        image_id,
                        heterogeneity_level,
                        name,
                        current_settings,
                    )
                    save_settings(algorithm_directory, current_settings)

                    print(f"    processing {name}")
                    history = run_algorithm(
                        name,
                        corrupted,
                        reference_point,
                        truth,
                        missing_mask,
                        client_masks,
                        local_functions,
                        heterogeneity_level,
                        algorithm_seed,
                        algorithm_directory,
                        current_settings,
                    )
                    histories[name].append(history)
                    print(
                        f"      error={history['error'][-1]:.3e}, "
                        f"PSNR={history['psnr'][-1]:.3f} dB, "
                        f"oracle={history['oracle'][-1]}, "
                        f"time={history['time'][-1]:.3f} s"
                    )

            plot_results(
                histories,
                truth,
                corrupted,
                missing_mask,
                reference_point,
                experiment_directory,
            )


if __name__ == "__main__":
    main()
