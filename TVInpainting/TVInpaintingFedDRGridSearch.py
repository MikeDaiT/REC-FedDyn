"""Select one FedDR stepsize for each BSDS300 TV-inpainting image."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import TVInpaintingExperiment as experiment
from Algorithms.FedDR import FedDR
from Algorithms.ProximalOperator import TV


STEPSIZE_GRID = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0)
GRID_SEARCH_ITERATIONS = 100
GRID_SEARCH_HETEROGENEITY_LEVEL = 1
GRID_TV_ACCURACY = 1e-2
GRID_TV_MAX_ITERATIONS = 10

RESULT_DIRECTORY = experiment.OUTPUT_DIRECTORY / "FedDRStepsizeGridSearchRough"
CANDIDATE_RESULTS_PATH = RESULT_DIRECTORY / "candidate_results.csv"
SELECTED_STEPSIZES_PATH = RESULT_DIRECTORY / "selected_stepsizes.csv"


def load_candidate_results() -> dict[tuple[str, int, float], dict[str, object]]:
    results = {}
    if not CANDIDATE_RESULTS_PATH.exists():
        return results

    with CANDIDATE_RESULTS_PATH.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            heterogeneity_level = int(row["heterogeneity_level"])
            if heterogeneity_level != GRID_SEARCH_HETEROGENEITY_LEVEL:
                continue
            key = (row["image_id"], heterogeneity_level, float(row["stepsize"]))
            results[key] = {
                "image_id": row["image_id"],
                "heterogeneity_level": heterogeneity_level,
                "stepsize": float(row["stepsize"]),
                "initial_value": float(row["initial_value"]),
                "final_value": float(row["final_value"]),
                "relative_decrease": float(row["relative_decrease"]),
            }
    return results


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grid_tv_proximal(point, parameter, constant, only_value):
    return TV(
        point,
        parameter,
        constant,
        only_value,
        accuracy=np.float64(GRID_TV_ACCURACY),
        maximumIteration=GRID_TV_MAX_ITERATIONS,
    )


def objective_value(
    point: np.ndarray,
    truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    residual = observed_mask * (point - truth)
    value = 0.5 * np.sum(residual**2)
    value += 0.5 * experiment.RIDGE_MU * np.sum(point**2)
    value += grid_tv_proximal(
        point,
        np.float64(1.0),
        np.float64(experiment.TV_REGULARIZATION),
        True,
    )
    return float(value)


def run_feddr(
    stepsize: float,
    truth: np.ndarray,
    corrupted: np.ndarray,
    observed_mask: np.ndarray,
    local_functions,
    heterogeneity_level: int,
) -> tuple[float, float, float]:
    local_smoothness = (
        experiment.CLIENTS_NUMBER / heterogeneity_level + experiment.RIDGE_MU
    )
    algorithm = FedDR(
        InitialPoint=corrupted,
        IterationBudget=GRID_SEARCH_ITERATIONS,
        ProxFunction=grid_tv_proximal,
        ClientsNumber=experiment.CLIENTS_NUMBER,
        SamplingNumber=experiment.SAMPLING_NUMBER,
        SmoothL=np.float64(local_smoothness),
        ConvexMu=np.float64(experiment.RIDGE_MU),
        Constant=np.float64(experiment.TV_REGULARIZATION),
        RandomSeed=experiment.ALGORITHM_SEED,
    )

    result = algorithm.go(
        ProximalParameter=np.float64(stepsize),
        RelaxationParameter=np.float64(experiment.FEDDR_RELAXATION_PARAMETER),
        LocalUpdates=experiment.FEDDR_LOCAL_UPDATES,
        Function=local_functions,
        StationarityParameter=np.float64(1.0 / (1.0 + experiment.RIDGE_MU)),
        RecordStatistics=False,
    )

    initial_value = objective_value(corrupted, truth, observed_mask)
    final_value = objective_value(result.iterate_point[-1], truth, observed_mask)
    relative_decrease = float(1.0 - final_value / initial_value)
    return initial_value, final_value, relative_decrease


def main() -> None:
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    candidate_results = load_candidate_results()

    for image_id in experiment.IMAGE_IDS:
        truth, corrupted, observed_mask, _, client_blocks, _ = (
            experiment.load_image(image_id)
        )
        print(f"Image {image_id}")

        assignments = experiment.client_block_assignments(
            GRID_SEARCH_HETEROGENEITY_LEVEL,
            experiment.DATA_SEED + int(image_id) + 1,
        )
        heterogeneity_level = GRID_SEARCH_HETEROGENEITY_LEVEL
        client_masks = experiment.make_client_masks(
            truth.shape,
            observed_mask,
            client_blocks,
            assignments,
            heterogeneity_level,
        )
        local_functions = experiment.make_local_objectives(
            truth,
            client_masks,
            heterogeneity_level,
        )

        for stepsize in STEPSIZE_GRID:
            key = (image_id, heterogeneity_level, stepsize)
            if key in candidate_results:
                if candidate_results[key]["relative_decrease"] < 0.0:
                    break
                continue

            initial_value, final_value, relative_decrease = run_feddr(
                stepsize,
                truth,
                corrupted,
                observed_mask,
                local_functions,
                heterogeneity_level,
            )
            candidate_results[key] = {
                "image_id": image_id,
                "heterogeneity_level": heterogeneity_level,
                "stepsize": stepsize,
                "initial_value": initial_value,
                "final_value": final_value,
                "relative_decrease": relative_decrease,
            }
            save_rows(CANDIDATE_RESULTS_PATH, list(candidate_results.values()))
            print(
                f"  k={heterogeneity_level}, stepsize={stepsize:g}, "
                f"final value={final_value:.6e}, "
                f"decrease={relative_decrease:.2%}",
                flush=True,
            )
            if relative_decrease < 0.0:
                break

    selected_stepsizes = []
    for image_id in experiment.IMAGE_IDS:
        rows = []
        for stepsize in STEPSIZE_GRID:
            key = (image_id, GRID_SEARCH_HETEROGENEITY_LEVEL, stepsize)
            if key not in candidate_results:
                continue
            row = candidate_results[key]
            rows.append(row)
            if row["relative_decrease"] < 0.0:
                break

        best = min(rows, key=lambda row: row["final_value"])
        selected_stepsizes.append(best)
        print(
            f"Selected: image={image_id}, "
            f"k={GRID_SEARCH_HETEROGENEITY_LEVEL}, "
            f"stepsize={best['stepsize']:g}"
        )

    save_rows(SELECTED_STEPSIZES_PATH, selected_stepsizes)


if __name__ == "__main__":
    main()
