# Distributed TV Inpainting on BSDS300

This experiment compares `REC_FedDyn`, FedDyn, FedDR, and iFedDR on a strongly convex distributed total-variation (TV) image-inpainting problem. Artificial scratches remove pixels from each input image, and the clients jointly recover the image using a data-fidelity term, ridge regularization, and TV
regularization.

This directory contains one formal comparison program and two parameter-selection programs.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `TVInpaintingExperiment.py` | **Main experiment** | Runs all four algorithms on six BSDS300 images, three heterogeneity levels, and the configured random seeds. It saves per-run data, aggregate statistics, reconstructed images, and comparison figures. |
| `TVInpaintingFedDRGridSearch.py` | **Parameter selection** | Tests a grid of FedDR proximal step sizes and selects one step size for each image. |
| `TVInpaintingFedDynGridSearch.py` | **Parameter selection** | Tests a grid of FedDyn step sizes and selects one step size for each image. |
| `ExperimentPlotStyle.py` | Plotting helper | Defines the common Matplotlib style used by the programs in this directory. |
| `ExperimentResults/` | Selected results | Archived formal results organized by client-sampling level. |
| `FedDRStepsizeGridSearchRough/` | Selected tuning results | Archived FedDR candidate and selected-step-size results. |
| `FedDynStepsizeGridSearchRough/` | Selected tuning results | Archived FedDyn candidate and selected-step-size results. |

The grid-search scripts are not additional main experiments. They explain how
the FedDR and FedDyn settings used by the formal comparison were selected.

## Required directory structure

Arrange the shared algorithms, experiment programs, input images, and generated
outputs as follows:

```text
repository root/
|-- Algorithms/
`-- TVInpainting/
    |-- TVInpaintingExperiment.py
    |-- TVInpaintingFedDRGridSearch.py
    |-- TVInpaintingFedDynGridSearch.py
    |-- ExperimentPlotStyle.py
    |-- Data/
    |   `-- BSDS300/
    |       `-- images/
    |           `-- test/
    |               |-- 102061.jpg
    |               |-- 108005.jpg
    |               |-- 167083.jpg
    |               |-- 208001.jpg
    |               |-- 241004.jpg
    |               `-- 86016.jpg
    |-- Results/                              # generated outputs
    |-- ExperimentResults/                    # selected archived results
    |-- FedDRStepsizeGridSearchRough/          # selected tuning results
    `-- FedDynStepsizeGridSearchRough/         # selected tuning results
```

Keep the BSDS300 images and newly generated `Results/` directory outside
version control.

## Main-experiment configuration

The default configuration in `TVInpaintingExperiment.py` uses 20 clients,
samples 5 clients per round, evaluates six images, and tests heterogeneity
levels 1, 2, and 3. The formal comparison uses five algorithm seeds. Edit the
constants near the top of the script to change these settings.

Required image IDs:

```text
102061.jpg
108005.jpg
167083.jpg
208001.jpg
241004.jpg
86016.jpg
```

By default, place them in:

```text
TVInpainting/Data/BSDS300/images/test/
```

The filename extension may follow the image-loading logic in the script, but
the numeric image IDs must remain unchanged.

## Running notes

Run with the repository root available on the Python import path. The two grid
searches are optional; use their `selected_stepsizes.csv` files only when
reproducing parameter selection. Before the main experiment, verify the image
directory, sampling level, iteration budget, seeds, and
`CONTINUE_PREVIOUS_RUN`.

The centralized reference and each algorithm use separate compatible
checkpoints. A resumed run restores the saved optimizer state and history;
`ITERATION_BUDGET` is cumulative, so a larger budget extends an earlier run.
Do not reuse checkpoints after changing the data partition, algorithm variant,
or incompatible parameters.

## Outputs

New runs are written under:

```text
TVInpainting/Results/BSDS300StronglyConvexTVInpainting/
```

The main program can resume compatible checkpoints when
`CONTINUE_PREVIOUS_RUN` is enabled. Increasing `ITERATION_BUDGET` continues a
completed shorter run to the new cumulative round budget.

The checked-in `ExperimentResults/` and grid-search directories contain
selected outputs for inspection. They are not used as input data by the main
experiment.

## Dataset link

[BSDS300 dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

<!-- Insert the official BSDS300 dataset URL in the empty parentheses above. -->
