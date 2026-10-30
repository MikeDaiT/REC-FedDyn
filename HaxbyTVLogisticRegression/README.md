# Haxby fMRI TV-Regularized Logistic Regression

This experiment compares `REC_FedDyn`, FedDR, and iFedDR on Haxby
face-versus-house classification. Each client corresponds to an fMRI run, and
the logistic-regression image weights are regularized with three-dimensional
total variation.

The workflow contains reference-solution computation, two parameter-selection
approaches, and one formal comparison program.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `HaxbyLogisticRegressionExperiment.py` | **Main experiment** | Runs the formal multi-seed algorithm comparison, supports compatible checkpoint continuation, and saves individual and aggregate results. |
| `ComputeHaxbyReference.py` | **Reference computation** | Computes or resumes a high-accuracy centralized reference point used to evaluate optimization error. It is not launched automatically by the main experiment. |
| `HaxbyStepsizeSelection.py` | **Parameter selection: factor form** | Tests step-size factors relative to the local smoothness bound and ranks candidates using normalized log-error area under the curve. |
| `HaxbyOverallStepsizeSelection.py` | **Parameter selection: direct form** | Tests direct overall step sizes. FedDR/iFedDR use the candidate directly, while `REC_FedDyn` uses the equivalent reciprocal penalty. |
| `HaxbyExperimentOutput.py` | Output helper | Saves per-run histories, aggregate CSV data, and plots used by the main experiment. |
| `load_haxby.py` | Data-loading helper | Loads the preprocessed memory-mapped arrays and constructs run-level clients. |
| `ExperimentPlotStyle.py` | Plotting helper | Applies the common Matplotlib style. |
| `ExperimentResults/` | Selected results | Archived formal results for sampled-client counts 15, 30, and 59. |
| `step_size_selection/` | Selected tuning results | Archived parameter-selection results. |

The two step-size programs are alternative tuning records. The direct
overall-step-size script does not modify or reuse the older factor-based
candidate directories.

## Main-experiment configuration

The stored main configuration defines 59 clients, samples 30 clients per
round, and runs up to 15,000 communication rounds. Partial-participation runs
use five seeds; full participation uses one seed. Change `SAMPLING_NUMBER` to
15, 30, or 59 to reproduce the corresponding archived configuration.

## Required directory structure

The default processed-data location is:

```text
HaxbyTVLogisticRegression/Data/Haxby/processed/
```

It must contain:

```text
manifest.json
train_X.npy
train_y.npy
test_X.npy
test_y.npy
train_metadata.csv
test_metadata.csv
common_brain_mask_3mm.nii.gz
model_reference_mni152_3mm.nii.gz
```

The `.npy` arrays are loaded with memory mapping to avoid duplicating the full
fMRI data in memory. Keep the processed dataset outside the Git upload.

The code imports the loader as `Data.Haxby.load_haxby`, while the processed
arrays are read from the experiment directory. Construct the folders as
follows before running the experiment:

```text
repository root/
|-- Algorithms/
|-- Data/
|   `-- Haxby/
|       `-- load_haxby.py
`-- HaxbyTVLogisticRegression/
    |-- HaxbyLogisticRegressionExperiment.py
    |-- ComputeHaxbyReference.py
    |-- load_haxby.py
    `-- Data/
        `-- Haxby/
            `-- processed/
                |-- manifest.json
                |-- train_X.npy
                |-- train_y.npy
                |-- test_X.npy
                |-- test_y.npy
                |-- train_metadata.csv
                |-- test_metadata.csv
                |-- common_brain_mask_3mm.nii.gz
                `-- model_reference_mni152_3mm.nii.gz
```

Create the top-level `Data/Haxby/` directory and place a copy of the included
`HaxbyTVLogisticRegression/load_haxby.py` at
`Data/Haxby/load_haxby.py`. Separately, create
`HaxbyTVLogisticRegression/Data/Haxby/processed/` and place the processed data
files there. The duplicate loader beside the experiment is the source copy
provided with this repository; the top-level copy is the import location used
by the current programs. The processed dataset should remain outside version
control.

## Recommended execution order

1. Prepare the processed Haxby data.
2. Place an accepted reference cache at the `REFERENCE_FILE` location, or run
   `ComputeHaxbyReference.py` to compute/resume it.
3. Optionally reproduce one of the step-size-selection procedures.
4. Copy the selected setting to `STEP_SIZE_FACTOR` in the main script when
   necessary.
5. Run the formal comparison.

The formal experiment does not automatically recompute the reference. If an
accepted reference cache already exists, the reference-computation step can be
skipped.

## Running notes

Run the programs with the repository root available on the Python import path
so the shared `Algorithms` package and top-level `Data.Haxby` loader can be
imported. Before a formal run, verify `SAMPLING_NUMBER`, the cumulative round
budget, step-size factor, seeds, reference paths, and
`CONTINUE_PREVIOUS_RUN`.

Haxby saves complete resumable checkpoints every `PROGRESS_INTERVAL = 1000`
rounds. A compatible checkpoint restores the optimizer state, final model,
and metric history; increasing `ITERATION_BUDGET` extends the saved run. Do not
reuse checkpoints after changing the data partition, algorithm variant, or
incompatible settings. An interruption inside an unfinished segment may
require that segment to be rerun.

## Outputs and continuation

New main runs are written under
`Results/HaxbyTVLogisticRegression/`. The smoothness cache, reference cache,
per-algorithm checkpoints, aggregate results, and figures are separated by
purpose. Tuning candidates use isolated subdirectories so that they cannot be
mistaken for formal checkpoints.

When `CONTINUE_PREVIOUS_RUN` is enabled, the main and tuning programs resume
compatible candidate-specific checkpoints. `ITERATION_BUDGET` is cumulative,
so increasing it extends an earlier run.

## Dataset link

[Haxby dataset page](https://nilearn.github.io/dev/modules/description/haxby2001.html)

<!-- Insert the official Haxby dataset URL in the empty parentheses above. -->
