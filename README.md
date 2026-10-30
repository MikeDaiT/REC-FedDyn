# Federated Composite Optimization Experiments

This directory contains the experiment code and selected results for comparing
federated optimization algorithms on image restoration, handwritten-character
classification, and fMRI classification tasks. The proposed method is named
`REC_FedDyn` throughout the code and documentation.

## Repository map

| Directory | Experiment type | Main task |
| --- | --- | --- |
| `TVInpainting/` | Main experiment | Distributed strongly convex TV inpainting on BSDS300 images |
| `MNISTNeuralNetworkL1/` | Main experiment | Federated MNIST classification with L1 regularization and Dirichlet heterogeneity |
| `FEMNISTNeuralNetworkOGL/` | Main experiment | FEMNIST-62 classification with overlapping group-lasso regularization |
| `HaxbyTVLogisticRegression/` | Main experiment | Haxby face-versus-house logistic regression with 3-D TV regularization |
| `Algorithms/` | Shared implementation | Optimizers, proximal operators, tensor helpers, and fMRI model utilities |
| `Algorithms/AlgorithmsValidation/` | Independent validation experiments | Reproductions used to check FedDR, FedNMap, and iFedDR implementations |

Each experiment directory has its own README with the required data layout,
the purpose of every program file, the recommended execution order, and the
output locations.

## Main experiments and parameter selection

The four top-level experiment directories contain two kinds of programs:

1. **Parameter-selection programs** test candidate step sizes or related
   hyperparameters on a controlled configuration.
2. **Main-experiment programs** use the selected settings to run the formal
   multi-algorithm comparisons and create result tables and figures.

The parameter-selection programs are documented for reproducibility. They do
not need to be rerun when the selected settings already stored in the main
scripts are being used.

### REC-FedDyn implementation used by each experiment

| Experiment | Implementation |
| --- | --- |
| TV inpainting | Corrected version: `Algorithms/REC_FedDyn_Correct.py` |
| Haxby fMRI classification | Corrected version: `Algorithms/REC_FedDyn_Correct.py` |
| MNIST classification | Standard version: `Algorithms/REC_FedDyn.py` |
| FEMNIST classification | Standard version: `Algorithms/REC_FedDyn.py` |

The main TV inpainting and Haxby programs import the corrected implementation
directly. MNIST and FEMNIST retain the standard implementation. Do not reuse a
`REC_FedDyn` checkpoint after switching between these two implementations.

## Shared algorithm files

| File | Purpose |
| --- | --- |
| `Algorithms/REC_FedDyn.py` | Standard `REC_FedDyn` implementation used by MNIST and FEMNIST |
| `Algorithms/REC_FedDyn_Correct.py` | Corrected `REC_FedDyn` implementation used by TV inpainting and Haxby |
| `Algorithms/FedDR.py` | FedDR implementation |
| `Algorithms/FedDyn.py` | FedDyn implementation |
| `Algorithms/iFedDR.py` | Inexact FedDR implementation |
| `Algorithms/FedNMap.py` | FedNMap implementation |
| `Algorithms/Zhang.py` | Zhang-method implementation used in the neural-network experiments |
| `Algorithms/ProximalOperator.py` | L1, TV, and overlapping group-lasso proximal operators |
| `Algorithms/Base_Optimizer.py` | Shared optimizer interfaces and result containers |
| `Algorithms/TorchUtils.py` | PyTorch model and parameter-vector utilities |
| `Algorithms/FMRI.py` | fMRI model structures and objective helpers |

The algorithm copies inside `Algorithms/AlgorithmsValidation/*/Algorithms/`
belong to their individual reproduction experiments. Keep those local copies
with the corresponding validation scripts so that a validation run is not
silently changed by later edits to the shared implementations.

## General workflow

Run the experiment programs with the repository root available on the Python
import path so that the top-level `Algorithms` package can be imported.

Then follow the README in the selected experiment directory. In general:

1. Download and prepare the dataset outside version control.
2. Fill the intentionally blank path variables or pass the documented command-line path.
3. Optionally rerun the parameter-selection program.
4. Copy the selected settings into the main experiment if they differ from the stored defaults.
5. Run the main script and inspect its CSV, checkpoint, and figure outputs.

Checkpoint behavior is experiment-specific. TV inpainting, Haxby, and some
validation programs save optimizer state that can continue an interrupted
run. MNIST and FEMNIST neural-network programs mainly use completed-result
caches: a finished, settings-matched run is skipped, but an interrupted run is
normally recomputed. The corresponding `Running notes` explain the distinction
and should be reviewed before a long experiment is started.

Large datasets, machine-specific absolute paths, temporary checkpoints, and
generated feature caches should not be committed to GitHub.

## Software environment

The experiments are written for Python and use NumPy, PyTorch, and Matplotlib.
Additional task-specific packages include Pillow for image experiments,
`pyarrow` for FEMNIST Parquet input, pandas and nibabel for Haxby outputs, and
torchvision for the CIFAR-10 validation experiment. Direct dependencies are
listed in `requirements.txt` without version pins. For exact reproduction,
record the package, Python, and CUDA versions used for the reported runs. GPU
use is optional in scripts that expose a `--device` argument, but CPU and GPU
runs can have different runtimes and small numerical differences.

## Reproducibility notes

- Random seeds and main experiment constants are defined near the top of each
  script.
- Result directories may contain resumable checkpoints. Do not combine
  checkpoints produced with different data partitions or parameter settings.
- The checked-in result folders are selected artifacts, not substitutes for
  the original datasets.
- Before publishing, verify that no raw data, private paths, credentials, or
  large temporary checkpoints are tracked by Git.

## Citation

If this code supports a paper, add its citation and BibTeX entry here before
the repository is made public.

[Paper page]()

<!-- Insert the public paper URL in the empty parentheses above. -->

## External resources

[Project page]()

<!-- Insert the public project URL in the empty parentheses above. -->

[Dataset sources]()

<!-- Insert a page listing the official dataset download URLs in the empty parentheses above. -->

## License

This repository is released under the MIT License. See `LICENSE` for details.
