# Federated MNIST Neural Network with L1 Regularization

This experiment compares `REC_FedDyn`, FedDR, iFedDR, FedNMap, and Zhang on a
shallow neural network for MNIST classification. The client data are generated
with a Dirichlet label split, and an L1 penalty is applied to the model
parameters.

The formal comparison and the parameter-selection sweep are separate programs.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `MNISTExperiment.py` | **Main experiment** | Runs the five-algorithm comparison over the configured Dirichlet concentrations and random seeds, then saves histories, aggregate tables, and figures. |
| `MNISTStepsizeTest.py` | **Parameter selection** | Sweeps common overall step-size candidates for three algorithm groups: `REC_FedDyn`; FedDR/iFedDR; and FedNMap/Zhang. |
| `ExperimentPlotStyle.py` | Plotting helper | Applies the common figure style. |
| `ExperimentResults/` | Selected results | Archived main-experiment outputs grouped by sampled-client count and Dirichlet concentration. |
| `MNISTStepsizeTest/` | Selected tuning results | Archived step-size candidate histories and summaries for each algorithm. |

`MNISTStepsizeTest.py` is a tuning experiment, not the formal result-producing
script. Use `MNISTExperiment.py` for the main reported comparison.

## Required directory structure

The scripts use blank path placeholders, so the data may be stored elsewhere.
The following layout is recommended and can be selected through the path
constants in both programs:

```text
repository root/
|-- Algorithms/
`-- MNISTNeuralNetworkL1/
    |-- MNISTExperiment.py
    |-- MNISTStepsizeTest.py
    |-- ExperimentPlotStyle.py
    |-- Data/
    |   `-- MNIST/
    |       |-- train-images-idx3-ubyte
    |       |-- train-labels-idx1-ubyte
    |       |-- t10k-images-idx3-ubyte
    |       `-- t10k-labels-idx1-ubyte
    |-- Results/                  # generated outputs
    |-- ExperimentResults/        # selected archived results
    `-- MNISTStepsizeTest/        # selected tuning results
```

Set `DATA_DIRECTORY` to the folder containing the four IDX files. Set the
project/output path variables to the location where `Results/` should be
created. Keep absolute machine-specific paths and raw data out of commits.

## Main-experiment configuration

The default main experiment uses:

- 20 clients and full participation (`SAMPLING_NUMBER = 20`);
- Dirichlet concentrations 0.1, 1.0, and 10.0;
- a 784-32-10 neural network;
- L1 regularization with coefficient `1e-4`;
- five algorithm seeds and 1,000 communication rounds.

These values are defined in the settings section near the top of
`MNISTExperiment.py`.

## Data and path placeholders

The MNIST directory must contain the four uncompressed IDX files:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Before running, fill the intentionally blank path variables near the top of
both scripts:

```python
DATA_DIRECTORY = Path("")
```

For `MNISTExperiment.py`, also set the parent/project output location. For
`MNISTStepsizeTest.py`, set `PROJECT_DIRECTORY` to the directory in which the
tuning `Results/` folder should be created. Keep machine-specific absolute
paths out of commits.

## Running notes

Run with the repository root available on the Python import path. Fill the
blank data and output paths, then verify the sampling level, Dirichlet
concentrations, round budget, seeds, and stored algorithm parameters. Run the
step-size program only when the parameter selection must be reproduced.

MNIST uses completed-result caches rather than mid-run checkpoints. A result
is reused only when its settings match and its metrics contain the requested
rounds. Completed runs are skipped, but an interrupted unfinished run must be
started again.

## Outputs

The main experiment writes to its configured `RESULT_DIRECTORY`. The tuning
program writes to `PROJECT_DIRECTORY/Results/MNISTStepsizeTest/`. The archived
`ExperimentResults/` and `MNISTStepsizeTest/` directories are included as
selected reference outputs and do not replace the raw MNIST data.

## Dataset link

[MNIST dataset page](https://www.tensorflow.org/datasets/catalog/mnist)

<!-- Insert the official MNIST dataset URL in the empty parentheses above. -->
