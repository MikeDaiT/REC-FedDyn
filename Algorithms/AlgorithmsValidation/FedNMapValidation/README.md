# FedNMap MNIST Validation Experiment

This directory contains a deterministic shallow-network reproduction used to
validate the FedNMap implementation. MNIST examples are sorted by label and split across clients to create a controlled heterogeneous problem with
elastic-net regularization.

This is an implementation-validation experiment, not a formal `REC_FedDyn`
comparison.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `FedNMapShallowReproduction.py` | **Validation experiment** | Runs the configured client-count and local-update combinations, saves one NPZ file per configuration, and creates a combined CSV file and stationarity plot. |
| `Algorithms/` | Validation-specific implementation | Frozen copies of `Base_Optimizer.py`, `FedNMap.py`, and `TorchUtils.py`. |
| `Results/` | Selected outputs | Archived validation arrays, summary tables, metadata, and figures. |

The default run evaluates client counts 20, 50, and 100 with 10 and 20 local
updates for 400 communication rounds.

## Required directory structure

```text
repository root/
`-- Algorithms/
    `-- AlgorithmsValidation/
        `-- FedNMapValidation/
            |-- FedNMapShallowReproduction.py
            |-- Algorithms/
            |   |-- Base_Optimizer.py
            |   |-- FedNMap.py
            |   `-- TorchUtils.py
            |-- Data/
            |   `-- MNIST/
            |       |-- train-images-idx3-ubyte
            |       `-- train-labels-idx1-ubyte
            `-- Results/
                `-- FedNMapShallowReproduction/
```

The data and output locations may be changed through the program settings.
Keep the raw MNIST files and newly generated results outside version control.

## Data

Pass a directory containing the uncompressed MNIST training files:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
```

The reproduction only needs the training split because it evaluates the
optimization quantities used in the validation. Do not commit the MNIST data.

## Running notes

Run with the validation-specific `Algorithms/` directory available for
imports. Verify the data and output directories, client counts, local updates,
round budget, hidden units, `gamma`, `nu1`, `nu2`, and L2 convention.

Each NPZ file is a completed-result cache, not a mid-run checkpoint. Resume
mode skips a configuration only when its saved history has the requested
length. An interrupted unfinished configuration must be rerun, while completed
configurations remain reusable.

## Outputs

The default output directory is `Results/FedNMapShallowReproduction/` below
this validation directory. Important files include:

```text
fednmap_n{clients}_q{local_updates}.npz
fednmap_shallow_deterministic.csv
fednmap_shallow_stationarity.png
metadata.json
```

## Reference link

[Original FedNMap paper](https://arxiv.org/abs/2602.03357)

<!-- Insert the public source used for this validation in the empty parentheses above. -->
