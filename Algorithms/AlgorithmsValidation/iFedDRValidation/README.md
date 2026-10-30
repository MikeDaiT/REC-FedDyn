# iFedDR CIFAR-10 Validation Experiment

This directory contains an independent reproduction used to validate iFedDR
on a CIFAR-10 linear-probing problem. Images are converted to fixed features by
an ImageNet-pretrained ResNet18, and iFedDR optimizes only the convex linear
classifier over a Dirichlet client partition.

This is an implementation-validation experiment, not a formal `REC_FedDyn`
comparison.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `CIFAR10iFedDRLinearProbingExperiment.py` | **Validation experiment** | Extracts or loads frozen ResNet18 features, partitions the training data among clients, runs iFedDR, supports checkpoint continuation, and saves histories and diagnostic figures. |
| `Algorithms/` | Validation-specific implementation | Frozen copies of `Base_Optimizer.py`, `FMRI.py`, `iFedDR.py`, and `TorchUtils.py`. |
| `Results/` | Selected outputs | Archived validation checkpoints, histories, metadata, and plots. |

## Required directory structure

```text
repository root/
`-- Algorithms/
    `-- AlgorithmsValidation/
        `-- iFedDRValidation/
            |-- CIFAR10iFedDRLinearProbingExperiment.py
            |-- Algorithms/
            |   |-- Base_Optimizer.py
            |   |-- FMRI.py
            |   |-- iFedDR.py
            |   `-- TorchUtils.py
            |-- Data/
            |   `-- CIFAR10/
            |       `-- cifar-10-batches-py/
            |           |-- data_batch_1
            |           |-- data_batch_2
            |           |-- data_batch_3
            |           |-- data_batch_4
            |           |-- data_batch_5
            |           `-- test_batch
            `-- Results/
                `-- CIFAR10iFedDRLinearProbing/
```

Set the data path to the directory containing `cifar-10-batches-py`. Keep the
dataset and generated ResNet18 feature cache outside version control.

## Data and feature cache

For the full experiment, set `DEFAULT_DATA_DIRECTORY = Path("")` in the script
or pass `--data-dir`. The directory must contain:

```text
cifar-10-batches-py/data_batch_1
cifar-10-batches-py/data_batch_2
cifar-10-batches-py/data_batch_3
cifar-10-batches-py/data_batch_4
cifar-10-batches-py/data_batch_5
cifar-10-batches-py/test_batch
```

The default feature cache is
`cifar10_resnet18_imagenet1k_v1_features.pt` inside the data directory. Use
`--feature-cache` to place it elsewhere and `--rebuild-features` to recreate
it. Do not commit the dataset or the generated feature cache.

The full run requires torchvision and may need access to pretrained ResNet18
weights if they are not already cached by the local PyTorch installation.

## Running notes

Run with the validation-specific `Algorithms/` directory available for
imports. Configure the data, output, and feature-cache paths, then verify the
device, precision, round budget, checkpoint interval, feature batch size, and
data-loader workers.

The default checkpoint interval is five rounds. A compatible checkpoint
restores the current point, internal iFedDR state, history, and smoothness
values. Do not resume after changing the partition, precision, regularization,
or algorithm settings. The ResNet18 feature cache is separate and may be
reused when its data and feature settings remain compatible.

## Outputs

The default output directory is `Results/CIFAR10iFedDRLinearProbing/` below
this validation directory. It contains:

```text
ifeddr_checkpoint.pt
ifeddr_history.csv
client_label_distribution.csv
figure3_ifeddr_linear_probing.png
ifeddr_diagnostics.png
metadata.json
```

## Reference link

[Original iFedDR paper](https://openreview.net/forum?id=XFpb3T5Zc9)

<!-- Insert the public source used for this validation in the empty parentheses above. -->
