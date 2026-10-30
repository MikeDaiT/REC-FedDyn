# FedDR FEMNIST Validation Experiment

This directory contains an independent reproduction used to validate the
FedDR implementation on the 26-output, 200-writer FEMNIST composite
optimization problem. It is an implementation-validation experiment and is
not one of the four formal `REC_FedDyn` comparison experiments described in
the repository root.

## Files and directories

| Item | Role | Description |
| --- | --- | --- |
| `FEMNISTFedDRCompositeExperiment.py` | **Validation and parameter sweep** | Runs FedDR with exact full-gradient local updates over one or more learning rates, saves resumable checkpoints and CSV histories, and creates the learning-rate comparison figure. |
| `Algorithms/` | Validation-specific implementation | Frozen copies of `Base_Optimizer.py`, `FedDR.py`, `ProximalOperator.py`, and `TorchUtils.py` used by this reproduction. |
| `Results/` | Selected outputs | Archived validation checkpoints, histories, and figures. |

There is no separate main-experiment script and tuning script in this folder.
The learning-rate sweep is built into
`FEMNISTFedDRCompositeExperiment.py` through `--learning-rates`.

## Required directory structure

Keep the frozen validation algorithms beside the experiment and place the
26-output FEMNIST files in the local data directory:

```text
repository root/
`-- Algorithms/
    `-- AlgorithmsValidation/
        `-- FedDRValidation/
            |-- FEMNISTFedDRCompositeExperiment.py
            |-- Algorithms/
            |   |-- Base_Optimizer.py
            |   |-- FedDR.py
            |   |-- ProximalOperator.py
            |   `-- TorchUtils.py
            |-- FEMNIST/
            |   |-- mytrain.json
            |   `-- mytest.json
            `-- Results/
                `-- FEMNISTFedDRComposite/
```

LEAF-style `FEMNIST/train/*.json` and `FEMNIST/test/*.json` directories may be
used instead of the two combined JSON files. Keep raw data and newly generated
results outside version control.

## Data

Use the writer-partitioned FEMNIST data released for the 26-output experiment.
Supported layouts are:

```text
FEMNIST/mytrain.json
FEMNIST/mytest.json
```

or LEAF-style shard directories:

```text
FEMNIST/train/*.json
FEMNIST/test/*.json
```

Pass the containing directory with `--data-directory`. This validation data
layout differs from the 62-class FEMNIST OGL main experiment. Do not commit the raw dataset.

## Running notes

Run with this validation directory's `Algorithms/` package taking precedence
over the repository-wide package. Verify the data and result directories,
learning-rate list, round budget, gradient batch size, device, and restart
setting.

Each learning-rate candidate has a separate full-state checkpoint saved every
`CHECKPOINT_ROUNDS = 10` rounds. Matching checkpoints resume at the last saved
boundary; restart mode discards that continuation. Do not reuse a checkpoint
when its data or experiment settings have changed.

## Outputs

By default, outputs are saved below this validation directory in
`Results/FEMNISTFedDRComposite/`. Each learning rate has its own checkpoint and
CSV history. `feddr_femnist_l1_lr_sweep.png` compares the candidates.

## Reference link

[Original FedDR paper](https://proceedings.neurips.cc/paper/2021/hash/fe7ee8fc1959cc7214fa21c4840dff0a-Abstract.html)

<!-- Insert the public source used for this validation in the empty parentheses above. -->
