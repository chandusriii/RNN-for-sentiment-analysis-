# Original Baseline Results

This file records only the result available from the original notebook, `RNN_for_sentimentanalysis.ipynb`.

## Experiment

| Item | Original implementation |
|---|---|
| Dataset | IMDb Movie Reviews |
| Rows after duplicate removal | 49,582 |
| Text representation | TF-IDF, maximum 5,000 features |
| Train/test split | 80% / 20% |
| Random state | 42 |
| Model | PyTorch `nn.RNN` |
| Hidden size | 128 |
| Training epochs | 10 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Recorded test accuracy | **85.16%** |

## Interpretation

The TF-IDF vectors are reshaped to `(batch_size, 1, 5000)`. The RNN therefore receives one timestep per review. This baseline should be described as a TF-IDF feature-vector classifier using an RNN layer, not as a word-by-word sequential language model.

## Metrics not recorded here

Precision, recall, F1-score, confusion matrix, per-epoch training history, and validation metrics are not reported in this document because they have not been verified from the saved experiment outputs. They can be generated in a reproducible evaluation pass later.

## Reproducibility note

The 85.16% figure is the accuracy recorded in the original notebook. Re-run the notebook in the intended environment before treating it as independently reproduced. Record any environment, preprocessing, or random-seed differences.
