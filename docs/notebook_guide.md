# Notebook Guide

## Original project notebook

`RNN_for_sentimentanalysis.ipynb` is the original notebook and the primary implementation in this repository.

It contains the original TF-IDF + PyTorch RNN workflow. Its input to the recurrent layer has one timestep per review; it should not be described as a word-by-word sequential RNN.

## How to run

1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Start Jupyter from the repository root.
4. Open `RNN_for_sentimentanalysis.ipynb`.
5. Run cells from top to bottom.
6. Keep `IMDB Dataset.csv` in the repository root.

## Keep learning notes separate

The `RNN_Learning/` folder contains conceptual Markdown chapters. These are study notes, not additional model implementations.

## Keep results separate

The `results/` folder contains documentation of verified results. Any new experiment should have its own clearly named notebook or script and a separate results file. Do not overwrite the original notebook or mix results from different datasets, splits, or architectures.
