# Dataset Guide

## File

The project uses the IMDb movie-review dataset stored as:

`IMDB Dataset.csv`

The file contains 50,000 labeled reviews and two columns:

- `review`: the movie-review text
- `sentiment`: the sentiment label (`positive` or `negative`)

## Location

Keep the CSV in the repository root, next to `RNN_for_sentimentanalysis.ipynb`.

The original notebook refers to the file by its name. Moving it into a subfolder or renaming it would require changing the notebook's data-loading path. To preserve the original implementation, this repository keeps the dataset in its existing location.

## Data handling

- Treat the supplied CSV as the source dataset.
- Do not overwrite it with a cleaned or sampled copy.
- If a future experiment needs a sample or processed data, save that as a separate file under an appropriately named folder and document how it was generated.
- Do not commit temporary caches, model checkpoints, or generated artifacts unless they are intentionally part of a documented release.

## Reproducibility note

The original notebook removes duplicate records during preprocessing, leaving 49,582 rows according to its recorded workflow. This does not modify the source CSV on disk.
