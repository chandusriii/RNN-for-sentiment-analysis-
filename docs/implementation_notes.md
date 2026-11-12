# Implementation Notes

## Baseline currently in the repository

The original notebook is `RNN_for_sentimentanalysis.ipynb`. Keep it intact as the record of the first experiment.

- Dataset: 50,000 IMDb reviews before duplicate removal.
- Rows remaining after duplicate removal: 49,582.
- Text representation: TF-IDF with up to 5,000 features.
- Split: 80% train / 20% test, `random_state=42`.
- Model: PyTorch `nn.RNN`, hidden size 128, one recurrent layer, one linear output.
- Training: 10 epochs, Adam optimizer, binary cross-entropy loss.
- Recorded test accuracy: 85.16%.

## Architecture interpretation

The TF-IDF feature matrix has shape `(number_of_reviews, 5000)`. The notebook adds a singleton sequence dimension, giving batches shaped `(batch_size, 1, 5000)`. There is only one recurrent timestep. The RNN processes a single feature vector per review; it does not process a sequence of word embeddings.

## Preprocessing issue to address in the extension

The original stopword-removal implementation uses repeated string replacement. This can remove character sequences inside other words, not just standalone stopword tokens. In the new notebook, tokenize first and filter complete tokens against the stopword set. Keep the original notebook unchanged so the baseline remains traceable.

## Sequence-model extension

Build the next experiment as a separate notebook. Use a train-only vocabulary to avoid leaking information from validation/test text. Convert reviews to token IDs, pad/truncate them, embed the IDs, and feed the resulting multi-timestep sequence into a vanilla RNN.

Do not report results until the notebook has actually been run. Record the split strategy, sequence length, vocabulary size, model configuration, training settings, and evaluation metrics so comparisons are meaningful.

## Comparison plan

Evaluate both approaches on the same held-out test examples where practical. Report accuracy, precision, recall, F1-score, and a confusion matrix. Explain that the baseline uses TF-IDF at one timestep, while the sequence model uses token embeddings across multiple timesteps.
