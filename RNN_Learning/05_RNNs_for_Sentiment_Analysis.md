# Chapter 05 — RNNs for Sentiment Analysis

## 1. Sentiment-analysis task

Sentiment analysis is a natural-language-processing task that identifies the opinion or emotional polarity expressed in text.

For this project, the target is binary classification:

- 0 — Negative
- 1 — Positive

Example:

**Review:** “The movie was excellent and I enjoyed every moment.”

**Expected sentiment:** Positive

## 2. Why a many-to-one architecture fits

A movie review contains a sequence of words, but the task requires one sentiment label for the entire review.

This is a **many-to-one** pattern:

- Many input tokens are processed across timesteps.
- The model uses a sequence representation, often the final hidden state or a suitable pooled representation.
- A classification layer produces one output for the review.

## 3. Conceptual sequence-model workflow

A word-sequence RNN sentiment classifier commonly follows this workflow:

1. **Text preprocessing:** Normalize and clean review text.
2. **Tokenization:** Split each review into tokens.
3. **Vocabulary creation:** Map known tokens to integer IDs. Build the vocabulary using training data only to avoid leakage.
4. **Numerical encoding:** Convert each token into its vocabulary ID.
5. **Padding and truncation:** Make batches of sequences compatible in length.
6. **Embedding:** Map token IDs to dense vector representations.
7. **Recurrent layer:** Process the embedded sequence with an RNN or another recurrent cell.
8. **Classification layer:** Convert the sequence representation into a sentiment prediction.
9. **Evaluation:** Measure performance on held-out data.

This is a conceptual sequence-model workflow; it should not be confused with the exact feature pipeline in the original notebook.

## 4. Connecting concepts to the repository's original implementation

The original notebook, RNN_for_sentimentanalysis.ipynb, is preserved as a TF-IDF + PyTorch RNN experiment.

It converts each complete review into a 5,000-feature TF-IDF vector and adds a sequence dimension of length one before passing the vector to nn.RNN. Therefore, the original implementation is not a word-by-word sequence model.

The conceptual word-sequence workflow in this chapter explains how recurrent models can be used for sequential text processing. It does not rewrite or re-label the original implementation.

## 5. Evaluation metrics

For binary classification, common evaluation metrics include:

- **Accuracy:** Fraction of all predictions that are correct.
- **Precision:** Among predictions for a class, the fraction that are correct.
- **Recall:** Among actual examples of a class, the fraction correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of correct and incorrect predictions by actual and predicted class.

Metrics should be calculated from actual held-out predictions. Do not report a metric unless it has been measured.

## 6. Key takeaway

Sentiment analysis is naturally expressed as many-to-one sequence classification. RNNs can process review tokens in order and use recurrent state to build a representation for a final sentiment prediction.

### Quick revision

1. Why is sentiment classification a many-to-one task?
2. What is tokenization?
3. Why should the vocabulary be built from training data only?
4. Why are padding and truncation used?
5. What does an embedding layer do?
6. What do accuracy, precision, recall, and F1-score measure?
7. How does the original notebook's TF-IDF input differ from a word-by-word sequence input?
