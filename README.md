# RNN for Sentiment Analysis

An educational deep learning project that explores **Natural Language Processing (NLP)** and **Recurrent Neural Networks (RNNs)** for classifying IMDb movie reviews as **positive** or **negative**.

The project starts with a TF-IDF-based baseline implemented with PyTorch. The next development stage is to build a genuinely sequential RNN using token IDs, padding, and an embedding layer, then compare it with an LSTM.

## Project Overview

Sentiment analysis is a text-classification task that identifies the opinion expressed in a piece of text. In this project, movie reviews from the IMDb dataset are processed and used to train a binary classifier.

**Objective:** Given a movie review, predict whether its sentiment is positive or negative.

## Dataset

- **Dataset:** IMDb Movie Reviews
- **Size:** 50,000 reviews (before duplicate removal)
- **Classes:** Positive and negative
- **Input:** A written movie review
- **Target:** Sentiment label

The dataset is not included in this repository. Download the IMDb Dataset and place the CSV file in the project directory with the filename:

`IMDB Dataset.csv`

The notebook expects the CSV to contain two columns: `review` and `sentiment`.

## Current Workflow

The current notebook, `RNN_for_sentimentanalysis.ipynb`, covers the following steps:

1. Load the IMDb dataset using Pandas.
2. Inspect missing values and remove duplicate reviews.
3. Convert review text to lowercase.
4. Remove URLs, punctuation, and HTML tags.
5. Tokenize text and remove English stopwords using NLTK.
6. Apply Porter stemming.
7. Encode sentiment labels.
8. Convert review text into TF-IDF features (up to 5,000 features).
9. Split the data into training and testing sets.
10. Create PyTorch datasets and data loaders.
11. Define and train a PyTorch RNN model.
12. Evaluate the model using test accuracy.

## Important Implementation Note

The current notebook is a **TF-IDF-based baseline**. It converts each review into a fixed-size feature vector and adds a singleton sequence dimension before passing it to the RNN. Therefore, the model does not process the review as a word-by-word sequence, and it does not learn the temporal relationships between successive words in the way a sequence-based RNN does.

This baseline is being retained as a learning milestone. A proper sequence-based implementation is planned next.

## Planned Improvements

- Build a vocabulary and convert each review into token IDs.
- Pad or truncate reviews to a consistent sequence length.
- Add an embedding layer to learn dense word representations.
- Implement a sequence-based vanilla RNN for sentiment classification.
- Implement an LSTM model to learn longer-term dependencies.
- Compare model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Add example predictions for new, unseen movie reviews.

## Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **NLP:** NLTK
- **Feature Extraction:** Scikit-learn TF-IDF
- **Development:** Jupyter Notebook

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chandusriii/RNN-for-sentiment-analysis-.git
cd RNN-for-sentiment-analysis-
```

### 2. Create and activate a virtual environment (optional)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy nltk scikit-learn torch jupyter
```

### 4. Download the IMDb dataset

Place `IMDB Dataset.csv` in the repository root.

### 5. Download required NLTK resources

Run this once in Python:

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 6. Launch the notebook

```bash
jupyter notebook
```

Open `RNN_for_sentimentanalysis.ipynb` and run the cells in order.

## Repository Structure

```text
RNN-for-sentiment-analysis-/
│
├── RNN_for_sentimentanalysis.ipynb
├── IMDB Dataset.csv          # Download separately; not committed
├── README.md
├── LICENSE
└── .gitignore
```

## Learning Goals

Through this project, I am strengthening my understanding of:

- NLP text preprocessing and tokenization
- Text representation and TF-IDF
- Recurrent neural network architecture
- Hidden states and sequential learning
- Model training, loss calculation, and backpropagation
- Sentiment classification and model evaluation
- Differences between vanilla RNNs and LSTMs

## Project Status

**In progress.** The initial TF-IDF-based PyTorch baseline is documented. Sequence-based RNN and LSTM implementations, detailed evaluation, and model comparisons are planned development milestones.

## Author

**Chandu Sri**

- GitHub: [@chandusriii](https://github.com/chandusriii)
- Portfolio: [chandusriii.github.io](https://chandusriii.github.io/chandusriii/)

---

If you find this project useful for learning NLP and deep learning, feel free to explore the repository and follow its progress.
