# Chapter 01 — Introduction to Recurrent Neural Networks

## 1. What is an RNN?

A **Recurrent Neural Network (RNN)** is a neural-network architecture designed to process sequential data. It maintains a hidden state that carries information from earlier timesteps into the processing of later inputs.

A conventional feedforward network typically treats input features as a fixed collection. An RNN is designed for situations where the order of inputs matters.

### Common applications

- Sentiment analysis
- Text generation and language modeling
- Language translation
- Speech recognition
- Time-series forecasting

## 2. Understanding sequential data

Sequential data is data whose order carries meaning.

| Data type | Example sequence |
|---|---|
| Text | Words in a sentence |
| Speech | Audio frames over time |
| Finance | Stock prices over successive dates |
| Weather | Temperature readings over time |
| Video | Frames in chronological order |

Consider:

- “The movie is good.”
- “The movie is not good.”

The word **not** changes the meaning. A model processing text as a sequence can use word order and context when learning such patterns.

## 3. Why use an RNN?

An RNN processes one element at a time and updates its hidden state after each element. The hidden state provides a compact representation of information from the sequence seen so far.

For “I love machine learning,” the tokens can be processed in order: I → love → machine → learning.

At each step, the network uses the current token representation and the previous hidden state.

## 4. Key terminology

- **Timestep:** One position in a sequence.
- **Input (x_t):** The input presented at timestep t.
- **Hidden state (h_t):** The recurrent representation passed from one timestep to the next.
- **Output (y_t):** The output produced at a timestep or from the final sequence representation.
- **Sequence length:** Number of timesteps in an input sequence.

## 5. Key takeaway

RNNs are designed for ordered data. Their defining idea is recurrence: information is carried forward through a hidden state as the sequence is processed.

### Quick revision

1. What makes data sequential?
2. Why does word order matter in sentiment analysis?
3. What information does an RNN hidden state carry?
4. Name three applications of RNNs.
