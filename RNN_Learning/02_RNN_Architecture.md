# Chapter 02 — RNN Architecture

## 1. Main components

An RNN uses three central components:

1. **Input (x_t):** The current input at timestep t.
2. **Hidden state (h_t):** Information carried from previous timesteps and updated using the current input.
3. **Output (y_t):** A prediction or representation produced by the network.

The hidden state is the recurrent memory of the network.

## 2. How an RNN processes a sequence

Consider “I love machine learning.” The model receives token representations one timestep at a time. At each timestep it:

1. Receives the current input.
2. Combines it with the previous hidden state.
3. Applies an activation function to calculate the new hidden state.
4. Optionally produces an output.
5. Passes the new hidden state to the next timestep.

## 3. Mathematical representation

A common vanilla RNN formulation is:

**Hidden state**

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

**Output**

y_t = W_hy h_t + b_y

Where:

- x_t — current input
- h_t — current hidden state
- h_(t-1) — previous hidden state
- W_xh — input-to-hidden weight matrix
- W_hh — hidden-to-hidden recurrent weight matrix
- W_hy — hidden-to-output weight matrix
- b_h and b_y — bias terms
- tanh — activation function in this common formulation

The exact output activation depends on the task. For binary classification, a sigmoid may be applied to a final logit.

## 4. Weight sharing

A defining feature of RNNs is that the same recurrent weights are reused at every timestep.

For a four-token sentence, the network does not create a new recurrent weight matrix for every word. It applies the same learned transformations repeatedly.

Benefits:

- Fewer parameters than using unrelated transformations at every timestep.
- The same learned transition can process sequences of different lengths.
- The network can learn patterns that occur at different positions.

## 5. Common input-output patterns

| Pattern | Meaning | Example |
|---|---|---|
| One-to-one | One input, one output | Basic image classification |
| One-to-many | One input, a sequence of outputs | Image caption generation |
| Many-to-one | A sequence of inputs, one output | Sentiment classification |
| Many-to-many | A sequence of inputs, a sequence of outputs | Sequence labeling or translation |

These are task patterns, not different recurrent-cell equations.

## 6. Many-to-one sentiment classification

For sentiment analysis, the model can read a review as a sequence and use a final sequence representation to predict one label.

Example input: “The movie was excellent.”

Example output: **Positive**

## 7. Key takeaway

At every timestep, an RNN updates its hidden state using the current input and previous hidden state. Shared recurrent weights allow the same cell to be applied across the sequence.

### Quick revision

1. What are the input, hidden state, and output?
2. Write the hidden-state equation.
3. What is weight sharing?
4. Which input-output pattern is commonly used for sentiment classification?
