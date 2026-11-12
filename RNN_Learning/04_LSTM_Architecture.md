# Chapter 04 — Long Short-Term Memory (LSTM) Architecture

## 1. What is LSTM?

**Long Short-Term Memory (LSTM)** is a gated recurrent-neural-network architecture designed to learn dependencies over longer sequences more effectively than a basic vanilla RNN.

An LSTM maintains two state vectors:

- **Hidden state (h_t):** The representation passed to the next timestep and often used to produce outputs.
- **Cell state (C_t):** The memory pathway that carries information through the sequence.

LSTM uses gates to regulate information flow.

## 2. The gates and candidate memory

An LSTM cell uses three gates—forget, input, and output—and a candidate cell-state update. The gate activations commonly use sigmoid, which produces values between 0 and 1.

### A. Forget gate

Decides how much of the previous cell state to retain.

f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)

A value near 0 reduces the corresponding information; a value near 1 retains it.

### B. Input gate

Controls how much new candidate information is written to the cell state.

i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)

### C. Candidate cell content

Creates candidate information that may be added to the cell state.

C_tilde_t = tanh(W_C [h_(t-1), x_t] + b_C)

### D. Cell-state update

Combines retained old information with selected candidate information.

C_t = f_t * C_(t-1) + i_t * C_tilde_t

### E. Output gate and hidden state

The output gate controls what information is exposed through the hidden state.

o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)

h_t = o_t * tanh(C_t)

Here, * denotes element-wise multiplication.

## 3. Activation functions

### Sigmoid

sigmoid(x) = 1 / (1 + exp(-x))

- Range: 0 to 1.
- Used by the forget, input, and output gates.
- Controls the amount of information passed through each gate.

### Tanh

- Range: -1 to +1.
- Used to create candidate cell content and transform the cell state before producing the hidden state.

## 4. Why the cell state matters

The cell state provides a pathway for information to travel through the sequence, with the forget and input gates controlling updates. This gated design helps LSTMs preserve useful information and learn long-term dependencies more effectively than a basic RNN in many tasks.

LSTM does not guarantee that vanishing gradients will never occur; it is designed to mitigate the difficulty.

## 5. RNN vs. LSTM

| Feature | Vanilla RNN | LSTM |
|---|---|---|
| State | Hidden state | Hidden state and cell state |
| Gates | No gating mechanism | Forget, input, and output gates |
| Long-range dependencies | Can be difficult to learn | Designed to handle them more effectively |
| Complexity | Simpler | More parameters and computation |
| Gradient behavior | More susceptible to vanishing gradients over long sequences | Gated cell-state path helps mitigate the issue |

## 6. Key takeaway

LSTM extends recurrence with a cell state and gates. These mechanisms regulate what to forget, what new information to store, and what information to expose as the hidden state.

### Quick revision

1. What are the two states maintained by an LSTM?
2. What does the forget gate control?
3. What is the role of the input gate?
4. What does the candidate cell content represent?
5. How is the cell state updated?
6. What does the output gate control?
7. Which activation functions are commonly used in LSTM gates?
