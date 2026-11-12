# Chapter 03 — Training RNNs

## 1. Forward propagation

Forward propagation passes input data through a network to generate predictions.

For a sequence-based RNN:

1. The first input is processed with an initial hidden state.
2. The network calculates a new hidden state.
3. The next input is processed together with that hidden state.
4. This repeats for the remaining timesteps.
5. The model produces an output, such as a sentiment prediction.

## 2. Backpropagation Through Time (BPTT)

**Backpropagation Through Time (BPTT)** is the standard training approach for a basic RNN. It applies backpropagation to the recurrent computation across the sequence.

### Training process

1. Process the input sequence from the first timestep to the last.
2. Calculate the model's prediction.
3. Compare the prediction with the target using a loss function.
4. Unroll the recurrent computation across timesteps for gradient calculation.
5. Propagate gradients backward through the unrolled sequence.
6. Use an optimizer to update the shared parameters.

For example, if a positive review is predicted as negative, the loss measures the prediction error. BPTT calculates how the model's parameters contributed to that error.

## 3. Vanishing gradient problem

During BPTT, gradients pass backward through many recurrent steps. Repeated multiplications can cause gradients to become extremely small.

When this happens, earlier timesteps receive very weak learning signals. The model may struggle to learn dependencies between words that are far apart in a long sequence.

Possible effects:

- Difficulty learning long-term dependencies.
- Earlier sequence information may have little influence on learning.
- Training becomes challenging for long sequences.

## 4. Exploding gradient problem

The opposite problem can also occur: gradients can become excessively large. This can cause unstable parameter updates and make training difficult.

### Gradient clipping

Gradient clipping limits the magnitude or norm of gradients before the optimizer updates model parameters. It is a common way to reduce the impact of exploding gradients.

## 5. Why LSTM was introduced

Traditional RNNs can struggle to preserve useful information over long sequences. LSTM introduces a cell state and gates that regulate what information is retained, added, and exposed.

The next chapter explains this architecture.

## 6. Key takeaway

BPTT trains recurrent networks by propagating gradients through time. Vanishing and exploding gradients are important challenges when training recurrent models on long sequences.

### Quick revision

1. What does BPTT stand for?
2. Why is an RNN unrolled during training?
3. What causes vanishing gradients?
4. What can exploding gradients do to training?
5. How does gradient clipping help?
