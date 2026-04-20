# MLP from Scratch

A general-purpose feedforward neural network implemented in pure Python with no ML libraries.
Backpropagation and gradient descent are implemented manually without autograd.

## Overview

Implements a fully connected neural network that supports:
- Arbitrary layer sizes and depth
- ReLU activation on hidden layers, linear output
- MSE loss
- Stochastic gradient descent

Trained and tested on XOR — a classic non-linearly separable problem that requires
at least one hidden layer to solve.

## Usage

```python
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]
layer_sizes = [2, 4, 1]

[weights, biases] = train(inputs, targets, layer_sizes, epochs=1000, learning_rate=0.1, print_every=100)
```

## Implementation Details

- `forward_prop` — computes activations layer by layer, applying ReLU to all hidden layers
- `back_prop` — computes gradients via chain rule, propagating from output to input
- Weights initialized with uniform random values in [-1, 1]
- Biases initialized to 0

## Notes

Occasionally converges to local minima due to random initialization and the non-convex
loss landscape of XOR. Re-running with a different random seed typically resolves this.