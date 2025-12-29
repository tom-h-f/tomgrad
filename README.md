# TomGrad

A simple autograd and neural network library written in C. Building this to learn how automatic differentiation and neural networks work under the hood.

## Current Status

Working on basic tensor operations and computational graph setup. Got element-wise multiplication with autograd working, along with some basic scalar operations.

## TODO

### Misc
- Should the backprop function be per-value or per-tensor?

### High Priority

**Autograd Operations**
- Add matrix multiplication backward pass
- Add sum reduction operation with backward pass
- Add mean reduction operation with backward pass

**Matrix Operations**
- Implement matrix multiplication (this is crucial for neural nets)
- Implement tensor transpose
- Add reshape operation
- Add broadcasting support for operations on different shapes

**Activation Functions**
- Add ReLU with backward pass
- Add Sigmoid with backward pass
- Add Tanh with backward pass
- Add Softmax with stable numerical implementation
- Add backward pass for Softmax

**Loss Functions**
- Implement MSE loss with gradient
- Implement cross-entropy loss
- Implement binary cross-entropy loss

### Medium Priority

**Neural Network Layers**
- Build a dense/linear layer struct
- Add weight and bias parameter management
- Create a way to compose layers together
- Add layer abstraction API

**Optimizers**
- Implement SGD optimizer
- Implement Adam optimizer
- Add learning rate scheduling

**Training Infrastructure**
- Add utility to zero out gradients
- Add parameter update step
- Build a simple training loop example

### Low Priority

**Utilities**
- Add parameter serialization (save/load weights)
- Add model checkpointing
- Add more tensor initialization options (Xavier, He initialization)
- Add better error messages

**Documentation**
- Add examples directory with working neural net examples
- Document the computational graph internals
- Add inline comments for complex operations

