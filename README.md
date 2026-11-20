# NeuralNetwork

Before anything, this is a project as a CS student.
and a **WIP**

It became a  way to understand how to build a practical and
user-friendly neural network framework with no external dependencies. It is also an opportunity to
explore how neural networks work internally and how their components can
be designed and implemented in C.

The priorities are a coherent architecture, a clear and approachable
API, and reliable numerical behavior. Further performance optimizations
will follow once the existing components and pipelines have been
thoroughly validated.


A personal project to develop a usable, user-friendly neural network
framework in C11.

The goal is to understand deep-learning and making a modular framework for defining, training, and
evaluating neural networks, while keeping the underlying mechanisms
understandable and accessible. The project focuses on a clear API,
reusable components, and control over the implementation of core
operations such as memory management, forward propagation,
backpropagation, and parameter updates.

## Features

### General architecture

Networks are built by assembling layers with a `network\_builder`. The
builder prepares the layers and transfers ownership to the network when
it is initialized.

Layers share a common interface and provide implementation-specific
operations, including:

* forward propagation;
* backpropagation;
* resource destruction.

### Layers and components

\---

Component               Purpose                 Status

\---

Dense                   Fully connected layer   Implemented and used

Flatten                 Flattens input data     Implemented and used

Reservoir               Layer for Reservoir     Under development /
Computing experiments   requires further
validation

Convolution             Foundation for future   To be developed and
CNN support             validated
---

The project currently includes the following activation functions:

* `RELU`
* `LRELU` (Leaky ReLU)
* `TANH`
* `LINEAR`
* `SOFTPLUS`

In the current API, an activation function can be associated directly
with a Dense layer. A more explicit separation between affine
transformations and activation layers may be considered in a future API
revision.

### Optimizers

The project includes multiple parameter-update algorithms:

* `SGD` --- Stochastic Gradient Descent
* `ADAMW` --- AdamW

Optimizers are kept separate from layer implementations to make them
easier to maintain and extend.

### Data loading and training

The `DataLoader` module can load datasets such as MNIST in IDX format
and prepare inputs and labels for training and evaluation.

The training pipeline handles forward propagation, backpropagation, and
parameter updates. The project also provides prediction and evaluation
functions.


## Project structure

``` text
.
├── data/                   # Datasets and test files
├── include/                # Public headers
└── src/
    ├── core/               # Network, builder, and training
    ├── optimizer/          # Optimizers
    ├── activationFunction/  # Activation functions
    ├── Dataloader/         # Data loading and preparation
    └── layer/              # Layer implementations
```

File names and the exact directory structure may evolve as the project
develops.

## Building

The project uses CMake and the **C11** standard.

Example configuration with GCC or Clang:

``` bash
cmake -S . -B build -DCMAKE\_BUILD\_TYPE=Release
cmake --build build --config Release
```

The current CMake configuration also enables `-O3` and `-march=native`
for compatible compilers. `-march=native` generates code optimized for
the machine used to compile the project, so the resulting binary may not
run on older or different processors.

Dataset files are copied to the output directory specified by the CMake
configuration. If a data file cannot be found at runtime, check the
program's working directory.

## Example: building an MLP

main.c in /core shows some examples of building networks 

The builder API makes it possible to describe a network by adding layers
sequentially. The following example illustrates an MLP for flattened
MNIST images. Adapt the function names and arguments to the exact API
available in the current project version.

``` c
network\_builder\_t builder = builder\_init(28 \* 28);

builder\_add\_dense(builder, 128, LRELU);
builder\_add\_dense(builder, 10, LRELU);

Network\_t network = init\_network(builder, ADAMW);
```

Available optimizer identifiers:

* `SGD`
* `ADAMW`

Available activation identifiers:

* `RELU`
* `LRELU`
* `TANH`
* `LINEAR`
* `SOFTPLUS`

Use the project's creation and destruction functions according to their
memory-ownership contracts.

## Testing and validation

Before adding further features, the following areas should be tested
systematically:

1. Creation and destruction of every layer type.
2. Input/output dimensions and values produced by forward propagation.
3. Backpropagation gradients, ideally checked using finite differences.
4. SGD and AdamW updates.
5. DataLoader dimensions, labels, and data transformations.
6. Predictions and evaluation on an independent test set.
7. Allocation-error handling and memory-leak detection.

Experimental results should be interpreted in the context of random
initialization, data normalization, batch size, and training
hyperparameters.

## Planned improvements

The next development steps include:

* stabilizing and testing the existing pipelines;
* validating the Reservoir Computing layer using synthetic time
series, such as Mackey--Glass or Lorenz data;
* introducing a tensor representation suitable for multidimensional
data;
* developing and validating convolutional layers and related
operations;
* investigating a GPU computing backend at a later stage.



