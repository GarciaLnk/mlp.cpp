# mlp.cpp

Implementation of a multi-layer perceptron (MLP) in pure C++ with no external dependencies.

This is a library developed for educational purposes, it aims to provide the basic building blocks to allow training and inference of a fully connected feed-forward artificial neural network with multiple layers. It is designed to be simple and easy to understand, and it is not intended to be used in production environments, it was hacked together in a few evenings so it is not optimized for performance and may contain bugs, although it has been tested with some examples and it seems to work fine.

## Features

- Fully connected feed-forward neural network
- Arbitrary number of layers and neurons
- Customizable activation functions
- Stochastic gradient descent with backpropagation
- Save and load trained networks
- Simple and easy to understand
- Built with C++20 and no external dependencies

## Compilation

The library uses `cmake` to generate the build files and `make` to automate the build process, so both of them have to be installed in the system. Any C++20-compatible compiler should work. To compile everything just run `make` in the root directory of the repository, this will build the library and the examples in the `build` directory.

## Usage

You can statically link the library and use it in your own project, the usage is very simple, you just need to create an instance of the `MLP` class with the desired learning rate, and add to it the layers you want to use, specifying the number of neurons in each layer, the activation function and its derivative. Some activation functions are predefined in `utils.h`.

```cpp
#include "mlp.h"
#include "utils.h"

// Neural network with one hidden layer
MLP mlp(0.1, true); // learning rate = 0.1, apply softmax to the output layer
mlp.addLayer(2, nullptr, nullptr); // 2 inputs
mlp.addLayer(4, frelu, freluDerivative); // 4 neurons, ReLU activation function
mlp.addLayer(1, fidentity, fidentityDerivative); // 1 output
```

Then you can train the network with the `train` method, passing the input and the expected output, and use the `predict` method to get the output of the network for a given input. The trained network can be saved to a file and loaded later with the `save` and `load` methods respectively, only the weights are saved, so you need to recreate the network with the same architecture before loading the weights.

```cpp
std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {1.0}, {0.0}};

// Train the network
mlp.train(inputs, targets, 1000); // 1000 epochs

// Test prediction
mlp.predict({0, 0}); // 0.0

// Save and load the trained network
mlp.save("network.bin");
mlp.load("network.bin");
```

## Examples

The `examples` directory contains an example of usage of the library on the Iris dataset. It contains a program that trains a neural network to classify the Iris flowers into the three different species, and another program that uses the trained network to predict the species of a flower given its measurements. The dataset is included in the repository, and the programs can be compiled and run with the following commands:

```sh
make iris_train
make iris_predict
```

Tests are also included in the repository, they can be compiled and run with `make test`.
