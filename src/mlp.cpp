#include "mlp.h"
#include "layer.h"
#include "neuron.h"
#include <cstddef>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

class EmptyNetwork : public std::logic_error {
    using std::logic_error::logic_error;
};

class ModelIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

MLP::MLP(const std::vector<size_t> &layersNodes, double lr, const std::function<double(double)> &activationFunc,
         const std::function<double(double)> &derivActivationFunc, const bool softmax, const bool constantWeightInit)
    : learningRate(lr), softmax(softmax) {
    if (layersNodes.size() < 2) {
        throw std::invalid_argument("Network must have at least two layers (input and output).");
    }

    for (std::size_t i = 0; i < layersNodes.size(); ++i) {
        // inputsPerNeuron is the number of neurons in the previous layer, 0 for the first layer
        std::size_t inputsPerNeuron = (i == 0 ? 0 : layersNodes[i - 1]);
        layers.emplace_back(layersNodes[i], inputsPerNeuron, activationFunc, derivActivationFunc, false,
                            constantWeightInit);
    }

    for (std::size_t i = 1; i < layers.size(); ++i) {
        layers[i].connectLayer(layers[i - 1]);
    }
}

MLP::MLP(double lr) : learningRate(lr) {}

MLP::MLP(double lr, const bool softmax) : learningRate(lr), softmax(softmax) {}

std::vector<double> MLP::getResult() const {
    if (layers.empty()) {
        throw EmptyNetwork("No layers in the network.");
    }
    return layers.back().getOutputs();
}

std::vector<Layer> &MLP::getLayers() noexcept { return layers; }

void MLP::setWeightsAllLayers(const std::vector<std::vector<std::vector<double>>> &newWeights) {
    if (newWeights.size() != layers.size()) {
        throw std::invalid_argument(
            std::format("Mismatch in number of layers and number of weight vectors, expected {}, got {}", layers.size(),
                        newWeights.size()));
    }
    for (size_t i = 0; i < newWeights.size(); ++i) {
        layers[i].setAllWeights(newWeights[i]);
    }
}

void MLP::addLayer(size_t numNodes, const std::function<double(double)> &activationFunc,
                   const std::function<double(double)> &derivActivationFunc, const bool normalize,
                   const bool constantWeightInit) {
    std::size_t inputsPerNeuron = layers.empty() ? 0 : layers.back().getNeurons().size();
    layers.emplace_back(numNodes, inputsPerNeuron, activationFunc, derivActivationFunc, normalize, constantWeightInit);
    if (layers.size() > 1) {
        layers.back().connectLayer(layers[layers.size() - 2]);
    }
}

void MLP::feedForward(const std::vector<double> &inputValues) {
    if (layers.empty()) {
        throw EmptyNetwork("No layers in the network.");
    }

    // Directly set the outputs of the input layer.
    layers.front().setOutputs(inputValues);

    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i].setInputsForAllNeurons(layers[i - 1].getOutputs());
        layers[i].calculateOutputs();
    }

    if (softmax) {
        // Apply softmax to the output layer
        layers.back().applySoftmax();
    }
}

void MLP::backPropagate(const std::vector<double> &targetValues) {
    if (layers.empty()) {
        throw EmptyNetwork("No layers in the network.");
    }

    // Calculate output layer gradients
    Layer &outputLayer = layers.back();
    for (size_t i = 0; i < outputLayer.getNeurons().size(); ++i) {
        double output = outputLayer.getNeurons()[i].getOutput();
        double gradient = output - targetValues[i];
        outputLayer.getNeurons()[i].setGradient(gradient);
    }

    // Calculate gradients on hidden layers
    for (int layerNum = static_cast<int>(layers.size()) - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];
        for (auto &neuron : hiddenLayer.getNeurons()) {
            double sum = 0.0;
            for (size_t n = 0; n < nextLayer.getNeurons().size(); ++n) {
                sum += neuron.getWeights()[n] * nextLayer.getNeurons()[n].getGradient();
            }
            double gradient = sum * hiddenLayer.getDerivActivationResult(neuron.getOutput());
            neuron.setGradient(gradient);
        }
    }

    // Update weights
    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];
        for (Neuron &neuron : layer.getNeurons()) {
            std::vector<double> newWeights = neuron.getWeights();
            for (size_t w = 0; w < newWeights.size(); ++w) {
                newWeights[w] -= learningRate * neuron.getGradient() * prevLayer.getNeurons()[w].getOutput();
            }
            neuron.setWeights(newWeights);
        }
    }
}

void MLP::train(const std::vector<std::vector<double>> &inputData, const std::vector<std::vector<double>> &targetData,
                std::size_t epochs) {
    if (inputData.size() != targetData.size()) {
        throw std::invalid_argument("Input data and target data must have the same number of entries.");
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t i = 0; i < inputData.size(); ++i) {
            feedForward(inputData[i]);
            backPropagate(targetData[i]);
        }
    }
}

std::vector<double> MLP::predict(const std::vector<double> &input) {
    feedForward(input);
    return getResult();
}

void MLP::save(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw ModelIOError("Unable to open file for saving: " + filename);
    }

    // Serialize each layer
    for (const auto &layer : getLayers()) {
        layer.save(file);
    }
}

void MLP::load(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw ModelIOError("Unable to open file for loading: " + filename);
    }

    // Deserialize each layer
    for (auto &layer : getLayers()) {
        layer.load(file);
    }
}