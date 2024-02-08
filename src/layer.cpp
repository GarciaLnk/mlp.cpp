#include "layer.h"
#include "neuron.h"
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

Layer::Layer(size_t size, size_t inputsPerNeuron, std::function<double(double)> activationFunc,
             std::function<double(double)> derivActivationFunc, const bool normalize, const bool constantWeightInit)
    : normalize(normalize), activationFunction(std::move(activationFunc)),
      derivActivationFunction(std::move(derivActivationFunc)) {
    neurons.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        neurons.emplace_back(inputsPerNeuron, constantWeightInit);
    }
}

std::vector<Neuron> &Layer::getNeurons() noexcept { return neurons; }

std::vector<double> Layer::getOutputs() const {
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
    for (const auto &neuron : neurons) {
        outputs.push_back(neuron.getOutput());
    }
    return outputs;
}

double Layer::getActivationResult(double output) const { return activationFunction(output); }

double Layer::getDerivActivationResult(double output) const { return derivActivationFunction(output); }

// Set the weights for all neurons in the layer
void Layer::setAllWeights(const std::vector<std::vector<double>> &newWeights) {
    if (newWeights.size() != neurons.size()) {
        throw std::invalid_argument(
            std::format("Mismatch in number of neurons and number of weight vectors, expected {}, got {}",
                        neurons.size(), newWeights.size()));
    }
    for (size_t i = 0; i < newWeights.size(); ++i) {
        if (newWeights[i].size() != neurons[i].getWeights().size()) {
            throw std::invalid_argument(std::format("Mismatch in number of weights for neuron {}, expected {}, got {}",
                                                    i, neurons[i].getWeights().size(), newWeights[i].size()));
        }
        neurons[i].setWeights(newWeights[i]);
    }
}

void Layer::setInputsForAllNeurons(const std::vector<double> &inputs) {
    for (auto &neuron : neurons) {
        neuron.setInputs(inputs);
    }
}

void Layer::setOutputs(const std::vector<double> &outputs) {
    size_t numNeurons = neurons.size();
    if (outputs.size() != numNeurons) {
        throw std::invalid_argument(
            std::format("Mismatch in number of outputs provided, expected {}, got {}", numNeurons, outputs.size()));
    }
    for (size_t i = 0; i < numNeurons; ++i) {
        neurons[i].setOutput(outputs[i]);
    }
}

// Connect the layer to the previous layer by setting the inputs for each neuron and initializing the weights
void Layer::connectLayer(Layer &previousLayer) {
    size_t numInputsPerNeuron = previousLayer.getNeurons().size();

    for (auto &neuron : neurons) {
        std::vector<double> inputs;
        inputs.reserve(numInputsPerNeuron);
        for (const auto &prevNeuron : previousLayer.getNeurons()) {
            inputs.push_back(prevNeuron.getOutput());
        }
        neuron.setInputs(inputs, true);
    }
}

void Layer::calculateOutputs() {
    if (!normalize) {
        for (auto &neuron : neurons) {
            double preOutput = neuron.calculatePreOutput();
            neuron.setOutput(activationFunction(preOutput));
        }
    } else {
        double sum = 0.0;
        double sq_sum = 0.0;
        double epsilon = 1e-5;

        // Calculate pre-outputs for normalization
        for (auto &neuron : neurons) {
            double preOutput = neuron.calculatePreOutput();
            sum += preOutput;
            sq_sum += preOutput * preOutput;
        }

        double mean = sum / static_cast<double>(neurons.size());
        double variance = sq_sum / static_cast<double>(neurons.size()) - mean * mean;
        double stddev = std::sqrt(variance + epsilon);

        // Normalize and apply activation
        for (auto &neuron : neurons) {
            double normalizedOutput = (neuron.getOutput() - mean) / stddev;
            neuron.setOutput(activationFunction(normalizedOutput));
        }
    }
}

void Layer::applySoftmax() {
    double sumOfExponentials = 0.0;
    for (const auto &neuron : neurons) {
        sumOfExponentials += std::exp(neuron.getOutput());
    }

    for (auto &neuron : neurons) {
        double output = neuron.getOutput();
        double softmaxOutput = std::exp(output) / sumOfExponentials;
        neuron.setOutput(softmaxOutput);
    }
}

void Layer::save(std::ofstream &out) const {
    std::size_t numNeurons = neurons.size();
    out.write(reinterpret_cast<const char *>(&numNeurons), sizeof(numNeurons));
    for (const auto &neuron : neurons) {
        neuron.save(out);
    }
}

void Layer::load(std::ifstream &in) {
    std::size_t numNeurons;
    in.read(reinterpret_cast<char *>(&numNeurons), sizeof(numNeurons));
    neurons.resize(numNeurons);
    for (auto &neuron : neurons) {
        neuron.load(in);
    }
}
