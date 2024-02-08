#include "neuron.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

Neuron::Neuron(size_t numInputs, const bool constantWeightInit) : constantWeightInit(constantWeightInit) {
    inputs.resize(numInputs, 0.0);
    initializeWeights(numInputs);
}

double Neuron::getOutput() const noexcept { return output; }

double Neuron::getGradient() const noexcept { return gradient; }

const std::vector<double> &Neuron::getWeights() const noexcept { return weights; }

const std::vector<double> &Neuron::getInputs() const noexcept { return inputs; }

// Set the inputs for the neuron and reinitialize the weights if needed
void Neuron::setInputs(std::vector<double> newInputs, bool initWeights) {
    if (initWeights) {
        initializeWeights(newInputs.size());
    } else {
        if (newInputs.size() != inputs.size()) {
            throw std::invalid_argument("Mismatch in number of inputs");
        }
    }
    inputs = std::move(newInputs);
}

void Neuron::setWeights(std::vector<double> newWeights) {
    if (newWeights.size() != weights.size()) {
        throw std::invalid_argument("Mismatch in number of weights");
    }
    weights = std::move(newWeights);
}

void Neuron::setOutput(double newOutput) { output = newOutput; }

void Neuron::setGradient(double newGradient) {
    gradient = std::clamp(newGradient, -10.0, 10.0); // Gradient clipping
}

// Initialize the weights using He initialization
void Neuron::initializeWeights(size_t numInputs) {
    if (numInputs == 0) {
        return;
    }
    weights.resize(numInputs);
    double variance = 2.0 / static_cast<double>(numInputs);
    double stddev = std::sqrt(variance);
    static unsigned seed = constantWeightInit ? 42 : std::random_device{}();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution dis(0.0, stddev);
    std::ranges::generate(weights, [&]() { return dis(gen); });
}

// Calculate the pre-output of the neuron by taking the dot product of the inputs and weights and adding the bias
double Neuron::calculatePreOutput() { return std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias); }

// Save the number of weights and the weights themselves to the output stream
void Neuron::save(std::ofstream &out) const {
    std::size_t numWeights = weights.size();
    out.write(reinterpret_cast<const char *>(&numWeights), sizeof(numWeights));
    out.write(reinterpret_cast<const char *>(weights.data()), sizeof(double) * numWeights);
}

// Read in the number of weights and the weights themselves from the input stream
void Neuron::load(std::ifstream &in) {
    std::size_t numWeights;
    in.read(reinterpret_cast<char *>(&numWeights), sizeof(numWeights));
    weights.resize(numWeights);
    in.read(reinterpret_cast<char *>(weights.data()), sizeof(double) * numWeights);
}