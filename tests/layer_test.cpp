#include "layer.h"
#include "utils.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

void testWeightSetting();
void testOutputCalculation();
void testLayerConnection();

int main() {
    try {
        testWeightSetting();
        testOutputCalculation();
        testLayerConnection();

        std::cout << "All layer tests passed successfully.\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Test failed: " << ex.what() << '\n';
        return 1;
    }
}

void testWeightSetting() {
    Layer layer(2, 3, fidentity, fidentityDerivative);
    std::vector<std::vector<double>> newWeights{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    layer.setAllWeights(newWeights);

    const auto &neurons = layer.getNeurons();
    for (size_t i = 0; i < neurons.size(); ++i) {
        const auto &weights = neurons[i].getWeights();
        assert(weights == newWeights[i]);
    }
}

void testOutputCalculation() {
    Layer layer(2, 3, fidentity, fidentityDerivative);
    std::vector<double> inputs{1.0, 1.0, 1.0};
    layer.setInputsForAllNeurons(inputs);

    // Set weights to 1.0
    std::vector<std::vector<double>> newWeights{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    layer.setAllWeights(newWeights);

    layer.calculateOutputs();
    std::vector<double> outputs = layer.getOutputs();

    // With identity activation, each output should be the sum of inputs, plus the bias
    double expectedOutput = 4.0;
    for (auto output : outputs) {
        assert(approxEqual(output, expectedOutput));
    }
}

void testLayerConnection() {
    Layer layer1(3, 0, fidentity, fidentityDerivative);
    Layer layer2(2, 1, fidentity, fidentityDerivative);
    layer1.setOutputs({1.0, 1.0, 1.0});

    // Connect layer2 to layer1
    layer2.connectLayer(layer1);

    const auto &neurons = layer2.getNeurons();
    for (const auto &neuron : neurons) {
        const auto &inputs = neuron.getInputs();
        assert((inputs == std::vector<double>{1.0, 1.0, 1.0}));
    }
}
