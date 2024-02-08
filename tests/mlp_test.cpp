#include "mlp.h"
#include "utils.h"
#include <cassert>
#include <cstdlib>
#include <exception>
#include <ext/string_conversions.h>
#include <iostream>
#include <string>
#include <vector>

void testFeedForward();
void testBackPropagate();
void testTrainingAndPrediction();
void testSaveAndLoad();

int main() {
    try {
        testTrainingAndPrediction();
        testFeedForward();
        testBackPropagate();
        testSaveAndLoad();

        std::cout << "All MLP tests passed successfully.\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Test failed: " << ex.what() << '\n';
        return 1;
    }
}

void testFeedForward() {
    MLP mlp({2, 3, 2}, 0.1, fidentity, fidentityDerivative);
    // Structure the weights for each layer based on the network architecture
    std::vector<std::vector<std::vector<double>>> weightsForLayers{
        {{}, {}},                             // Layer 0 weights: 2 neurons, each with 0 weights
        {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}, // Layer 1 weights: 3 neurons, each with 2 weights
        {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}}    // Layer 2 weights: 2 neurons, each with 3 weights
    };

    // Set weights for each layer
    mlp.setWeightsAllLayers(weightsForLayers);

    // Define inputs and feed them forward through the network
    std::vector<double> inputs{1.0, 1.0};
    mlp.feedForward(inputs);
    std::vector<double> outputs = mlp.getResult();

    // With identity activation and bias = 1.0, expect each output to be 4.0
    double expectedOutput = 4.0;
    assert(outputs.size() == 2);
    for (double output : outputs) {
        assert(approxEqual(output, expectedOutput));
    }
}

void testBackPropagate() {
    MLP mlp(0.001);
    mlp.addLayer(2, fidentity, fidentityDerivative);
    mlp.addLayer(3, fidentity, fidentityDerivative);
    mlp.addLayer(1, fidentity, fidentityDerivative);

    std::vector<std::vector<std::vector<double>>> weightsForLayers{
        {{}, {}},                             // Layer 0 weights
        {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}, // Layer 1 weights
        {{0.5, 0.5, 0.5}}                     // Layer 2 weights
    };

    mlp.setWeightsAllLayers(weightsForLayers);

    std::vector<double> inputs{1.0, 1.0};
    mlp.feedForward(inputs);

    std::vector<double> target{1.0}; // Target for backpropagation
    mlp.backPropagate(target);

    // Verifying that the network's output has changed after backpropagation
    std::vector<double> newOutputs = mlp.getResult();
    assert(newOutputs.size() == 1);
    assert(std::abs(newOutputs[0] - 2.25) > 1e-5); // Expecting a change in the output
}

void testSaveAndLoad() {
    MLP mlp1({2, 3, 1}, 0.0001, fidentity, fidentityDerivative);
    std::string filename = "test_mlp_model.bin";

    // Train and save the model
    mlp1.train({{0, 0}, {1, 1}}, {{0}, {1}}, 100);
    mlp1.save(filename);

    // Load the model into a new instance and test prediction matches
    MLP mlp2({2, 3, 1}, 0.0001, fidentity, fidentityDerivative);
    mlp2.load(filename);

    assert(mlp1.predict({0, 0})[0] == mlp2.predict({0, 0})[0]);
    assert(mlp1.predict({1, 1})[0] == mlp2.predict({1, 1})[0]);

    std::remove(filename.c_str()); // Delete the model file
}

void testTrainingAndPrediction() {
    // Create a simple network with 2 hiddens layer and deterministic weights
    MLP mlp(0.1);
    mlp.addLayer(2, frelu, freluDerivative, false, true);
    mlp.addLayer(4, frelu, freluDerivative, false, true);
    mlp.addLayer(4, frelu, freluDerivative, false, true);
    mlp.addLayer(1, fsigmoid, fsigmoidDerivative, false, true);

    // XOR problem inputs and targets
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {1.0}, {0.0}};

    mlp.train(inputs, targets, 10000); // Train for 10000 epochs

    // Test prediction accuracy
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto prediction = mlp.predict(inputs[i]);
        std::cout << "Prediction: " << prediction[0] << ", Target: " << targets[i][0] << '\n';
        assert(std::abs(prediction[0] - targets[i][0]) < 0.5); // Assert predictions are close to targets
    }
}