#include "neuron.h"
#include "utils.h"
#include <cassert>
#include <exception>
#include <iostream>

void testOutputCalculation();

int main() {
    try {
        testOutputCalculation();

        std::cout << "All neuron tests passed successfully.\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Test failed: " << ex.what() << '\n';
        return 1;
    }
}

void testOutputCalculation() {
    Neuron neuron(3);
    neuron.setInputs({1.0, 2.0, 3.0});
    neuron.setWeights({0.5, 0.5, 0.5});
    neuron.setOutput(neuron.calculatePreOutput()); // no activation function (identity)

    double expectedOutput = 4.0; // 0.5*1 + 0.5*2 + 0.5*3 + 1.0 (bias)
    assert(approxEqual(neuron.getOutput(), expectedOutput));
}
