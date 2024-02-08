// Example of inference with the trained model from examples/iris_train.cpp

#include "mlp.h"
#include "utils.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

int main() {
    const std::unordered_map<std::string, double> conversionRules = {
        {"Iris-setosa", 0.0}, {"Iris-versicolor", 1.0}, {"Iris-virginica", 2.0}};

    // Create the same network structure as in the training example
    MLP mlp(0.00001, true);
    mlp.addLayer(4, frelu, freluDerivative);
    mlp.addLayer(10, frelu, freluDerivative);
    mlp.addLayer(10, frelu, freluDerivative);
    mlp.addLayer(3, fidentity, fidentityDerivative);

    mlp.load("iris_model.bin");

    // Make predictions based on the user input until the user decides to quit
    std::string userInput;
    while (true) {
        std::cout << "Enter the 4 features of the iris flower (sepal length, sepal width, petal length, petal width), "
                     "separated by spaces, or 'q' to quit:\n";
        std::getline(std::cin, userInput);

        if (userInput == "q") {
            break;
        }

        std::istringstream iss(userInput);
        std::vector<double> input;
        double value;
        while (iss >> value) {
            input.push_back(value);
        }

        if (input.size() != 4) {
            std::cout << "Invalid input. Please enter 4 numbers.\n";
            continue;
        }

        std::vector<double> output = mlp.predict(input);

        // Print the class name converting the max value index to the class name
        auto maxIndex = static_cast<int>(std::distance(output.begin(), std::ranges::max_element(output)));
        for (const auto &pair : conversionRules) {
            if (pair.second == maxIndex) {
                std::cout << "Predicted class: " << pair.first << std::fixed << std::setprecision(2) << " ("
                          << output[maxIndex] * 100 << "%) \n";
                break;
            }
        }

        std::cout << '\n';
    }

    return 0;
}