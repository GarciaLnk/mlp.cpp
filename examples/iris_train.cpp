// Example of a neural network trained on the Iris dataset
// https://archive.ics.uci.edu/dataset/53/iris

#include "mlp.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int main() {
    const std::string filename = "iris.csv";
    std::ifstream file;
    std::string dir;
    try {
        file.open(dir + filename);
        if (!file.is_open()) {
            dir = "build/";
            file.open(dir + filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not find file: " + filename);
            }
        }
    } catch (const std::runtime_error &e) {
        std::cerr << "Exception caught: " << e.what() << '\n';
        return 1;
    }

    const std::unordered_map<std::string, double> conversionRules = {
        {"Iris-setosa", 0.0}, {"Iris-versicolor", 1.0}, {"Iris-virginica", 2.0}};
    std::vector<std::vector<double>> dataset = parseCSV(file, 0, {}, conversionRules);

    // Shuffle the dataset in a deterministic way for reproducibility in the validation step
    std::mt19937 gen(42);
    std::ranges::shuffle(dataset, gen);

    // Split the dataset into training and validation sets
    auto trainingSize = static_cast<long>(0.8 * static_cast<double>(dataset.size()));
    std::vector<std::vector<double>> trainingData(dataset.begin(), dataset.begin() + trainingSize);
    std::vector<std::vector<double>> validationData(dataset.begin() + trainingSize, dataset.end());

    // Prepare the training inputs and targets
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;
    for (const auto &row : trainingData) {
        trainingInputs.emplace_back(row.begin(), row.begin() + 4); // First 4 elements are features
        trainingTargets.push_back(oneHotEncode(row.back(), 3));    // Last element is the label
    }

    MLP mlp(0.00001, true); // use softmax
    mlp.addLayer(4, frelu, freluDerivative);
    mlp.addLayer(10, frelu, freluDerivative);
    mlp.addLayer(10, frelu, freluDerivative);
    mlp.addLayer(3, fidentity, fidentityDerivative);

    mlp.train(trainingInputs, trainingTargets, 10000);

    // Validate the network
    std::vector<std::vector<int>> confusionMatrix(3, std::vector<int>(3, 0));
    int correctPredictions = 0;
    for (const auto &row : dataset) {
        auto input = std::vector<double>(row.begin(), row.begin() + 4);
        std::vector<double> target = oneHotEncode(row.back(), 3);
        std::vector<double> output = mlp.predict(input);

        // Compare the highest output value's index with the target's index
        if (std::distance(output.begin(), std::ranges::max_element(output)) ==
            std::distance(target.begin(), std::ranges::max_element(target))) {
            correctPredictions++;
        }

        // Update the confusion matrix
        auto targetIndex = static_cast<int>(std::distance(target.begin(), std::ranges::max_element(target)));
        auto outputIndex = static_cast<int>(std::distance(output.begin(), std::ranges::max_element(output)));
        confusionMatrix[targetIndex][outputIndex]++;
    }

    // Print the confusion matrix
    std::cout << "Confusion matrix:\n";
    for (const auto &row : confusionMatrix) {
        for (int cell : row) {
            std::cout << cell << ' ';
        }
        std::cout << '\n';
    }

    double accuracy = static_cast<double>(correctPredictions) / static_cast<double>(dataset.size());
    std::cout << "Accuracy: " << accuracy * 100 << "%" << '\n';

    // Save the trained model
    mlp.save("iris_model.bin");

    return 0;
}
