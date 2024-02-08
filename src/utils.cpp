#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Utility function for approximate comparison of floating-point numbers
bool approxEqual(double a, double b, double epsilon) { return std::abs(a - b) < epsilon; }

// Activation functions and their derivatives
double fsigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double fsigmoidDerivative(double x) {
    double output = fsigmoid(x);
    return output * (1.0 - output);
}

double ftanh(double x) { return std::tanh(x); }

double ftanhDerivative(double x) {
    double output = ftanh(x);
    return 1.0 - output * output;
}

double frelu(double x) { return std::max(0.0, x); }

double freluDerivative(double x) { return x > 0.0 ? 1.0 : 0.0; }

double fidentity(double x) { return x; }

double fidentityDerivative(double /*x*/) { return 1.0; }

// Utility function for one-hot encoding
std::vector<double> oneHotEncode(double value, int categories) {
    std::vector<double> encoded(categories, 0.0);
    encoded[static_cast<int>(value)] = 1.0;
    return encoded;
}

// Utility function to parse a CSV file into a vector
std::vector<std::vector<double>> parseCSV(std::ifstream &file, int skipHeaderLines, const std::vector<int> &skipColumns,
                                          const std::unordered_map<std::string, double> &conversionRules) {
    std::vector<std::vector<double>> data;
    std::string line;

    // Skip header lines if any
    for (int i = 0; i < skipHeaderLines; ++i) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream lineStream(line);
        std::string cell;
        int columnIndex = 0;
        int effectiveColumnIndex = 0;

        while (std::getline(lineStream, cell, ',')) {
            // Skip columns if needed
            if (std::find(skipColumns.begin(), skipColumns.end(), columnIndex) != skipColumns.end()) {
                columnIndex++;
                continue;
            }

            // Convert string to double, applying conversion rules for specific strings
            auto it = conversionRules.find(cell);
            if (it != conversionRules.end()) {
                row.push_back(it->second);
            } else {
                row.push_back(std::stod(cell));
            }

            columnIndex++;
            effectiveColumnIndex++;
        }

        if (!row.empty()) {
            data.push_back(std::move(row));
        }
    }

    return data;
}