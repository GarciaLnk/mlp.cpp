#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <unordered_map>

bool approxEqual(double a, double b, double epsilon = 1e-5);
double fsigmoid(double x);
double fsigmoidDerivative(double x);
double ftanh(double x);
double ftanhDerivative(double x);
double frelu(double x);
double freluDerivative(double x);
double fidentity(double x);
double fidentityDerivative(double x);
std::vector<double> oneHotEncode(double value, int categories);
std::vector<std::vector<double>> parseCSV(std::ifstream &file, int skipHeaderLines, const std::vector<int> &skipColumns,
                                          const std::unordered_map<std::string, double> &conversionRules);

#endif // UTILS_H
