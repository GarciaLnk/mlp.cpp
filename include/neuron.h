#ifndef NEURON_H
#define NEURON_H

#include <cstddef>
#include <fstream>
#include <functional>
#include <vector>

class Neuron {
  public:
    Neuron() = default;
    explicit Neuron(size_t numInputs, const bool constantWeightInit = false);

    double getOutput() const noexcept;
    double getGradient() const noexcept;
    const std::vector<double> &getWeights() const noexcept;
    const std::vector<double> &getInputs() const noexcept;

    void setInputs(std::vector<double> newInputs, bool initWeights = false);
    void setWeights(std::vector<double> newWeights);
    void setOutput(double newOutput);
    void setGradient(double newGradient);

    void initializeWeights(size_t numInputs);
    double calculatePreOutput();

    void save(std::ofstream &out) const;
    void load(std::ifstream &in);

  private:
    double output{0.0};
    double gradient{0.0};
    double bias{1.0};
    bool constantWeightInit{false};
    std::vector<double> inputs{};
    std::vector<double> weights{};
};

#endif // NEURON_H
