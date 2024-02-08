#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <cstddef>
#include <fstream>
#include <functional>
#include <vector>

class Layer {
  public:
    explicit Layer(size_t size, size_t inputsPerNeuron, std::function<double(double)> activationFunc,
                   std::function<double(double)> derivActivationFunc, bool normalize = false,
                   bool constantWeightInit = false);

    [[nodiscard]] std::vector<Neuron> &getNeurons() noexcept;
    std::vector<double> getOutputs() const;
    double getActivationResult(double output) const;
    double getDerivActivationResult(double output) const;

    void setAllWeights(const std::vector<std::vector<double>> &newWeights);
    void setInputsForAllNeurons(const std::vector<double> &inputs);
    void setOutputs(const std::vector<double> &outputs);

    void connectLayer(Layer &previousLayer);
    void calculateOutputs();
    void applySoftmax();

    void save(std::ofstream &out) const;
    void load(std::ifstream &in);

  private:
    bool normalize{false};
    std::vector<Neuron> neurons{};
    std::function<double(double)> activationFunction{nullptr};
    std::function<double(double)> derivActivationFunction{nullptr};
};

#endif // LAYER_H
