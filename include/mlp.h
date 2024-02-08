#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

class MLP {
  public:
    MLP(const std::vector<size_t> &layersNodes, double lr, const std::function<double(double)> &activationFunc,
        const std::function<double(double)> &derivActivationFunc, const bool softmax = false,
        const bool constantWeightInit = false);

    explicit MLP(double lr);
    explicit MLP(double lr, const bool softmax);

    [[nodiscard]] std::vector<double> getResult() const;
    std::vector<Layer> &getLayers() noexcept;

    void setWeightsAllLayers(const std::vector<std::vector<std::vector<double>>> &newWeights);

    void addLayer(size_t numNodes, const std::function<double(double)> &activationFunc,
                  const std::function<double(double)> &derivActivationFunc, const bool normalize = false,
                  const bool constantWeightInit = false);
    void feedForward(const std::vector<double> &inputValues);
    void backPropagate(const std::vector<double> &targetValues);

    void train(const std::vector<std::vector<double>> &inputData, const std::vector<std::vector<double>> &targetData,
               std::size_t epochs);

    std::vector<double> predict(const std::vector<double> &input);

    void save(const std::string &filename);
    void load(const std::string &filename);

  private:
    double learningRate{0.01};
    std::vector<Layer> layers{};
    bool softmax{false};
};

#endif // MLP_H
