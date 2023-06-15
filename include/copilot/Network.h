#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "settings.h"
#include <vector>
#include <string>


namespace copilot {
    class Network{
        public:
            std::vector<std::vector<double>> weights;
            std::vector<std::vector<double>> biases;
            std::vector<double> weighted_sums;
            std::vector<double> layout;
            int layers;
            std::string activation;

            Network();
            Network(std::string f_weights);
            ~Network();

            std::vector<double> feed(std::vector<double> input);
            std::vector<double> forward_propagate(std::vector<double> input, int layer);

            std::vector<double> ReLU_vectorized(std::vector<double> x);
            std::vector<double> sigmoid_vectorized(std::vector<double> x);
            std::vector<double> softmax(std::vector<double> x);
            std::vector<double> tanh_vectorized(std::vector<double> x);

    };
};

#endif