#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>
#include <string>


namespace copilot {
    class Network{
        public:
            std::vector<std::vector<std::vector<double>>> weights;
            std::vector<std::vector<double>> biases;
            std::vector<std::vector<double>> weighted_sums;
            std::vector<std::vector<double>> neurons;
            int layers;
            std::string activation;

            Network(std::string f_weights, std::vector<std::vector<std::vector<double>>> weights, std::vector<std::vector<double>> biases);
            ~Network();

            void load_from_file(std::string f_weights);
            void save(std::string outfile, const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<double>>& biases);
            std::vector<double> feed(std::vector<double> input);
            std::vector<double> Network::forward_propagate(std::vector<double> input, int layer);

    };
};

#endif