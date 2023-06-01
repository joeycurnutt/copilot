#include "settings.h"
#include "Network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace copilot;

    Network::Network() {
        std::vector<std::vector<std::vector<double>>> weights = {};
        std::vector<std::vector<double>> biases = {};
        std::vector<std::vector<double>> weighted_sums = {};
        std::vector<std::vector<double>> neurons = {};
        int layers = Settings().layers;
        std::string activation = Settings().activation;
    }

    Network::~Network(){

    }


    void Network::load_from_file(std::string f_weights) {
        int i = 0;
        std::ifstream file(f_weights);
        std::string line;

        while (std::getline(file, line)) {
            std::vector<double> data;
            std::stringstream ss(line);
            std::string item;

            while (std::getline(ss, item, ',')) {
                data.push_back(std::stod(item));
            }

            if (i % 2 == 0) {
                // weights
                std::vector<std::vector<double>> new_weight;
                std::vector<double> new_row;

                for (double num : data) {
                    new_row.push_back(num);
                    if (new_row.size() >= Settings().layout[i / 2]) {
                        new_weight.push_back(new_row);
                        new_row.clear();
                    }
                }

                // check if rows and columns match up
                if (new_weight.size() != Settings().layout[i / 2 + 1] || new_weight[0].size() != Settings().layout[i / 2]) {
                    std::cout << "the weights at line " << i << " do not match up with settings" << std::endl;
                    return;
                }

                weights.push_back(new_weight);
            } else {
                // biases
                std::vector<double> new_biases;

                for (double num : data) {
                    new_biases.push_back(num);
                }

                // check if vector length match up
                if (new_biases.size() != Settings().layout[i / 2 + 1]) {
                    std::cout << "the biases at line " << i << " do not match up with settings" << std::endl;
                    return;
                }

                biases.push_back(new_biases);
            }

            i++;
        }

        file.close();
    }

    void Network::save(std::string outfile, const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<double>>& biases) {
        std::ofstream file(outfile);

        if (!file.is_open()) {
            std::cout << "Failed to open the file." << std::endl;
            return;
        }

        for (int i = 0; i < weights.size(); i++) {
            std::stringstream ss;

            for (const auto& row : weights[i]) {
                for (double num : row) {
                    ss << num << ", ";
                }
                ss << "\n";
            }

            file << ss.str();

            for (double bias : biases[i]) {
                file << bias << ", ";
            }
            file << "\n";
        }

        file.close();
    }

    std::vector<double> Network::feed(std::vector<double> input) {
        weighted_sums.clear();
        neurons.clear();
        return forward_propagate(input, 0);
    }

    std::vector<double> Network::forward_propagate(std::vector<double> input, int layer) {
            this->neurons.push_back(input);
            if (layer >= this->layers - 1) {
                return input;
            }
            std::vector<double> weighted_sums;
            for (int i = 0; i < this->weights[layer].size(); i++) {
                double neuron_weighted_sum = 0;
                for (int j = 0; j < this->weights[layer][i].size(); j++) {
                    neuron_weighted_sum += this->weights[layer][i][j] * input[j];
                }
                neuron_weighted_sum += this->biases[layer][i];
                weighted_sums.push_back(neuron_weighted_sum);
            }
            this->weighted_sums.push_back(weighted_sums);
            std::vector<double> next_layer;
            if (layer >= this->layers - 2) {
                next_layer = softmax(weighted_sums);
            } else if (this->activation == "relu") {
                next_layer = ReLU_vectorized(weighted_sums);
            } else if (this->activation == "sigmoid") {
                next_layer = sigmoid_vectorized(weighted_sums);
            } else if (this->activation == "tanh") {
                next_layer = tanh_vectorized(weighted_sums);
            }
            return forward_propagate(next_layer, layer + 1);
    }
    std::vector<double> ReLU_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(x[i] * (x[i] > 0));
        }
        return result;
    }
    std::vector<double> dReLU_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(1. * (x[i] > 0));
        }
        return result;
    }
    std::vector<double> sigmoid_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(1 / (1 + exp(-x[i])));
        }
        return result;
    }
    std::vector<double> dsigmoid_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back((1/(1+exp(-(1*(x[i]))))) * (1 - (1/(1+exp(-(1*(x[i])))))));
        }
        return result;
    }
    std::vector<double> softmax(std::vector<double> x) {
        std::vector<double> result;
        double sum = 0;
        for (int i = 0; i < x.size(); i++) {
            sum += exp(x[i]);
        }
        for (int i = 0; i < x.size(); i++) {
            result.push_back(exp(x[i]) / sum);
        }
        return result;
    }
    std::vector<double> tanh_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(tanh(x[i]));
        }
        return result;
    }
    std::vector<double> dtanh_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(1 - pow(tanh(x[i]), 2));
        }
        return result;
    }