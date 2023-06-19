#include "copilot/settings.h"
#include "Network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace copilot;

    Network::Network() {
        
    }

    Network::Network(std::string f_weights) : Network(){
        Settings settings;
        
        layout = settings.layout;
        layers = settings.layers;
        activation = settings.activation;
        classification = settings.classification;
        last_layer = settings.last_layer;

        int i = 0;
        std::ifstream file(f_weights);
        
        if (file.is_open()){
            while (file) {
                while (i < (layers * 2) - 2){
                    std::string line;
                    std::vector<double> data;
                    while (std::getline(file, line)) {
                        std::stringstream ss(line);
                        std::string item;

                        while (std::getline(ss, item, ',')) {
                            data.push_back(std::stod(item)); // Convert each item to integer and add it to the row vector
                        }
                        if (i % 2 == 0) {
                            // weights
                            weights.push_back(data);

                            // check if rows and columns match up
                            if (weights[i / 2].size() != layout[i / 2 + 1] * layout[i / 2]) {
                                std::cout << "the weights at line " << i + 1 << " do not match up with settings" << std::endl;
                            }
                        } else {
                            // biases
                            biases.push_back(data);

                            // check if vector length match up
                            if (biases[(i-1)/2].size() != layout[(i-1)/2 + 1]) {
                                std::cout << "the biases at line " << i + 1 << " do not match up with settings" << std::endl;
                            }
                        }
                        data.clear();
                        i++;
                    }          
                }
                file.close();
            }
        } else {
            std::cout << "Cannot open file" << std::endl;
        }
    }

    Network::~Network(){

    }

    std::vector<double> Network::feed(std::vector<double> input) {
        weighted_sums.clear();
        return forward_propagate(input, 0);
    }

    std::vector<double> Network::forward_propagate(std::vector<double> input, int layer) {
        // a(1) = activation(W*a(0)+b)
        // 3x1 = activation(3x5 * 5x1 + 3x1)
            if (layer >= layers - 1) {
                std::vector<double> result_vector;
                auto num = std::minmax_element(input.begin(), input.end());
		        int result = std::distance(input.begin(), num.second);
                result_vector.push_back(result);
                return result_vector;
            }
            std::vector<double> layerBiases = biases[layer];  // Create a copy of biases

            double weighted_sum = std::inner_product(std::begin(weights[layer]), std::end(weights[layer]), std::begin(input), 0.0);

            std::transform(layerBiases.begin(), layerBiases.end(), layerBiases.begin(),
               [weighted_sum](double bias) { return bias + weighted_sum; });
            
            weighted_sums = layerBiases;

            std::cout << "Layer " << layer + 1 << " " << "weighted sum: ";
            for (int a = 0; a < weighted_sums.size(); a++){
                std::cout << weighted_sums[a] << " ";
            }
            std::cout << "\n\n";
            std::vector<double> next_layer;
            if (layer >= layers - 2) {
                if(last_layer == "softmax"){
                    next_layer = softmax(weighted_sums);
                } else if (last_layer == "sigmoid"){
                    next_layer = sigmoid_vectorized(weighted_sums);
                } else if (last_layer == "linear"){
                    next_layer = linear_vectorized(weighted_sums);
                }
                std::cout << "Last layer: ";
                for (int a = 0; a < next_layer.size(); a++){
                    std::cout << next_layer[a] << " ";
                }
            } else if (activation == "relu") {
                next_layer = ReLU_vectorized(weighted_sums);
            } else if (activation == "sigmoid") {
                next_layer = sigmoid_vectorized(weighted_sums);
            } else if (activation == "tanh") {
                next_layer = tanh_vectorized(weighted_sums);
            }
            
            return forward_propagate(next_layer, layer + 1);
    }

    std::vector<double> Network::ReLU_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back((x[i] > 0) ? x[i]: 0);
        }
        return result;
    }

    std::vector<double> Network::sigmoid_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(1 / (1 + exp(-x[i])));
        }
        return result;
    }

    std::vector<double> Network::softmax(std::vector<double> x) {
        std::vector<double> result(x.size());
        
        // Find the maximum value in the input vector
        double max_value = *std::max_element(x.begin(), x.end());
        
        // Subtract the maximum value from each element
        std::transform(x.begin(), x.end(), result.begin(), [max_value](double c) {
            return std::exp(c - max_value);
        });
        
        // Calculate the sum of the exponentiated values
        double sum = std::accumulate(result.begin(), result.end(), 0.0);
        
        // Divide each element by the sum to get the softmax probabilities
        std::transform(result.begin(), result.end(), result.begin(), [sum](double c) {
            return c / sum;
        });
        
        return result;
    }

    std::vector<double> Network::tanh_vectorized(std::vector<double> x) {
        std::vector<double> result;
        for (int i = 0; i < x.size(); i++) {
            result.push_back(tanh(x[i]));
        }
        return result;
    }

    std::vector<double> Network::linear_vectorized(std::vector<double> x) {
        return x;
    }