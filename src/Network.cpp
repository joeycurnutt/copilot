#include "settings.hpp"
using namespace std;

class Network {
public:
    vector<double> weights;
    vector<double> biases;
    vector<double> weighted_sums;
    vector<double> neurons;
    int layers;
    this->layers = Settings.layers;
    string activation;
    this->activation = Settings.activation;

    Network(string file = '', vector<double> weights, vector<double> biases) {
        if(file != ''){
            this->load_from_file(file);
        }
        else{
            this->weights = weights;
            this->biases = biases;
        }
    }


    void load_from_file(string file_name){

    }

}