// Network settings
// To prevent double-declaration of settings
#ifndef _SETTINGS_H_
#define _SETTINGS_H_

#include <string>
#include <vector>

// Network settings
class Settings {
    public:
        int layers; // number of layers [2,inf)
        std::vector<double> layout; // format as [in_nodes,hidden_1 nodes,...,output_nodes]
        std::string activation; // relu, sigmoid, tanh
        double alpha; // learning rate (0,1)
        double decay; // decay rate (0,1)
        int batches;

        Settings();
        ~Settings();
};

#endif