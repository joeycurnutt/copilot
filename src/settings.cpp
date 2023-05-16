// Network settings
#include <string>
#include <vector>

class Settings {
    public:
        Settings(){
            int layers = 4; // number of layers [2,inf)
            std::vector<double> layout(4); // format as [in_nodes,hidden_1 nodes,...,output_nodes]
            std::string activation; // relu, sigmoid, tanh
            double alpha = 0.5; // learning rate (0,1)
            double decay = 0.3; // decay rate (0,1)
            int batches = 200;
        }
};