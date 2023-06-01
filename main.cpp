//Usage example
#include "cppsrc/Network.h"
#include "cppsrc/Settings.h"
#include <bits/stdc++.h>

using namespace copilot;

int main(){
    Network network;
    network.load_from_file("weights/weights.txt");

    std::vector<double> test_input = {1,9999,9999,9999,0};

    std::vector<double> result_vector = network.feed(test_input);
    int result = *std::max_element(result_vector.begin(), result_vector.end());

    return 0;
}