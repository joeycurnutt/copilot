//Usage example
#include "cppsrc/Network.h"
#include "cppsrc/Settings.h"
#include <bits/stdc++.h>

using namespace copilot;

int main(){
    Network network("weights/weights.txt");

    std::vector<double> test_input = {0,9999,9999,9999,0}; // vector of sensor values

    std::vector<double> result_vector = network.feed(test_input); // feeds data through. Make this a loop of your sensor values when on robot.
    int result = int(result_vector[0]) // gets the index of the max value (highest probability = predicted class)

    return 0;
}