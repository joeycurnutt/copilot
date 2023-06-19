// Network settings
#include "settings.h"
#include <string>
#include <vector>

using namespace copilot;
            
    Settings::Settings(){
        layers = 3; // match these with what you trained on in python
        layout = {5,8,3};
        activation = "tanh";
        classification = true; // is a classification vs a regression

        if(classification){
            if(layout.back() > 1){
                last_layer = "softmax";
            } else {
                last_layer = "sigmoid";
            }
        } else{
            last_layer = "linear"; //activation of the last layer of the network
        }
    }

    Settings::~Settings(){

    }
