// Network settings
#include "settings.h"
#include <string>
#include <vector>

using namespace copilot;
            
    Settings::Settings(){
        layers = 3; // match these with what you trained on in python
        layout = {4,8,3};
        activation = "tanh";
        classification = true; // is a classification vs a regression
        

        if(classification){
            if(layout.back() > 1){
                last_layer = "softmax"; // if multi-class classification use softmax
            } else {
                last_layer = "sigmoid"; // if not use sigmoid
            }
        } else{
            last_layer = "linear"; //last layer of regression network is linear
        }
    }

    Settings::~Settings(){

    }
