// Network settings
#include "settings.h"
#include <string>
#include <vector>

using namespace copilot;
            
    Settings::Settings(){
        layers = 3; // match these with what you trained on in python
        layout = {5,8,3};
        activation = "tanh";
    }

    Settings::~Settings(){

    }
