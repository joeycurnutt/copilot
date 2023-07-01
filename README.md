# copilot
copilot is a library for VEX Robotics that implements deep learning techniques to skillfully automate robotics

A C++ implementation that pairs with [Mark Dai's](github.com/Markerpullus) Neural Network from scratch in Python

### Installation
This is designed as a pros template, so there will be a specific list of steps to follow for installation:
  1. Download the latest release. This represents the last stable update of code and can be modified as you wish for your needs.
  2. Extract and open the folder in a pros project
  3. Use "pros make template" to create a zipped folder, should be "copilot@version_number"
  4. In your project that will be uploaded to the robot, open a terminal in that directory and run "pros c fetch copilot"
  5. At this point the code is ready for use

### Capabilities
* Binary classification
* Multiclass classification
* Regression
* Training with dropout

New release coming once functionality is confirmed, v1.0.0 only includes multiclass classification

### TODO
* Optimize memory usage and speed
* RNN and CNN?
