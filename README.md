# copilot
copilot is a library for VEX Robotics that implements deep learning techniques to skillfully automate robotics

A C++ implementation that pairs with [Mark Dai's](https://github.com/Markerpullus) Neural Network from scratch in Python

### Installation and Use - C++
This is designed as a pros template, so there will be a specific list of steps to follow for installation:
  1. Download the code. Choose the release for latest stable version, or download the branch for more experimental code.
  2. Extract and open the folder in a pros project
  3. Use "pros make template" to create a zipped folder, should be "copilot@version_number"
  4. In your project that will be uploaded to the robot, open a terminal in that directory and run "pros c fetch copilot"
  5. At this point the code is ready for use

### Installation and Use - Python
Once the code has been downloaded, the python code can be run in the downloaded directory, no need to move it into your main project unless you desire. The template will not include anything but source and header files in C++.

### Capabilities
* Binary classification
* Multiclass classification
* Regression
* Training with dropout

New release coming once functionality is confirmed, v1.0.0 only includes multiclass classification

### TODO
* Optimize memory usage and speed
* RNN and CNN?
