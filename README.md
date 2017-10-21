# AlexNet Feature Extraction
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This lab guides you through using AlexNet and TensorFlow to build a feature extraction network.

## Setup
Before you start the lab, you should first install:
* Python 3
* TensorFlow
* NumPy
* SciPy
* matplotlib

## Yangchun's addition

### trained\_feature\_inference.py

Load the previously trained model (by train_feature_extraction.py) and do inference on traiffic sign images. Output:

|   <img src="construction.jpg">         |     Prediction  | 
|:----------------:|:-------------------------:| 
| Road work | 1.000 |
| Children crossing | 0.000 |
| Slippery road | 0.000 |
| Road narrows on the right | 0.000 |
| Speed limit (70km/h) | 0.000 |

|   <img src="stop.jpg">         |     Prediction  | 
|:----------------:|:-------------------------:| 
| Speed limit (70km/h) | 0.757 |
| Stop | 0.232 |
| Speed limit (100km/h) | 0.011 |
| Keep right | 0.000 |
| No passing | 0.000 |

### train\_lenet\_cifra10.py

Train the customized lenet (LuoNet in project 2) with CIFRA-10 dataset. Accuracy after 10 epochs:

* Train: 59.2%
* Validation: 56.7%
* Test: 56.6%