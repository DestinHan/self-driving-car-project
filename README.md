# Self Driving Car Project 
Computer Vision Final Project

## Group 5
GitHub : https://github.com/DestinHan/self-driving-car-project


## 1. Group Members
Seung Hoon Han  
Anna Sokol  
Tommy Chabiras  

## 2. Introduction
In this project, we developed a convolutional neural network model that predicts the steering angle of a self-driving car using images captured from the front camera. The main goal was to train the model using simulator data and then use it to control the car autonomously in real time.

## 3. Approach
### Data Collection
We used the given self driving car simulator to collect our dataset. While manually driving the car, images from the center camera and the corresponding steering angles were recorded and stored in a CSV file driving_log.csv
### Preprocessing
To improve model performance, we applied several preprocessing steps:  
- Balanced the dataset to reduce straight driving bias  
- Cropped the image to focus only on the road area  
- Converted images from RGB to YUV color space  
- Applied Gaussian blur to reduce noise  
- Resized images to 200×66 pixels  
- Normalized pixel values to the range [0, 1]  
### Augmentation
To further improve model performance and prevent overfitting, we used the following augmentation techniques:
- Image flipping
- Brightness adjustments
- Zoom
- Panning
- Rotations


## 4. Model Architecture

We implemented a CNN model. The model includes:  
- Multiple convolutional layers for feature extraction  
- Fully connected layers for prediction  
- A final output layer with one neuron to predict the steering angle  


## 5. Results
Successfully trained model from the collected dataset, that can complete the car simulation track.  
- Moves forward along the track (forwards and back)  
- Correctly responds and steers during curves in the track
  
However, the model does struggle when facing scenarios not faced during dataset recording such as being manually put in unfamiliar positions.  

## 6. Challenges and Solutions
During our project implementation we faced a few challenges:
- Different TensorFlow versions caused confusion and errors. We standardized on TensorFlow 2.19.1 for the whole team.
- Thinking 20 epochs would always be best. Training can stop improving earlier on validation, so we used early stopping and saved the best validation checkpoint (in our runs this often stopped around 15 epochs, but the number depends on validation, not a fixed choice).
- Different driving behavior on different laptops. We focused on matching the same model file, same dependencies, and the same Term 1 simulator setup. 
- The car did not move during testing. This was fixed by using the Term 1 behavioral cloning simulator instead of Term 2.
- Augmentation was too harsh. We reduced how often augmentations were applied by using a 0.3 probability for several transforms.
- Training runs looked different each time because random batch sampling changes training dynamics. We added code in train.py to save the best model based on validation loss, so we keep the best weights instead of assuming the last epoch is best.

## 6. Work Distribution

### Anna
- Implemented various data augmentation techniques
- Debugged issues in the model training process
- Tuned hyperparameters to improve model performance
- Tested and evaluated model results 
### Seung
- Designed and implemented image preprocessing
- Created the CNN model architecture
- Incorporated batch generation for large dataset loading
- Developed the model training process
### Tommy
- Prepared dataset from the simulator
- Analyzed and visualized the data
- Balanced the dataset to reduce straight driving bias
- Assisted in testing and tuning hyperparameters to improve performance



## 8. How to Run the Project
1. Use Python 3.11 in a virtual environment.
2. Install dependencies:
   pip install tensorflow==2.19.1 pandas==2.2.3 python-socketio==4.6.1 python-engineio==3.13.2 eventlet flask pillow opencv-python matplotlib scikit-learn
3. Use the Udacity Term 1 behavioral cloning simulator 
4. Run the training script: python src/train.py  
5. Copy the trained model: cp models/model.h5 model.h5  
6. Run the simulator and test script: python simulator/TestSimulation.py  

## 9. Conclusion

Overall the project demonstrates the ability of using convolutional neural networks in autonomous driving tasks. Training using the simulator data allowed the model to effectively learn how to steer from the simulator’s visual input.

Although the system performs well in standard conditions, there could be further improvements made with additional training data, as well as different or more advanced architectures to improve the system in different environments and situations.
