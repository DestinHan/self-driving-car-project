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

## 7. How to Run the Project
1. Install required dependencies  
2. Run the training script: python src/train.py  
3. Copy the trained model: cp models/model.h5 model.h5  
4. Run the simulator and test script: python simulator/TestSimulation.py  

## 8. Conclusion

Overall the project demonstrates the ability of using convolutional neural networks in autonomous driving tasks. Training using the simulator data allowed the model to effectively learn how to steer from the simulator’s visual input.

Although the system performs well in standard conditions, there could be further improvements made with additional training data, as well as different or more advanced architectures to improve the system in different environments and situations.
