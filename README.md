Sign Language Detector

A deep learning-based Sign Language Detection system that uses a custom CNN model and real-time hand tracking to predict sign language gestures.

Overview

This project implements an end-to-end pipeline for detecting sign language gestures using:

A custom-built Convolutional Neural Network (CNN)
MediaPipe hand tracking for real-time detection
TensorFlow/Keras for training and inference
Webcam-based live prediction
Model Features
Custom CNN architecture with multiple convolutional layers
Input image size: 48×48 (grayscale)
Training pipeline includes:
image_dataset_from_directory
Mixed precision for faster GPU performance
Dropout for regularization
Early stopping
Learning rate reduction
Final model exported in .h5 format

Real-Time Detection

The system performs the following steps:

Detects hand landmarks using MediaPipe
Extracts the hand region from the frame
Converts to grayscale and resizes to 48×48
Normalizes and feeds into the CNN model
Displays the predicted sign on screen

Core implementation is available in:

Current Limitations
Dataset images are low resolution (48×48)
Compared to standard datasets (224×224 or 256×256), this results in:
Loss of fine details
Pixel distortion
Reduced accuracy for certain classes
Some classes are not predicted correctly due to:
Dataset imbalance
Limited variation
Small input size
Future Improvements
Use higher-resolution datasets (128×128 or 224×224)
Improve dataset balance and diversity
Experiment with transfer learning models such as MobileNet or EfficientNet
Add prediction confidence thresholds
Improve real-time stability
Deploy as a web or mobile application
