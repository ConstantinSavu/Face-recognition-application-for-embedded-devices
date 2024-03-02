# Face Recognition Application Project Report

## Overview
This project report details the design and implementation of a face recognition application. The application uses a compact Convolutional Neural Network (CNN) architecture, MobileNet V3 Small, which is optimized for functionality on an embedded device.

## Key Features
- **Compact CNN Architecture**: The application utilizes MobileNet V3 Small for efficient processing.
- **Euclidean Distance Metric**: The application employs Euclidean distance for identifying the 'closest' match.
- **K-Nearest Neighbors (KNN) Classifier**: The application leverages a KNN classifier for improved identification accuracy.
- **Optimized for Embedded Devices**: The application has been successfully tested on a Raspberry Pi.

## Pre-processing Stage
The application includes a pre-processing stage where initial images undergo face cropping using a Haar Cascade Classifier from the open-cv library. This step is crucial for enhancing the accuracy of the system.

## Results
The system achieved an accuracy of 80% with 3 neighbors and 100% with 7 neighbors. This indicates that the optimal number of neighbors for this application is 7.

## Future Work
The success of this project opens avenues for further exploration and refinement. Future work could include investigating the impact of different distance metrics, classifiers, or CNN architectures on performance.
