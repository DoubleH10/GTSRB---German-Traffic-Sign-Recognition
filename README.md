# German Traffic Sign Recognition with HOG+SVM

## Overview
In this project, I implemented a traffic sign recognition system using HOG (Histogram of Oriented Gradients) feature extraction combined with SVM (Support Vector Machine) classification. This approach provides an efficient and accurate method for recognizing traffic signs from the German Traffic Sign Dataset.

## Dataset
The German Traffic Sign Dataset contains more than 50,000 images across 43 different classes of traffic signs. The dataset includes various challenges like different lighting conditions, occlusions, and varying distances.

## Key Features
- **High Accuracy**: Achieves over 96% accuracy on the test set
- **Efficient Processing**: HOG feature extraction coupled with optimized SVM provides fast prediction
- **Preprocessing Pipeline**: Includes grayscale conversion, standardization, and feature scaling
- **User-friendly Interface**: Includes a Gradio web interface for easy testing

## How It Works
1. **Preprocessing**: Images are resized to 64x64 pixels and converted to grayscale
2. **Feature Extraction**: HOG features are computed from each image
3. **Model Training**: An SGD-based SVM classifier is trained on the standardized features
4. **Recognition**: New images are processed through the same pipeline for classification

## Dependencies
- NumPy - For numerical operations
- OpenCV - For image processing
- Scikit-learn - For SVM classifier and scaling
- Scikit-image - For HOG feature extraction
- Joblib - For model serialization
- Matplotlib - For visualization
- Gradio - For interactive web interface

## Usage
The project includes two main notebooks:
- `HOG+SVM.ipynb`: Trains the model and saves it to disk
- `GUI.ipynb`: Provides a user interface for testing the model with new images

## Results
Model Accuracy of 96,98%. The model demonstrates robust performance across different traffic sign categories, effectively handling variations in lighting, perspective, and image quality. The HOG+SVM approach provides a good balance between computational efficiency and recognition accuracy.

## Conclusion
This project demonstrates that classical computer vision techniques can still achieve impressive results for specific recognition tasks. While deep learning approaches may provide higher accuracy in some cases, the HOG+SVM pipeline remains relevant for applications with limited computational resources or when interpretability is important.

For real-world applications, this classification system would need to be paired with a detection algorithm to first locate traffic signs within larger images.