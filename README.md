# Leg Disease Severity Detection System

This is an AI-based web application that detects leg disease severity from medical images using a two-stage classification approach.
The system first verifies whether the uploaded image is a leg image or not, and then predicts the disease severity level.

It implements a two-stage AI pipeline:
### Stage 1 – Leg Detection
A deep learning model checks whether the uploaded image contains a leg or not.
### Stage 2 – Disease Severity Prediction
If the image is valid, machine learning models classify the severity level of the infection.

## Features
✔ Two-stage AI classification system

✔ Random image rejection 

✔ Machine learning + deep learning hybrid model 

✔ Image preprocessing and feature extraction 

✔ Web interface for prediction 

✔ Patient information tracking

✔ Prediction history display 

## System Architecture
User upload image → Image Prepreprocessing → Stage 1 (Deep Learning) Leg / Non leg detection → If not leg, reject image → If leg image → Stage 2 Models (Machine Learning) SVM + KNN Classification → Disease Severity Prediction (Level1 / Level 2 / Level 3)

## Technologies Used
1. Python
2. Flask
3. TensorFlow / Keras
4. Scikit-learn
5. OpenCV
6. NumPy
7. HTML, CSS, JavaScript

## Models
Stage 1 – Leg Detection
- Model: MobileNet (Transfer Learning)
- Task: Binary Classification
- Classes:Leg, Non-Leg

Stage 2 – Disease Severity Prediction
- Two machine learning models are used: K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- Feature Extraction: Histogram of Oriented Gradients (HOG)

## Application Interface

Input interface:
User provides patient information and uploads an image

Output Example:
- Patient Name: Sanju
- Age: 31
- Gender: Male

KNN Prediction: Level 2       
SVM Prediction: Level 2

If the uploaded image is not a leg image:
→ The image is not valid for prediction and shows undefined



