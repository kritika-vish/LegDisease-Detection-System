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
![image Alt](https://github.com/kritika-vish/LegDisease-Detection-System/blob/096f98f53e5576e2f5d92ece71c600d862d0c406/App%20interface/Screenshot%202026-03-15%20111748.png)

Input interface:
User provides patient information and uploads an image

Output Example:

![image Alt](https://github.com/kritika-vish/LegDisease-Detection-System/blob/9b494afff20dc6ed475b7db48dd88d6b074f1f30/App%20interface/Screenshot%202026-03-15%20113715.png)

If the uploaded image is not valid:

![image Alt](https://github.com/kritika-vish/LegDisease-Detection-System/blob/190bd3f8b1ef914e3a77a9c8d2794544a2509faa/App%20interface/Screenshot%202026-03-15%20113912.png)

The history of the predictions done is also shown at the bottom.
![image Alt](https://github.com/kritika-vish/LegDisease-Detection-System/blob/f850ff6d8e013afe40e63092dae43dbb7441c9c3/App%20interface/Screenshot%202026-03-15%20114040.png)



