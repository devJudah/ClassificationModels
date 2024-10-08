# Logistic Regression Classifier - Gradient Descent

## 📄 Project Overview
This repository contains a simple implementation of a logistic regression classifier built from scratch, trained using the gradient descent optimization algorithm. The goal of this project is to demonstrate how logistic regression can be implemented without using any external machine learning libraries, focusing on understanding the underlying mathematical principles behind classification and model optimization.

The lab session is organized into two components:
- Logistic Regression Classifier ('logisticregression.py').
- Training and Evaluating the Classifier using a Jupyter Notebook ('trainlogreg.ipynb')

## 📂 Files in This Repository
### logisticregression.py
This script contains the core implementation of the logistic regression classifier:

- __init__(self): Initializes model parameters.
- sigmoid(self, z): Computes the sigmoid activation function.
- compute_cost(self, X, y): Implements the binary cross-entropy loss function.
- gradient_descent(self, X, y, learning_rate, num_iterations): Performs gradient descent to minimize the cost function.
- predict(self, X): Predicts the binary class labels (0 or 1) for the given input.
