# Logistic Regression Classifier - Gradient Descent

## 📄 Overview
This repository contains a simple implementation of a logistic regression classifier built from scratch, trained using the gradient descent optimization algorithm. The goal of this project is to demonstrate how logistic regression can be implemented without using any external machine learning libraries, focusing on understanding the underlying mathematical principles behind classification and model optimization.

The lab session is organized into two components:
- Logistic Regression Classifier (`logisticregression.py`).
- Training and Evaluating the Classifier using a Jupyter Notebook (`trainlogreg.ipynb`)

## 📂 Files in This Repository
(`logisticregression.py`).
This script contains the core implementation of the logistic regression classifier:

- `__init__(self)`: Initializes model parameters.
- `sigmoid(self, z)`: Computes the sigmoid activation function.
- `compute_cost(self, X, y)`: Implements the binary cross-entropy loss function.
- `gradient_descent(self, X, y, learning_rate, num_iterations)`: Performs gradient descent to minimize the cost function.
- `predict(self, X)`: Predicts the binary class labels (0 or 1) for the given input.

`trainlogreg.ipynb`
This Jupyter notebook demonstrates:

Dataset Loading: Preparing a dataset for training.
Model Training: Training the logistic regression classifier using gradient descent.
Evaluation: Assessing the model's accuracy and visualizing its performance.

## 🔑 Key Features
From-Scratch Implementation: A basic logistic regression classifier is implemented without external ML libraries.
Gradient Descent: Trains the model by minimizing the cost function through gradient descent.
Accuracy Evaluation: The Jupyter Notebook provides a step-by-step guide for training the model and visualizing the results.
