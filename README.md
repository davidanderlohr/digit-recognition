# Handwritten Digit Recognition with MNIST

This repository contains a project where a machine learning model was trained to recognize handwritten digits using the MNIST dataset. The model achieved a test accuracy of 98.03%.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)

## Introduction

Handwritten digit recognition is a classic problem in the field of computer vision and machine learning. The goal of this project is to develop a model that can accurately classify digits (0-9) from the MNIST dataset.

## Dataset

The MNIST dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. It includes 60,000 training images and 10,000 test images. The dataset is automatically downloaded using `torchvision`.

## Model Architecture

The model was built using a fully connected neural network with the following architecture:

- **Input layer:** 28x28 grayscale images (flattened to 784 inputs)
- **Hidden layer 1:** 128 units, ReLU activation
- **Hidden layer 2:** 64 units, ReLU activation
- **Output layer:** 10 units (one for each digit), LogSoftmax activation

## Training

The model was trained using the following parameters:

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss function:** CrossEntropyLoss
- **Batch size:** 64
- **Epochs:** 200

## Results

The model achieved a test accuracy of **98.03%**. This high accuracy demonstrates the effectiveness of the model in recognizing handwritten digits.

## Usage

To use the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/davidanderlohr/digit-recognition.git
    cd handwritten-digit-recognition
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the training script train.py (optional):
    ```
    python train.py
    ```

4. To test the model with your own handwritten digits, use the provided predict.py script.
    ```
    python predict.py
    ```






