# AI Image Processing and Classification Project

This project is designed to give you hands-on experience working with an image classifier and enhancing your programming skills using AI assistance. The project has two parts, each focused on different aspects of image classification and processing. By the end, you'll have explored fundamental concepts like Grad-CAM, image classification, and creative image filtering.

## Overview of Scripts

This repository contains two main Python scripts:

### 1. `base_classifier.py`
This script uses a pre-trained Deep Learning model (**MobileNetV2** via TensorFlow/Keras) to classify the main object in an image.
- It loads an image, resizes it to 224x224 pixels (the format expected by MobileNetV2), and preprocesses it.
- It feeds the image to the model to predict what the image contains.
- It outputs the top 3 most likely classifications along with their confidence scores.

### 2. `basic_filter.py`
This script applies a simple image processing effect (a Gaussian blur) to an image.
- It opens an image using the **Pillow (PIL)** library and resizes it to 128x128 pixels.
- It applies a `GaussianBlur` filter with a radius of 2 to blur the image.
- Finally, it uses `matplotlib` to save the new blurred image with a `_blurred` suffix added to the original filename.

## Setup and Installation

To run this project, you need to set up a Python virtual environment and install the required dependencies:

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   Make sure you have pip upgraded and install the packages from `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Running the Image Classifier
Run the classifier script from your terminal:
```bash
python base_classifier.py
```
You will be prompted to enter the path to an image file. The script will output the top 3 predictions. Type `exit` to quit.

### Running the Image Filter
Run the simple blur filter script from your terminal:
```bash
python basic_filter.py
```
Type the path to an image file when prompted. The script will quickly apply a Gaussian blur to the image and save a new file in the same directory, appending `_blurred` to the name. Type `exit` to quit.
