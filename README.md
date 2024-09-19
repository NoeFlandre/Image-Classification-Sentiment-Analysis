# Image Classification and Sentiment Analysis Web App

## Overview
This project is a Flask-based web application that performs two main tasks:
1. **Image Classification**: Users can upload an image, and the app will predict the object in the image using a pre-trained DenseNet121 model from PyTorch.
2. **Sentiment Analysis**: Users can input a sentence, and the app will analyze and return the sentiment (positive or negative) using a Hugging Face model.

The application is styled using Bootstrap and custom CSS, providing an easy-to-use interface with a hub to switch between image classification and sentiment analysis functionalities.

## Features
- **Image Classification**: Upload an image and get predictions of the object in the image.
- **Sentiment Analysis**: Enter a sentence to get a sentiment score and confidence level.
- **Simple Interface**: User-friendly design with navigation back to the home page after processing results.

## Project Structure
Here is an overview of the main files in this project:

- **`app.py`**: The main Flask application that handles routing, file uploads, image prediction, and sentiment analysis.
- **`commons.py`**: Utility functions and helper code used by the app.
- **`imagenet_class_index.json`**: Contains the ImageNet class IDs and corresponding labels for image classification.
- **`index.html`**: The home page of the application where users can choose between image classification and sentiment analysis.
- **`result.html`**: Displays the prediction results for the uploaded image.
- **`sentiment_result.html`**: Displays the sentiment analysis results for the input sentence.
- **`style.css`**: Custom CSS for styling the web app.
