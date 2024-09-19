import os
from flask import Flask, render_template, request, redirect, jsonify
import torch
from torchvision import models
from commons import preprocess_image, get_prediction  # Assuming commons.py has these functions
from transformers import pipeline


app = Flask(__name__)

# Load the pre-trained DenseNet121 model once when the app starts
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

# Load pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Renders the file upload interface and processes uploaded files to make predictions.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        
        # Read the uploaded image file
        img_bytes = file.read()
        
        # Save the image temporarily for prediction
        temp_image_path = 'static/temp_image.png'
        with open(temp_image_path, 'wb') as temp_file:
            temp_file.write(img_bytes)
        
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(temp_image_path)
        
        # Get the prediction from the model
        class_id, class_name = get_prediction(preprocessed_image, model)
        
        # Clean up the temporary image file (optional, you can keep it if you need it)
        os.remove(temp_image_path)
        
        # Render the result template with class_id and class_name
        return render_template('result.html', class_id=class_id, class_name=class_name)
    
    # Render the upload form template for GET request
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Analyzes the sentiment of the provided sentence.
    """
    sentence = request.form.get('sentence')
    
    # Use the sentiment analysis pipeline
    result = sentiment_analyzer(sentence)
    
    # Extract the label (positive/negative) and confidence score
    sentiment = result[0]['label']
    confidence = result[0]['score']
    
    # Render the sentiment analysis result
    return render_template('sentiment_result.html', sentiment=sentiment, confidence=confidence)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """
    Predict the class of a fixed image (image.png) and return JSON response.
    """
    # Preprocess the fixed image (image.png)
    image_path = 'Tajine.jpeg'  # Fixed image path
    preprocessed_image = preprocess_image(image_path)
    
    # Get the prediction from the model
    class_id, class_name = get_prediction(preprocessed_image, model)

    # Define the JSON response with the prediction
    response = {
        'class_id': class_id,
        'class_name': class_name
    }

    # Return the JSON response
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
