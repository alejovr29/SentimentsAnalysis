import os
import re
import pickle  # To load the tokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

# --- Configuration ---
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_SEQUENCE_LENGTH = 100  # Must match the training max_length

# --- Load Model and Tokenizer ---
# Load the trained Keras model
# Error handling is basic; consider more robust checks in a real app
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("* Keras model loaded successfully.")
except Exception as e:
    print(f"* Error loading Keras model: {e}")
    model = None # Set model to None if loading fails

# Load the tokenizer object
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("* Tokenizer loaded successfully.")
except Exception as e:
    print(f"* Error loading tokenizer: {e}")
    tokenizer = None # Set tokenizer to None if loading fails

# Initialize Flask application
app = Flask(__name__, template_folder=os.path.abspath('templates'))

# --- Text Cleaning Function (Copied from your Colab code) ---
def clean_text(text):
    """Cleans the input text."""
    # Ensure text is not None before processing
    if text is None:
        return ""
    text = str(text).lower() # Ensure input is string and lowercase
    text = re.sub(r"<.*?>", "", text)          # Remove HTML-like tags
    text = re.sub(r"[^a-z\s]", "", text)      # Remove punctuation, numbers, symbols
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

# --- Prediction Function ---
def predict_sentiment(text):
    """
    Cleans, preprocesses, and predicts sentiment for the input text.
    Returns 'Positive' or 'Negative'.
    """
    # Check if model and tokenizer loaded correctly
    if model is None or tokenizer is None:
        print("* Error: Model or Tokenizer not loaded.")
        return "Error: Model/Tokenizer unavailable" # Return error message

    # 1. Clean the input text
    cleaned_text = clean_text(text)
    print(f"* Cleaned text: {cleaned_text}")

    # Handle empty string after cleaning
    if not cleaned_text:
        print("* Input text is empty after cleaning.")
        return "No text to analyze"

    # 2. Tokenize the cleaned text
    # Note: texts_to_sequences expects a list of texts
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    print(f"* Sequenced text: {sequence}")

    # 3. Pad the sequence
    # Use the same max_length, padding, and truncating settings as training
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )
    print(f"* Padded sequence (first 10): {padded_sequence[0][:10]}")

    # 4. Make prediction using the loaded model
    try:
        prediction_probability = model.predict(padded_sequence)[0][0] # Get the probability from the output
        print(f"* Prediction probability: {prediction_probability}")
    except Exception as e:
        print(f"* Error during prediction: {e}")
        return "Error during prediction"

    # 5. Interpret the prediction
    sentiment = "Positive" if prediction_probability > 0.5 else "Negative"
    print(f"* Predicted sentiment: {sentiment}")

    return sentiment

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles requests for the main page."""
    prediction_result = None
    input_text = ""

    if request.method == 'POST':
        # Get text from the form submission
        input_text = request.form.get('texto_usuario', '') # Use .get for safer access
        if input_text:
            # Call the prediction function
            prediction_result = predict_sentiment(input_text)
        else:
            prediction_result = "Please enter some text."

    # Render the HTML template, passing the result and original text
    return render_template('index.html', prediccion=prediction_result, texto=input_text)

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False when deploying to production!
    app.run(debug=True)