# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request, jsonify

# --- Model Training Section ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Text Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load and preprocess the data
try:
    df = pd.read_csv("spam.csv", on_bad_lines="skip", sep="\t", encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure the file is in the same directory.")
    exit()

df.columns = ["Label", "Message"]
df['Label'] = df['Label'].map({'spam': 0, 'ham': 1})
df.dropna(inplace=True)
df.drop_duplicates(keep='first', inplace=True)
df['transformed_message'] = df['Message'].apply(transform_text)

# Train the model
X = df['transformed_message']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(max_features=3000)),
    ('naive_bayes', MultinomialNB())
])
model_pipeline.fit(X_train, y_train)

# --- Flask Web Server Section ---
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(force=True)
    message = data['message']
    
    # Preprocess the message
    transformed_message = transform_text(message)
    
    # Get the prediction from the trained model
    prediction = model_pipeline.predict([transformed_message])
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    # Ensure the template folder is correct if you have issues
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True)