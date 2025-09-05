**Spam Detection Web App**
This project is a machine learning-based web application that classifies messages as either "spam" or "ham" (not spam). The backend is built with Python and Flask, and the model uses a Naive Bayes classifier trained on a preprocessed text dataset.

**Prerequisites**
Before running the application, ensure you have Python installed. Then, install the required libraries by running the following command in your terminal:

Bash

pip install pandas scikit-learn nltk flask
Project Structure
Your project directory should be organized as follows:

spam-detector/
├── spam.csv
├── app.py
└── templates/
    └── index.html


spam.csv: The dataset containing messages and their corresponding spam/ham labels.

app.py: The main Python script that contains the machine learning model, text preprocessing logic, and the Flask web server.

templates/: A folder required by Flask to store HTML files.

index.html: The frontend web page where users can input text and get a classification.

How to Run the Application
Place the files: Make sure spam.csv and app.py are in the same directory, and the index.html file is inside the templates folder.

Start the server: Open your terminal or command prompt, navigate to the project directory, and run the Python script:

Bash

python app.py
Access the web app: Once the server is running, open a web browser and go to the following URL:

http://127.0.0.1:5000
You will see the web interface where you can test the spam detection model.

<img width="1920" height="1080" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/37cac157-ad24-4400-ab23-b6b27ebe1bee" />

How It Works
Backend (app.py):

The script first loads and cleans the spam.csv dataset.

<img width="1920" height="1080" alt="Screenshot (114)" src="https://github.com/user-attachments/assets/a4ef1709-1bca-47e3-9ab6-51c524c45eb8" />

<img width="1920" height="1080" alt="Screenshot (115)" src="https://github.com/user-attachments/assets/b8060b18-9c4b-4f36-a404-5e0e397310f7" />

It then preprocesses the text data by removing stopwords, punctuation, and applying stemming.

A machine learning pipeline is created using TfidfVectorizer (for text feature extraction) and a MultinomialNB (Naive Bayes) classifier.

This pipeline is trained on the dataset to learn the patterns of spam and ham messages.

Finally, a Flask web server is set up to serve the index.html file and handle classification requests.

Frontend (index.html):

The HTML page provides a simple user interface with a text area and a button.

When a user clicks "Classify," a JavaScript function sends the message to the /classify endpoint on the Flask server.

The server uses the trained model to make a prediction and sends the result back to the frontend.

The UI then dynamically updates to show whether the message is "SPAM" or "HAM" with clear visual feedback.
