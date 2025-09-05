**Spam Detection Web App**
This project is a machine learning-based web application that classifies messages as either "spam" or "ham" (not spam). The backend uses Python and Flask, with a Naive Bayes classifier trained on a text dataset. A clean, user-friendly frontend allows for real-time classification.

**Prerequisites**
Before running the application, ensure you have Python installed. Then, install the required libraries by running the following command in your terminal:

Bash

pip install pandas scikit-learn nltk flask

**Project Structure**
Your project directory should be organized as follows:

spam-detector/
├── spam.csv
├── app.py
└── templates/
    └── index.html
**spam.csv**: The dataset containing messages and their corresponding spam/ham labels.

**app.py**: The main Python script that handles the machine learning model, text preprocessing, and the Flask web server.

**templates/**: A folder for HTML files, required by Flask.

**index.html**: The frontend web page where users can interact with the classifier.

**How to Run the Application**
**Place the files**: Make sure spam.csv and app.py are in the same directory, and index.html is inside the templates folder.

**Start the server**: Open your terminal, navigate to the project directory, and run the Python script:

Bash

python app.py
**Access the web app**: Once the server is running, open a web browser and go to http://127.0.0.1:5000.


<img width="1920" height="1080" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/d0ba7509-9444-403f-97b0-acae7d463ef1" />


**Exploratory Data Analysis**
The dataset shows a significant class imbalance, with ham messages being far more common than spam messages. .

We also observe clear differences in the characteristics of spam and ham messages. Spam messages tend to be longer in terms of character count, word count, and sentence count compared to ham messages. .

**Model Performance**
The model was evaluated on a test set and demonstrated excellent performance with high accuracy, precision, and perfect recall for spam messages.

Accuracy: 97.14%

Precision (Spam): 96.83%

Recall (Spam): 100.00%

F1-Score (Spam): 98.39%

<img width="600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/06c1baad-8e90-4213-9ad6-4a2c55e7ccee" />
<img width="1200" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/7361c6c4-c5fe-47ef-8d40-fe4473083bb5" />

The confusion matrix below provides a detailed breakdown of the model's predictions on the test data.

Predicted Ham	Predicted Spam
Actual Ham	1129 (True Negatives)	37 (False Positives)
Actual Spam	0 (False Negatives)	126 (True Positives)

<img width="1920" height="1080" alt="Screenshot (114)" src="https://github.com/user-attachments/assets/1bfaea4f-a814-4eab-9b67-e86d4db0ee12" />
<img width="1920" height="1080" alt="Screenshot (115)" src="https://github.com/user-attachments/assets/87d53bd8-0de1-49c3-9e8d-e112c52feea0" />


Export to Sheets
The results show that the model is highly effective at identifying all spam messages, with a minimal number of false alarms.

**How It Works**
**Backend (app.py)**: The script preprocesses the text data using TF-IDF vectorization and trains a Multinomial Naive Bayes classifier. A Flask server is then used to serve the web interface and handle classification requests from the frontend.

**Frontend (index.html)**: The HTML page provides a clean UI where users can input text. A JavaScript function sends the message to the backend, and the UI dynamically updates to display whether the message is "SPAM" or "HAM" with clear visual feedback.
