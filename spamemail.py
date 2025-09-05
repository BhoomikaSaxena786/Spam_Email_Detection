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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Initial Cleaning ---
try:
    df = pd.read_csv("spam.csv", on_bad_lines="skip", sep="\t", encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Set column names and map labels
df.columns = ["Label", "Message"]
df['Label'] = df['Label'].map({'spam': 0, 'ham': 1})

# Handle missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(keep='first', inplace=True)

# --- 2. Exploratory Data Analysis (EDA) ---
print("--- Data Information ---")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nNumber of duplicate values after removal:", df.duplicated().sum())
print("\nDataset shape:", df.shape)
print("\nLabel distribution:\n", df["Label"].value_counts())

# Visualize label distribution
plt.figure(figsize=(6, 6))
plt.pie(df["Label"].value_counts(), labels=['ham', 'spam'], autopct="%0.2f%%", colors=['skyblue', 'salmon'])
plt.title('Distribution of Spam vs. Ham Messages')
plt.show()

# --- 3. Feature Engineering from Text ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Number of characters
df['num_alphabet'] = df['Message'].apply(len)
# Number of words
df['num_words'] = df['Message'].apply(lambda x: len(nltk.word_tokenize(x)))
# Number of sentences
df['num_sent'] = df['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))

print("\n--- Message Feature Statistics ---")
print("\nSpam statistics:\n", df[df['Label'] == 0][['num_alphabet', 'num_words', 'num_sent']].describe())
print("\nHam statistics:\n", df[df['Label'] == 1][['num_alphabet', 'num_words', 'num_sent']].describe())

# Visualize feature distributions
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Label'] == 0]['num_alphabet'], color='red', label='Spam')
sns.histplot(df[df['Label'] == 1]['num_alphabet'], color='blue', label='Ham')
plt.title('Distribution of Number of Characters')
plt.legend()
plt.show()

# --- 4. Text Preprocessing and Model Building ---
ps = PorterStemmer()

def transform_text(text):
    """
    Cleans and processes text by:
    - Lowercasing
    - Tokenizing
    - Removing non-alphanumeric characters
    - Removing stopwords and punctuation
    - Applying stemming
    """
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

# Apply the text transformation
df['transformed_message'] = df['Message'].apply(transform_text)
print("\n--- Transformed Text Examples ---")
print(df[['Message', 'transformed_message']].head())

# --- 5. Model Training and Evaluation ---
X = df['transformed_message']
y = df['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline for TfidfVectorization and Naive Bayes Classifier
model_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(max_features=3000)),
    ('naive_bayes', MultinomialNB())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# --- 6. Performance Metrics ---
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example prediction on new message
new_message = ["Congratulations, you've won a free prize! Click the link now."]
prediction = model_pipeline.predict(new_message)
if prediction[0] == 0:
    print("\n'{}' is predicted as: SPAM".format(new_message[0]))
else:

    print("\n'{}' is predicted as: HAM".format(new_message[0]))
