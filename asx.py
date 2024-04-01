from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from joblib import load
import tkinter as tk
from tkinter import messagebox
import requests
from bs4 import BeautifulSoup

model = load('model.joblib')
tfidf_vectorizer = load("tfidf_vectorizer.joblib")


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text


# Map predicted label to category name
categories = {0: 'Politics', 1: 'Sport', 2: 'Technology', 3: 'Entertainment-Daily Life', 4: 'Business', }


# Function to classify text
def classify_text(text):
    try:
        # Preprocess the text
        clean_text = preprocess_text(text)

        # Transform the preprocessed text into TF-IDF representation
        text_tfidf = tfidf_vectorizer.transform([clean_text])

        # Make predictions using the trained model
        predicted_label = model.predict(text_tfidf)

        # Map predicted label to category name
        categories = {0: 'Politics', 1: 'Sport', 2: 'Technology', 3: 'Entertainment-Daily Life', 4: 'Business'}
        predicted_category = categories[predicted_label[0]]

        # Show predicted category in a message box
        messagebox.showinfo("Category Prediction", f"The provided text belongs to the category: {predicted_category}")

    except Exception as e:
        # Show error message if an exception occurs
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to classify webpage content
def classify_webpage(url):
    try:
        # Extract text from the webpage
        webpage_text = extract_text_from_url(url)

        # Preprocess the text
        clean_webpage_text = preprocess_text(webpage_text)

        # Transform the preprocessed text into TF-IDF representation
        webpage_tfidf = tfidf_vectorizer.transform([clean_webpage_text])

        # Make predictions using the trained model
        predicted_label = model.predict(webpage_tfidf)

        # Map predicted label to category name
        predicted_category = categories[predicted_label[0]]

        # Show predicted category in a message box
        messagebox.showinfo("Category Prediction",
                            f"The content of the webpage belongs to the category: {predicted_category}")

    except Exception as e:
        # Show error message if an exception occurs
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
print("ALLGOOOD TİLL NOW")

# Create the main window
root = tk.Tk()
root.title("Content Classifier")

# Create a label and an entry widget for entering text
text_label = tk.Label(root, text="Enter the text:")
text_label.pack()
text_entry = tk.Entry(root, width=50)
text_entry.pack()

# Create a button to trigger text classification
classify_text_button = tk.Button(root, text="Classify Text", command=lambda: classify_text(text_entry.get()))
classify_text_button.pack()

# Create a label and an entry widget for entering the URL
url_label = tk.Label(root, text="Enter the URL of the webpage:")
url_label.pack()
url_entry = tk.Entry(root, width=50)
url_entry.pack()

# Create a button to trigger webpage classification
classify_webpage_button = tk.Button(root, text="Classify Webpage", command=lambda: classify_webpage(url_entry.get()))
classify_webpage_button.pack()

# Run the Tkinter event loop
root.mainloop()
