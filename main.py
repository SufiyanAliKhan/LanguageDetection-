# Import necessary packages
import pickle
import streamlit as st
import re

# Load the model and vectorizer
with open('language_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer_lang.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)


# Function to clean the input text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text


# Streamlit app interface
st.title("Language Detection App")
st.write("Predict the language of the given text!")

# Input text box
input_text = st.text_area("Enter text here:")

# Detect language button
if st.button("Detect Language"):
    if input_text.strip():  # Check if input is not empty
        cleaned_text = clean_text(input_text)

        # Transform the input text using the loaded vectorizer
        input_vectorized = tfidf.transform([cleaned_text])

        # Predict the language using the loaded model
        predicted_language = model.predict(input_vectorized)[0]

        # Display the result
        st.write(f"Detected Language: {predicted_language}")
    else:
        st.write("Please enter some text to detect the language.")
