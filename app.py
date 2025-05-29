import streamlit as st
import pickle

# Load trained classifier
with open("feedback_model.pkl", "rb") as f:
    classifier = pickle.load(f)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Title
st.title("Customer Feedback Sentiment Analyzer")

# Input text
feedback = st.text_area("Enter customer feedback:")

# Analyze button
if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        vect_text = vectorizer.transform([feedback]).toarray()
        prediction = classifier.predict(vect_text)

        # Convert numeric prediction to label
        sentiment_map = {1: "Positive", 0: "Negative"}
        sentiment = sentiment_map[prediction[0]]


        st.success(f"Predicted Sentiment: **{sentiment}**")
