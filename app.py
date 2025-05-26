import streamlit as st
import pickle

with open("feedback_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Streamlit app
st.title("Customer Feedback Sentiment Analyzer")

feedback = st.text_area("Enter customer feedback:")

if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        vect_text = vectorizer.transform([feedback])
        prediction = model.predict(vect_text)
        st.success(f"Predicted Sentiment: **{prediction[0]}**")
