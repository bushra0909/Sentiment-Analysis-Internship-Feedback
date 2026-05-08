# Import Streamlit for web app UI
import streamlit as st
# Import joblib for loading saved model/vectorizer 
import joblib
# Import os for safe file paths
import os
# -------------------------------
# App Title
# -------------------------------
st.title("Intern Feedback Sentiment Analysis")
# -------------------------------
# Load saved model and vectorizer
# -------------------------------
# Get current folder path
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
# Create full paths
model_path=os.path.join(BASE_DIR,"model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
# Load trained files
model=joblib.load(model_path)
vectorizer=joblib.load(vectorizer_path)
# -------------------------------
# User text input
# -------------------------------
review=st.text_area("Enter Internship feedback")
# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Analyze Sentiment"):

    # Convert review into vector format
  # Convert user input into lowercase
# so comparisons become case-insensitive
 review = review.lower()

# -------------------------------
# Negation handling rules
# -------------------------------
# These rules handle special cases where
# words like "not" flip the sentiment meaning

# Example:
# "not negative" actually means positive
if "not negative" in review:
    st.success("Positive Feedback 🙂")

# Example:
# "not positive" means negative sentiment
elif "not positive" in review:
    st.error("Negative Feedback 😟")

# Example:
# "not good" changes positive word into negative meaning
elif "not good" in review:
    st.error("Negative Feedback 😟")

# Example:
# "not bad" usually means somewhat positive
elif "not bad" in review:
    st.success("Positive Feedback 🙂")

# -------------------------------
# ML prediction section
# -------------------------------
# If no special negation rule matches,
# use the trained ML model for prediction
else:

    # Convert review text into vector format
    # using trained TF-IDF vectorizer
    review_vectorized = vectorizer.transform([review])

    # Predict sentiment using trained model
    prediction = model.predict(review_vectorized)

    # -------------------------------
    # Display prediction result
    # -------------------------------

    # If prediction is positive
    if prediction[0] == "positive":
        st.success("Positive Feedback 🙂")

    # Otherwise show negative result
    else:
        st.error("Negative Feedback 😟")