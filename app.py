import os
import joblib
import streamlit as st

# App title and description
st.title("Internship Feedback Sentiment Analysis")
st.write("Enter feedback below to predict whether its sentiment is positive or negative.")

# Load the saved model and vectorizer from this project's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Collect feedback from the user
review = st.text_area("Enter internship feedback")

# Predict sentiment when the button is clicked
if st.button("Analyze Sentiment"):

    # Prevent prediction if the input is empty
    if not review.strip():
        st.warning("Please enter some feedback first.")

    else:
        # Normalize text for the simple negation rules
        review = review.lower().strip()

        # Handle a few common negation phrases
        if "not negative" in review:
            st.success("Positive Feedback 🙂")

        elif "not positive" in review or "not good" in review:
            st.error("Negative Feedback 😟")

        elif "not bad" in review:
            st.success("Positive Feedback 🙂")

        else:
            # Convert feedback into TF-IDF features
            review_vectorized = vectorizer.transform([review])

            # Predict sentiment using the trained model
            prediction = model.predict(review_vectorized)[0]

            # Display the result
            if prediction == "positive":
                st.success("Positive Feedback 🙂")
            else:
                st.error("Negative Feedback 😟")
