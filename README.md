# Sentiment-Analysis-Internship-Feedback
This project is a Machine Learning and NLP-based web application that analyzes internship feedback and predicts whether the sentiment is positive or negative.

## Features

- Text preprocessing
- TF-IDF vectorization
- Logistic Regression classification
- Sentiment prediction web app using Streamlit

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project Workflow

1. Load internship feedback dataset
2. Convert text into numerical vectors using TF-IDF
3. Train Logistic Regression model
4. Save trained model and vectorizer
5. Predict sentiment through Streamlit web app

## Files

- `train_model.py` → model training
- `app.py` → Streamlit web app
- `data.csv` → dataset
- `model.pkl` → saved trained model
- `vectorizer.pkl` → saved TF-IDF vectorizer

## How to Run

```bash
cd = go into folder
pip install = install libraries
streamlit run = launch app
pip install -r requirements.txt
streamlit run app.py
```
