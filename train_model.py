# -------------------------------
# Import required libraries
# -------------------------------

import pandas as pd  # for loading and handling dataset

from sklearn.feature_extraction.text import TfidfVectorizer  # text → numbers conversion

from sklearn.linear_model import LogisticRegression  # ML classification model

from sklearn.model_selection import train_test_split  # (optional in future use for splitting data)

import joblib  # for saving trained model and vectorizer


# -------------------------------
# Load dataset (training data only)
# -------------------------------

data = pd.read_csv("dataset.csv")
print(data["sentiment"].value_counts())

# -------------------------------
# Split dataset into input (X) and output (y)
# -------------------------------

# Text reviews (features)
x = data["review"]

# Sentiment labels (target)
y = data["sentiment"]

# -------------------------------
# Convert text into numerical format using TF-IDF
# -------------------------------

# ngram_range=(1,2) = single words + word pairs
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    max_df=0.8,
    min_df=2
)

# Learn vocabulary from text and transform into feature matrix
x_vectorized = vectorizer.fit_transform(x)

# -------------------------------
# Create Logistic Regression model
# -------------------------------

# max_iter=1000 ensures model converges properly
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(x_vectorized, y)

# -------------------------------
# Save trained model and vectorizer
# -------------------------------

joblib.dump(model, "model.pkl")          # saves trained model
joblib.dump(vectorizer, "vectorizer.pkl")  # saves TF-IDF transformer
