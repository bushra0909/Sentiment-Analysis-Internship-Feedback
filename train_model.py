
# ============================================================
# INTERNSHIP FEEDBACK SENTIMENT ANALYSIS
# ============================================================
# This script:
# 1. Loads the internship feedback dataset.
# 2. Splits reviews into training and testing sets.
# 3. Converts text into numerical features using TF-IDF.
# 4. Trains a Logistic Regression classifier.
# 5. Evaluates the model on unseen test reviews.
# 6. Saves the trained model and TF-IDF vectorizer for app.py.
# ============================================================

import os                       # Helps build reliable file paths
import joblib                   # Saves and loads trained ML objects
import pandas as pd             # Loads and processes tabular data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1. DEFINE FILE PATHS
# ------------------------------------------------------------

# Find the folder containing this Python script.
# This ensures the script can find dataset.csv and save its
# output files in the same folder, regardless of the terminal's
# current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def main():

    print("=" * 55, flush=True)
    print(" INTERNSHIP FEEDBACK SENTIMENT ANALYSIS", flush=True)
    print("=" * 55, flush=True)

    # --------------------------------------------------------
    # 2. LOAD AND CLEAN THE DATASET
    # --------------------------------------------------------

    print("\n[1/6] Loading dataset...", flush=True)

    # Read the CSV file into a pandas DataFrame.
    data = pd.read_csv(DATA_PATH)

    # Remove rows where the review or its sentiment label is missing.
    data = data.dropna(subset=["review", "sentiment"]).copy()

    # Ensure reviews are strings and remove leading/trailing spaces.
    data["review"] = data["review"].astype(str).str.strip()

    # Standardize labels, e.g. "Positive" becomes "positive".
    data["sentiment"] = (
        data["sentiment"].astype(str).str.lower().str.strip()
    )

    # Remove reviews that are empty after cleaning.
    data = data[data["review"] != ""]

    # X contains the text the model learns from.
    X = data["review"]

    # y contains the correct sentiment label for each review.
    y = data["sentiment"]

    # Display dataset size and number of examples per class.
    print(f"Total reviews: {len(data):,}", flush=True)
    print("\nSentiment distribution:", flush=True)
    print(y.value_counts().to_string(), flush=True)

    # Check that classification is possible.
    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two sentiment classes."
        )

    # Stratified splitting needs at least two examples per class.
    if y.value_counts().min() < 2:
        raise ValueError(
            "Each sentiment class needs at least two examples."
        )

    # --------------------------------------------------------
    # 3. SPLIT DATA INTO TRAINING AND TESTING SETS
    # --------------------------------------------------------

    print("\n[2/6] Splitting dataset...", flush=True)

    # Training data teaches the model.
    # Test data checks performance on reviews not used for training.
    #
    # test_size=0.2 means 20% for testing and 80% for training.
    # random_state=42 makes the split reproducible.
    # stratify=y preserves the approximate class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training reviews: {len(X_train):,}", flush=True)
    print(f"Testing reviews: {len(X_test):,}", flush=True)

    # --------------------------------------------------------
    # 4. CONVERT TEXT INTO NUMERICAL FEATURES USING TF-IDF
    # --------------------------------------------------------

    print("\n[3/6] Converting text into TF-IDF features...", flush=True)

    # Machine-learning models need numerical input rather than
    # raw sentences. TF-IDF represents words and phrases using
    # numerical weights based on their importance in the reviews.
    #
    # ngram_range=(1, 2): use single words and two-word phrases.
    # stop_words="english": ignore common English stop words.
    # max_df=0.8: ignore terms appearing in more than 80% of
    # training documents.
    # min_df=2: ignore terms appearing in fewer than two documents.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        stop_words="english",
        max_df=0.8,
        min_df=2
    )

    # Learn the vocabulary and TF-IDF weights from training text.
    # Never fit the vectorizer on test text before evaluation:
    # that would leak information from the test set.
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test text using the already learned vocabulary.
    # Do not call fit_transform() on the test set.
    X_test_vec = vectorizer.transform(X_test)

    print(
        f"Number of learned features: "
        f"{len(vectorizer.vocabulary_):,}",
        flush=True
    )

    # --------------------------------------------------------
    # 5. TRAIN THE LOGISTIC REGRESSION MODEL
    # --------------------------------------------------------

    print("\n[4/6] Training Logistic Regression...", flush=True)

    # Logistic Regression is a supervised classification algorithm.
    # It learns patterns that help distinguish positive and negative
    # feedback from the TF-IDF features.
    #
    # max_iter=1000 allows more optimization iterations if needed.
    model = LogisticRegression(max_iter=1000)

    # Train the model using training features and known labels.
    model.fit(X_train_vec, y_train)

    print("Model training complete.", flush=True)

    # --------------------------------------------------------
    # 6. EVALUATE THE MODEL ON UNSEEN TEST REVIEWS
    # --------------------------------------------------------

    print("\n[5/6] Evaluating model performance...", flush=True)

    # Predict the sentiment of the held-out test reviews.
    predictions = model.predict(X_test_vec)

    # Accuracy = correct predictions / total predictions.
    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 55, flush=True)
    print(f" TEST ACCURACY: {accuracy:.2%}", flush=True)
    print("=" * 55, flush=True)

    # Precision: of the reviews predicted as a class, how many
    # actually belong to that class?
    #
    # Recall: of all actual reviews in a class, how many did the
    # model correctly identify?
    #
    # F1-score: combines precision and recall into one measure.
    #
    # Support: number of actual test examples in each class.
    print("\nDetailed Classification Report:", flush=True)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        ),
        flush=True
    )

    # --------------------------------------------------------
    # 7. SAVE THE MODEL AND MATCHING VECTORIZER
    # --------------------------------------------------------

    print("\n[6/6] Saving model files...", flush=True)

    # Save the trained classifier so app.py can load it later.
    joblib.dump(model, MODEL_PATH)

    # Save the exact vectorizer used with this classifier.
    # The app must use this matching vectorizer to transform
    # new feedback into the same feature format.
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nSUCCESS: Files saved successfully.", flush=True)
    print(f"Model file: {MODEL_PATH}", flush=True)
    print(f"Vectorizer file: {VECTORIZER_PATH}", flush=True)

    print("\nTo launch the Streamlit app, run:", flush=True)
    print("streamlit run app.py", flush=True)


# ------------------------------------------------------------
# 8. RUN THE SCRIPT
# ------------------------------------------------------------

# This calls main() only when this file is run directly.
# It prevents the training process from starting automatically
# if another Python file imports this module.
if __name__ == "__main__":
    try:
        main()

    # Ctrl+C interrupts the program. If that happens before saving,
    # the new model files may not have been written.
    except KeyboardInterrupt:
        print(
            "\nProcess interrupted. Check whether the model files "
            "were saved.",
            flush=True
        )

    # Display a readable error message and preserve the traceback
    # so a developer can diagnose the actual problem.
    except Exception as error:
        print(
            f"\nERROR: {type(error).__name__}: {error}",
            flush=True
        )
        raise
