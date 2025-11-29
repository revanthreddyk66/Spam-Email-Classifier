# spam_classifier.py
import re
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# ---------------------------
# 0. Ensure NLTK stopwords
# ---------------------------
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# ---------------------------
# 1. Globals: stemmer & stopwords (create once)
# ---------------------------
PS = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

# ---------------------------
# 2. Text preprocessing
# ---------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # keep letters, replace others with space, lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = text.split()
    # remove stopwords and stem
    tokens = [PS.stem(tok) for tok in tokens if tok not in STOP_WORDS]
    return " ".join(tokens)

# ---------------------------
# 3. Load dataset
# ---------------------------
def load_data(path='spam.csv'):
    try:
        df = pd.read_csv(path, encoding='latin-1')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found in current directory.")
        print("Download the SMS Spam Collection dataset and place 'spam.csv' here.")
        sys.exit(1)

    # Keep only the expected columns (v1=label, v2=text) for this dataset
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
    else:
        # Try common alternate names
        if 'label' in df.columns and 'text' in df.columns:
            df = df[['label', 'text']]
        else:
            raise ValueError("Unexpected CSV format. Expect columns 'v1'/'v2' or 'label'/'text'.")

    df = df.dropna(subset=['text'])
    return df

# ---------------------------
# 4. Main training & eval
# ---------------------------
def train_and_evaluate(df, tfidf_max_features=3000, test_size=0.2, random_state=42):
    # Preprocess text column
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Feature extraction (TF-IDF). Keep sparse matrix (no .toarray()).
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']  # 'ham' / 'spam'

    # Print dataset stats
    counts = y.value_counts()
    print("Dataset size:", len(df))
    print("Class distribution:\n", counts.to_dict())

    # Split with stratify to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Model: Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict & evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=['ham', 'spam']))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Cross-validated scores for more robust estimate
    print("\nRunning 5-fold cross-validation (accuracy & f1_macro)...")
    cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    print(f"CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"CV F1-macro: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Return trained objects for saving or interactive use
    return tfidf, model

# ---------------------------
# 5. Save / load helpers
# ---------------------------
def save_artifacts(tfidf, model, tfidf_path='tfidf.joblib', model_path='nb_model.joblib'):
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(model, model_path)
    print(f"Saved TF-IDF to {tfidf_path} and model to {model_path}")

def load_artifacts(tfidf_path='tfidf.joblib', model_path='nb_model.joblib'):
    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)
    return tfidf, model

# ---------------------------
# 6. Interactive prediction loop
# ---------------------------
def interactive_predict(tfidf, model):
    print("\n--- Interactive Spam Prediction ---")
    print("Type a message and press Enter. Type 'quit' to exit.")
    while True:
        msg = input("\nEnter message: ")
        if msg.strip().lower() in ('quit', 'exit'):
            break
        proc = preprocess_text(msg)
        vec = tfidf.transform([proc])       # keep as sparse
        pred = model.predict(vec)[0]
        # If you want probability: prob = model.predict_proba(vec)
        print(f"Prediction: {pred.upper()}")

# ---------------------------
# main
# ---------------------------
if __name__ == '__main__':
    df = load_data('spam.csv')
    tfidf, model = train_and_evaluate(df)

    # Optionally save artifacts for later reuse
    save_artifacts(tfidf, model)

    # Start interactive loop
    interactive_predict(tfidf, model)
