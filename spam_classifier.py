`**
This is the single Python script containing all the code for the project.

```python
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# --- Download necessary NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- 1. Data Preprocessing Function ---
def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    # Remove all non-alphabetic characters and convert to lowercase
    processed_text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # Split text into words (tokenize)
    words = processed_text.split()
    
    # Remove stopwords and apply stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    
    return " ".join(words)

# --- 2. Load and Prepare Data ---
print("Loading dataset...")
# Note: Download 'spam.csv' from Kaggle: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
# It has columns 'v1' (label) and 'v2' (text)
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
except FileNotFoundError:
    print("\nError: 'spam.csv' not found.")
    print("Please download the dataset from [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)")
    print("and place it in the same directory as this script.")
    exit()

print("Preprocessing text data... (This may take a moment)")
df['processed_text'] = df['text'].apply(preprocess_text)

# --- 3. Feature Extraction (TF-IDF) ---
print("Extracting features with TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_text']).toarray()
y = df['label']

# --- 4. Split Data and Train Model ---
print("Splitting data and training the Naive Bayes model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# --- 6. Interactive Prediction ---
print("\n--- Interactive Spam Prediction ---")
print("Type an email/SMS message and press Enter to classify. Type 'quit' to exit.")

while True:
    user_input = input("\nEnter a message: ")
    if user_input.lower() == 'quit':
        break
        
    # Preprocess and transform the input
    processed_input = preprocess_text(user_input)
    vectorized_input = tfidf.transform([processed_input])
    
    # Predict
    prediction = model.predict(vectorized_input)[0]
    
    print(f"Prediction: This message is '{prediction.upper()}'")
