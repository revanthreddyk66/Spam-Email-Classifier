# Spam-Email-Classifier
# Spam Email Classifier using Naive Bayes

This project demonstrates a classic machine learning approach to text classification. It uses a **Multinomial Naive Bayes** model to classify emails as either "spam" or "ham" (not spam) based on their text content.

---

## 🚀 Project Overview

* **Goal**: To build an effective model that can accurately identify spam emails.
* **Algorithm**: Multinomial Naive Bayes, a probabilistic algorithm that is highly effective for text classification tasks.
* **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert email text into numerical feature vectors that the model can understand.
* **Key Skills**: This project covers the complete machine learning workflow: data loading, text preprocessing, feature extraction, model training, evaluation, and prediction.

---

## 🛠️ Tech Stack

* **Language**: Python
* **Libraries**:
    * Pandas (for data manipulation)
    * NLTK (for text preprocessing)
    * Scikit-learn (for the ML model and vectorizer)

---

## ⚙️ How to Run

### Prerequisites

* Python 3 installed.
* The required libraries installed.
* A dataset file named `spam.csv`. A popular one can be downloaded from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-directory>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the script:**
    ```bash
    python spam_classifier.py
    ```
