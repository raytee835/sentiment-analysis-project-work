# =========================================================
# Emotion-Specific Word Cloud & Classification from Amazon Reviews
# Fixed Streamlit App Version
# Authors: Rowland Walker Mensah, Ricky Nelson Adzakpa, Rene Dwamena,
# Rebecca Boakyewaa Anderson, Raymond Tetteh
# =========================================================

import streamlit as st
import pandas as pd
import re
import spacy
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pickle
import os
from nltk.corpus import stopwords

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Amazon Review Emotion Predictor",
    page_icon="😊",
    layout="wide"
)


# ---------------------------
# Download required resources (with error handling)
# ---------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english')) - {"not", "no", "never"}
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return set()


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy English model not found. Please run: python -m spacy download en_core_web_sm")
        return None


# Initialize resources
stop_words = download_nltk_data()
nlp = load_spacy_model()


# ---------------------------
# Function: Clean text
# ---------------------------
def clean_text(text):
    """
    Lowercase, remove punctuation, numbers, and symbols.
    Lemmatize words and remove stopwords.
    """
    if nlp is None:
        # Fallback if spacy model not loaded
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = [word for word in text.split() if word not in stop_words and len(word) > 1]
        return " ".join(words)

    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 1]
    return " ".join(words)


# ---------------------------
# Function: Map review scores to emotions
# ---------------------------
def map_score_to_emotion(score):
    """
    Map numeric review scores to emotion labels.
    1-2 -> Negative
    3   -> Neutral
    4-5 -> Positive
    """
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"


# ---------------------------
# Function: Load or create models
# ---------------------------
@st.cache_resource
def load_or_create_models():
    """
    Load existing models or create new ones if data is available
    """
    try:
        # Try to load existing models
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("logistic_regression_model.pkl", "rb") as f:
            log_reg = pickle.load(f)
        with open("random_forest_model.pkl", "rb") as f:
            rf = pickle.load(f)

        st.success(" Pre-trained models loaded successfully!")
        return vectorizer, log_reg, rf, True

    except FileNotFoundError:
        # Try to create models if data file exists
        if os.path.exists("DATA/Reviews.csv"):
            st.info(" Pre-trained models not found. Training new models...")
            return train_new_models()
        else:
            st.error("Neither pre-trained models nor training data (DATA/Reviews.csv) found!")
            return None, None, None, False


def train_new_models():
    """
    Train new models from the dataset
    """
    try:
        # Load dataset
        df = pd.read_csv("DATA/Reviews.csv")
        df = df.sample(min(5000, len(df)), random_state=42).reset_index(drop=True)

        # Map scores to emotions
        df['Emotion'] = df['Score'].apply(map_score_to_emotion)

        # Clean text
        df['Cleaned_Text'] = df['Text'].apply(clean_text)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['Cleaned_Text'])
        y = df['Emotion']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train models
        log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
        log_reg.fit(X_train, y_train)

        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)

        # Save models
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open("logistic_regression_model.pkl", "wb") as f:
            pickle.dump(log_reg, f)
        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump(rf, f)

        st.success("New models trained and saved successfully!")
        return vectorizer, log_reg, rf, True

    except Exception as e:
        st.error(f"Error training models: {e}")
        return None, None, None, False


# ---------------------------
# Load models
# ---------------------------
vectorizer, log_reg, rf, models_loaded = load_or_create_models()

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("Amazon Review Emotion Predictor")
st.write("Enter a product review and our AI will predict whether the sentiment is Positive, Negative, or Neutral.")

if not models_loaded:
    st.error("""
    **Unable to load or create models!** 

    Please ensure you have:
    1. Pre-trained model files (*.pkl) in the same directory, OR
    2. Training data file at `DATA/Reviews.csv`

    To get the Amazon Reviews dataset:
    - Download from: https://www.kaggle.com/snap/amazon-fine-food-reviews
    - Place the `Reviews.csv` file in a `DATA/` folder
    """)
    st.stop()

# Add sidebar with model information
st.sidebar.header("Model Information")
st.sidebar.write("""
**Available Models:**
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Ensemble method for better accuracy

**Emotion Categories:**
- **Positive**: Scores 4-5 ⭐⭐⭐⭐⭐
- **Neutral**: Score 3 ⭐⭐⭐
- **Negative**: Scores 1-2 ⭐⭐
""")

# Model selection
model_choice = st.selectbox("Choose a prediction model:", ["Logistic Regression", "Random Forest"])

# User review input
user_review = st.text_area(
    "Enter your review here:",
    height=150,
    placeholder="e.g., This product exceeded my expectations! Great quality and fast delivery."
)

# Predict button
if st.button("Predict Emotion", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        try:
            # Clean the input text
            cleaned = clean_text(user_review)

            if not cleaned.strip():
                st.warning(" The review appears to contain no meaningful words after cleaning.")
            else:
                # Vectorize the input
                vect_input = vectorizer.transform([cleaned])

                # Make prediction
                if model_choice == "Logistic Regression":
                    prediction = log_reg.predict(vect_input)[0]
                    probabilities = log_reg.predict_proba(vect_input)[0]
                else:
                    prediction = rf.predict(vect_input)[0]
                    probabilities = rf.predict_proba(vect_input)[0]

                # Get class labels
                if model_choice == "Logistic Regression":
                    classes = log_reg.classes_
                else:
                    classes = rf.classes_

                # Display results
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Main prediction
                    emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
                    st.subheader(f"Predicted Emotion: {emoji_map.get(prediction, '🤔')} **{prediction}**")

                    # Confidence scores
                    st.subheader("Confidence Scores:")
                    for class_name, prob in zip(classes, probabilities):
                        confidence = prob * 100
                        st.write(f"**{class_name}**: {confidence:.1f}%")
                        st.progress(prob)

                with col2:
                    # Create a simple bar chart of probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#ff6b6b' if c == 'Negative' else '#51cf66' if c == 'Positive' else '#74c0fc' for c in
                              classes]
                    bars = ax.bar(classes, probabilities, color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Emotion Prediction Probabilities')
                    ax.set_ylim(0, 1)

                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{prob:.3f}', ha='center', va='bottom')

                    plt.tight_layout()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Add sample reviews for testing
st.subheader("Try These Sample Reviews:")

sample_reviews = {
    "Positive": "This product is absolutely amazing! Exceeded all my expectations and arrived quickly.",
    "Negative": "Terrible quality, broke after one day. Complete waste of money. Very disappointed.",
    "Neutral": "It's okay, does what it's supposed to do. Nothing special but gets the job done."
}

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Try Positive Sample"):
        st.text_area("Sample Review:", sample_reviews["Positive"], key="pos_sample")

with col2:
    if st.button("Try Negative Sample"):
        st.text_area("Sample Review:", sample_reviews["Negative"], key="neg_sample")

with col3:
    if st.button("Try Neutral Sample"):
        st.text_area("Sample Review:", sample_reviews["Neutral"], key="neu_sample")

# Footer
st.markdown("---")
st.markdown("""
**Created by:** Rowland Walker Mensah, Ricky Nelson Adzakpa, Rene Dwamena, Rebecca Boakyewaa Anderson, Raymond Tetteh

**Note:** This model was trained on Amazon product reviews and works best with product-related text.
""")