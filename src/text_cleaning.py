"""
text_cleaning.py

This module provides text preprocessing utilities for SMS messages.
It defines a function to clean and lemmatize text, and a scikit-learn
compatible transformer for use in pipelines.
"""
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_special_chars(text):
    """
    Clean and preprocess a single text string.

    Steps:
        1. Convert to lowercase.
        2. Tokenize text into words.
        3. Remove English stopwords.
        4. Lemmatize remaining tokens.
        5. Join tokens back into a single string.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and lemmatized text.
    """
    
    if text is None:
        return ""
    text = str(text).lower()
    tokens = word_tokenize(text)                    # keeps punctuation/numbers as separate tokens
    filtered = [w for w in tokens if w not in stop_words]
    lemmas = [lemmatizer.lemmatize(w, pos="v") for w in filtered]
    return " ".join(lemmas)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(clean_special_chars)