"""
IndicSent-M: Classical ML Models Module
Implements TF-IDF vectorization and classical classifiers:
1. Majority (Dummy) Baseline
2. TF-IDF + Multinomial Naive Bayes
3. TF-IDF + Logistic Regression (Tuned C, Balanced Weights)
4. TF-IDF + Linear SVM (Calibrated LinearSVC)
5. TF-IDF + Random Forest
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

def get_tfidf_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """
    Returns optimal TF-IDF vectorizer configuration for Hinglish / Hindi-English text.
    Uses unigram + bigram features with sublinear term-frequency scaling.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        max_features=max_features,
        strip_accents=None  # Preserve Devanagari Unicode accents
    )

def build_classical_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Builds the suite of classical ML pipelines.
    Vectorizer is fitted ONLY on the training split during training.
    """
    pipelines = {
        "Majority Baseline": Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "TF-IDF + Multinomial Naive Bayes": Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs"
            ))
        ]),
        "TF-IDF + Linear SVM": Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                    dual="auto",
                    max_iter=2000
                ),
                method="sigmoid",
                cv=3
            ))
        ]),
        "TF-IDF + Random Forest": Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=35,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    }
    return pipelines
