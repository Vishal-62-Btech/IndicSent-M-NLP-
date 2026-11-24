import joblib
from pathlib import Path
from clean_text import clean_ocr_text

root = Path(r"D:\projectsss\NLP project (sentiment analysis)")
ckpt_path = root / "model_training" / "checkpoints" / "classical" / "tf_idf_plus_logistic_regression.joblib"
model = joblib.load(ckpt_path)

pipe_tfidf = model.named_steps["tfidf"]
pipe_clf = model.named_steps["clf"]

text = "Padhai chhod ke sab ho raha hai life me, bas syllabus complete nahi ho raha 😭"
cleaned = clean_ocr_text(text)
print("Cleaned text:", cleaned)

vec = pipe_tfidf.transform([cleaned])
features = pipe_tfidf.get_feature_names_out()

non_zero_idx = vec.nonzero()[1]
print("\nActive TF-IDF features and their weights:")
for idx in non_zero_idx:
    word = features[idx]
    tfidf_val = vec[0, idx]
    # Check logistic regression coefficients for classes [0: Neg, 1: Neu, 2: Pos]
    coefs = pipe_clf.coef_[:, idx]
    print(f"  {word:20s} (tfidf={tfidf_val:.3f}) -> Neg={coefs[0]:+.3f}, Neu={coefs[1]:+.3f}, Pos={coefs[2]:+.3f}")

probs = model.predict_proba([cleaned])[0]
print("\nFinal Probs:", probs)
