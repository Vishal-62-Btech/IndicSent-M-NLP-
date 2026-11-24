import joblib
from pathlib import Path
from clean_text import clean_ocr_text

root = Path(r"D:\projectsss\NLP project (sentiment analysis)")
ckpt_path = root / "model_training" / "checkpoints" / "classical" / "tf_idf_plus_logistic_regression.joblib"
model = joblib.load(ckpt_path)

pipe_tfidf = model.named_steps["tfidf"]
vocab = pipe_tfidf.vocabulary_

check_tokens = ["emoji_crying", "nahi_ho", "crying", "nahi", "complete", "complete_nahi", "nahi_hua"]
for t in check_tokens:
    print(f"Token '{t}' in vocab: {t in vocab}")
    if t in vocab:
        idx = vocab[t]
        coefs = model.named_steps["clf"].coef_[:, idx]
        print(f"   -> Neg={coefs[0]:+.3f}, Neu={coefs[1]:+.3f}, Pos={coefs[2]:+.3f}")
