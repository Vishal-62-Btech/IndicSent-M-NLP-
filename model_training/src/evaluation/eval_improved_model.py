import sys, os, re, unicodedata
from pathlib import Path
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(r"D:\projectsss\NLP project (sentiment analysis)")
train_df = pd.read_csv(project_root / "model_training" / "data_splits" / "train.csv")
val_df = pd.read_csv(project_root / "model_training" / "data_splits" / "validation.csv")

EMOJI_MAP = {
    '🔥': ' emoji_fire ',
    '😍': ' emoji_heart_eyes ',
    '🎉': ' emoji_party ',
    '😭': ' emoji_crying ',
    '😂': ' emoji_laugh ',
    '🤣': ' emoji_laugh ',
    '❤️': ' emoji_heart ',
    '😡': ' emoji_angry ',
    '😴': ' emoji_sleep ',
    '⏰': ' emoji_clock ',
    '💯': ' emoji_hundred ',
    '✨': ' emoji_sparkles ',
    '👍': ' emoji_thumbsup ',
    '👎': ' emoji_thumbsdown '
}

HIND_NEG_PHRASES = [
    (r'\bbina\s+(\w+)', r'bina_\1'),
    (r'(\w+)\s+nahi\s+hua\b', r'\1_nahi_hua'),
    (r'(\w+)\s+nhi\s+hua\b', r'\1_nahi_hua'),
    (r'(\w+)\s+nahi\s+hai\b', r'\1_nahi_hai'),
    (r'(\w+)\s+nahi\s+gaya\b', r'\1_nahi_gaya'),
    (r'\bnahi\s+(\w+)', r'nahi_\1'),
    (r'\bnhi\s+(\w+)', r'nahi_\1'),
    (r'\bmat\s+(\w+)', r'mat_\1')
]

def preprocess_text(text):
    if text is None: return '[NO_TEXT]'
    s = str(text)
    if not s or s.lower() in ['nan', 'none', 'null']: return '[NO_TEXT]'
    s = unicodedata.normalize('NFKC', s)
    for em, tok in EMOJI_MAP.items():
        s = s.replace(em, tok)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'!{2,}', ' ! ', s)
    s = re.sub(r'\?{2,}', ' ? ', s)
    for pattern, repl in HIND_NEG_PHRASES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else '[NO_TEXT]'

X_tr = [preprocess_text(t) for t in train_df['ocr']]
y_tr = train_df['label_id'].values
X_va = [preprocess_text(t) for t in val_df['ocr']]
y_val = val_df['label_id'].values

union = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b|!\b|\?\b')),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=18000, sublinear_tf=True))
])
pipe = Pipeline([
    ('feats', union),
    ('clf', LogisticRegression(C=0.4, class_weight={0: 1.15, 1: 0.85, 2: 1.15}, max_iter=1000, random_state=42))
])
pipe.fit(X_tr, y_tr)

# Validation set metrics
val_preds = pipe.predict(X_va)
acc = accuracy_score(y_val, val_preds) * 100
macro_f1 = f1_score(y_val, val_preds, average='macro') * 100
print(f"Validation Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.2f}%")
print(classification_report(y_val, val_preds, target_names=['Negative', 'Neutral', 'Positive']))

hinglish_cases = [
    ('Code bina error ke run ho gaya! 🔥', 'Positive', 2),
    ('Finally assignment complete ho gaya! 😍', 'Positive', 2),
    ('Aaj exam bahut achha gaya!', 'Positive', 2),
    ('Teacher ne bola well done beta! 🎉', 'Positive', 2),
    ('Kal exam hai.', 'Neutral', 1),
    ('Code phir se error de raha hai 😭', 'Negative', 0),
    ('Project successfully complete kar diya!', 'Positive', 2),
    ('Mera result kharab aa gaya.', 'Negative', 0),
    ('Aaj kuch khaas nahi hua.', 'Context-dependent', 1),
    ('Assignment abhi tak complete nahi hua.', 'Negative', 0)
]

labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
print("=" * 80)
print(f"{'Hinglish Input':<42} | {'Expected':<18} | {'Predicted':<10} | {'Probabilities [Neg/Neu/Pos]'}")
print("=" * 80)
for txt, exp_str, exp_id in hinglish_cases:
    pr = pipe.predict([preprocess_text(txt)])[0]
    pr_str = labels_map[pr]
    probs = pipe.predict_proba([preprocess_text(txt)])[0]
    probs_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
    status = "MATCH" if (pr == exp_id or (exp_str == "Context-dependent" and pr in [1, 2])) else "DIFF"
    print(f"{txt:<42} | {exp_str:<18} | {pr_str:<10} | {probs_str} [{status}]")
