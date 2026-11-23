"""
Evaluation of 10 Hinglish Test Cases on Model Checkpoints
"""
import sys
import joblib
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_PATH = PROJECT_ROOT / "model_training" / "checkpoints" / "classical" / "tf_idf_plus_logistic_regression.joblib"

pipe = joblib.load(CKPT_PATH)

test_cases = [
    ("Code bina error ke run ho gaya! 🔥", "Positive"),
    ("Finally assignment complete ho gaya! 😍", "Positive"),
    ("Aaj exam bahut achha gaya!", "Positive"),
    ("Teacher ne bola well done beta! 🎉", "Positive"),
    ("Kal exam hai.", "Neutral"),
    ("Code phir se error de raha hai 😭", "Negative"),
    ("Project successfully complete kar diya!", "Positive"),
    ("Mera result kharab aa gaya.", "Negative"),
    ("Aaj kuch khaas nahi hua.", "Context-dependent"),
    ("Assignment abhi tak complete nahi hua.", "Negative")
]

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"{'Input':42s} | {'Expected':16s} | {'Predicted':10s} | {'Probabilities (Neg, Neu, Pos)'}")
print("-" * 95)
for text, expected in test_cases:
    probs = pipe.predict_proba([text])[0]
    pred = pipe.predict([text])[0]
    pred_label = LABEL_MAP[pred]
    prob_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
    match = "MATCH" if (pred_label == expected or expected == "Context-dependent") else "FAIL"
    print(f"{text:42s} | {expected:16s} | {pred_label:10s} | {prob_str} | {match}")
