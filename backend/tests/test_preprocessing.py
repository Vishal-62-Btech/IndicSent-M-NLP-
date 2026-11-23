"""
IndicSent-M: Preprocessing Unit Tests
Validates lexical token preservation, negation compounding, emoji translation,
and Unicode normalization across positive, negative, and neutral Hinglish expressions.
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SRC_DIR = PROJECT_ROOT / "model_training" / "src"
if str(MODEL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC_DIR))

from preprocessing.clean_text import clean_ocr_text, HIND_NEG_PHRASES, HINGLISH_COMPOUND_PHRASES, EMOJI_SENTIMENT_MAP


def test_positive_expression_preprocessing():
    """Verify that celebration emojis and praise compound idioms are properly preserved."""
    raw = "Assignment Submitted. Well done beta! Shabash! 🎉 🥳"
    cleaned = clean_ocr_text(raw)
    assert "well_done_beta" in cleaned
    assert "Shabash" in cleaned
    assert "emoji_party" in cleaned
    assert "emoji_partying_face" in cleaned
    assert cleaned == "Assignment Submitted. well_done_beta! Shabash! emoji_party emoji_partying_face"


def test_negative_negation_preprocessing():
    """Verify that negation structures are bound without affecting unrelated words."""
    raw = "Assignment abhi tak complete nahi hua."
    cleaned = clean_ocr_text(raw)
    assert "complete_nahi_hua" in cleaned
    assert cleaned == "Assignment abhi tak complete_nahi_hua."


def test_bina_negation_compound():
    """Verify 'bina <word>' compound binding."""
    raw = "Code bina error ke run ho gaya! 🔥"
    cleaned = clean_ocr_text(raw)
    assert "bina_error" in cleaned
    assert "emoji_fire" in cleaned


def test_neutral_everyday_preprocessing():
    """Verify that neutral everyday Hinglish text remains unchanged and natural."""
    raw = "Kal subah normal train se college jaana hai."
    cleaned = clean_ocr_text(raw)
    assert cleaned == "Kal subah normal train se college jaana hai."


def test_empty_and_whitespace_handling():
    """Verify edge case handling of whitespace, nan, and empty strings."""
    assert clean_ocr_text("   ") == "[NO_TEXT]"
    assert clean_ocr_text("") == "[NO_TEXT]"
    assert clean_ocr_text("nan") == "[NO_TEXT]"
    assert clean_ocr_text(None) == "[NO_TEXT]"


def test_separation_of_negation_and_positive_idioms():
    """Verify that HIND_NEG_PHRASES contains zero positive expressions."""
    neg_patterns = [p for p, _ in HIND_NEG_PHRASES]
    for pattern in neg_patterns:
        assert "well" not in pattern
        assert "badhai" not in pattern
        assert "mubarak" not in pattern

    compound_patterns = [p for p, _ in HINGLISH_COMPOUND_PHRASES]
    assert any("well" in p for p in compound_patterns)
    assert any("badhai" in p for p in compound_patterns)
