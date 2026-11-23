"""
IndicSent-M: Text Preprocessing Module
Provides reproducible, linguistically sensitive cleaning for Hinglish/Hindi-English meme OCR text.
Preserves Hinglish slang, Hindi Devanagari script, emotional punctuation, and handles missing OCR.
"""

import sys
import re
import unicodedata

# Common sentiment-bearing emoji mapping to linguistic tokens
EMOJI_SENTIMENT_MAP = {
    '🔥': ' emoji_fire ',
    '😍': ' emoji_heart_eyes ',
    '🎉': ' emoji_party ',
    '🥳': ' emoji_partying_face ',
    '🎊': ' emoji_confetti ',
    '👏': ' emoji_applause ',
    '🙌': ' emoji_celebration ',
    '🏆': ' emoji_trophy ',
    '😎': ' emoji_cool ',
    '💪': ' emoji_strong ',
    '🎯': ' emoji_target ',
    '🤩': ' emoji_starstruck ',
    '😁': ' emoji_grin ',
    '😄': ' emoji_smile ',
    '😃': ' emoji_joy ',
    '🌟': ' emoji_star ',
    '⭐': ' emoji_star ',
    '😭': ' emoji_crying ',
    '😂': ' emoji_laugh ',
    '🤣': ' emoji_rofl ',
    '❤️': ' emoji_heart ',
    '💔': ' emoji_broken_heart ',
    '😡': ' emoji_angry ',
    '😤': ' emoji_frustrated ',
    '😢': ' emoji_sad ',
    '😴': ' emoji_sleep ',
    '⏰': ' emoji_clock ',
    '💯': ' emoji_hundred ',
    '✨': ' emoji_sparkles ',
    '👍': ' emoji_thumbsup ',
    '👎': ' emoji_thumbsdown ',
    '🙏': ' emoji_namaste '
}

# Strict Hinglish negation structures (syntactic negation compounds)
HIND_NEG_PHRASES = [
    (r'\bbina\s+(\w+)', r'bina_\1'),
    (r'(\w+)\s+nahi\s+hua\b', r'\1_nahi_hua'),
    (r'(\w+)\s+nhi\s+hua\b', r'\1_nahi_hua'),
    (r'(\w+)\s+nahi\s+hai\b', r'\1_nahi_hai'),
    (r'(\w+)\s+nhi\s+hai\b', r'\1_nahi_hai'),
    (r'(\w+)\s+nahi\s+gaya\b', r'\1_nahi_gaya'),
    (r'(\w+)\s+nhi\s+gaya\b', r'\1_nahi_gaya'),
    (r'\bnahi\s+(\w+)', r'nahi_\1'),
    (r'\bnhi\s+(\w+)', r'nahi_\1'),
    (r'\bmat\s+(\w+)', r'mat_\1')
]

# Multi-word Hinglish compound expressions (lexical binding to preserve phrases without forcing sentiment)
HINGLISH_COMPOUND_PHRASES = [
    (r'\bwell\s+done\s+beta\b', r'well_done_beta'),
    (r'\bwell\s+done\b', r'well_done'),
    (r'\bbadhai\s+ho\b', r'badhai_ho'),
    (r'\bmubarak\s+ho\b', r'mubarak_ho'),
    (r'\bkya\s+baat\s+hai\b', r'kya_baat_hai')
]

def clean_ocr_text(text: str, placeholder_for_empty: str = "[NO_TEXT]") -> str:
    """
    Cleans raw OCR text from memes while preserving sentiment tokens.
    
    Rules:
    - Normalizes Unicode using NFKC.
    - Translates expressive emojis to sentiment tokens (e.g. emoji_fire, emoji_party).
    - Preserves Hinglish negation compounds (bina_error, nahi_hua).
    - Strips URLs and meme generator watermarks.
    - Normalizes whitespace and newlines.
    - Preserves Devanagari (Hindi) and Latin (Hinglish/English) characters.
    - Retains expressive punctuation (exclamation and question marks).
    - If text is empty or whitespace-only, assigns placeholder_for_empty.
    """
    if text is None:
        return placeholder_for_empty
    
    s = str(text)
    if not s or s.lower() in ["nan", "none", "null"]:
        return placeholder_for_empty
    
    # Unicode NFKC normalization
    s = unicodedata.normalize("NFKC", s)
    
    # Translate expressive emojis
    for em, tok in EMOJI_SENTIMENT_MAP.items():
        s = s.replace(em, tok)
        
    # Remove URLs
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    
    # Remove common meme watermark domains
    s = re.sub(r"\b(?:memecenter|imgflip|memegenerator|reddit|instagram|fb)\.(?:com|net|org)\b", " ", s, flags=re.IGNORECASE)
    
    # Replace carriage returns and newlines with spaces
    s = re.sub(r"[\r\n\t]+", " ", s)
    
    # Collapse multiple consecutive punctuation marks to spaced single marks
    s = re.sub(r"!{2,}", " ! ", s)
    s = re.sub(r"\?{2,}", " ? ", s)
    
    # Bind multi-word Hinglish compound expressions
    for pattern, repl in HINGLISH_COMPOUND_PHRASES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # Bind Hinglish negation compounds
    for pattern, repl in HIND_NEG_PHRASES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    
    if len(s) == 0:
        return placeholder_for_empty
        
    return s

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    samples = [
        "Relationship status.\nSingle\nMarried\nIt's complicated https://memecenter.com",
        "  ",
        "nan",
        "Padhai chhod ke sab ho raha hai life me 😭",
        "ये बहुत बढ़िया मीम है भाई... http://test.com/img.jpg"
    ]
    for sample in samples:
        print(f"ORIGINAL: {repr(sample)} -> CLEAN: {repr(clean_ocr_text(sample))}")
