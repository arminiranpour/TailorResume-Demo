from app.scoring_normalize import generate_ngrams, normalize_text, normalize_tokens, tokenize


def extract_signals(text: str) -> dict:
    normalized = normalize_text(text)
    tokens = tokenize(text)
    phrases = generate_ngrams(tokens, 3)
    return {"normalized": normalized, "tokens": tokens, "phrases": phrases}
