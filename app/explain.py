import re
from .model import predict_proba_fn
from .features import clean_text, matched_fraud_words

def word_impact(text, top_k=10):
    """
    For each significant word, remove it and see how fraud probability shifts.
    Positive delta = word pushed toward fraud. Negative = pushed toward legit.
    """
    base_proba = predict_proba_fn([text])[0][1]  # base fraud prob
    words = list(dict.fromkeys(re.findall(r"[a-zA-Z]{3,}", text)))[:40]  # unique, capped for speed
    if not words:
        return []

    variants = []
    for w in words:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        variants.append(pattern.sub("", text, count=1))

    probs = predict_proba_fn(variants)[:, 1]
    impacts = []
    for w, p in zip(words, probs):
        delta = base_proba - p  # if removing word drops fraud prob, word was pushing fraud up
        impacts.append((w, round(float(delta), 4)))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts[:top_k]

def get_reasons(text, verdict):
    clean = clean_text(text)
    fraud_hits = matched_fraud_words(clean)
    patterns = {
        "payment request": r"(pay|payment|fee|deposit|amount|transfer|charges)\s*(₹|\$|inr|usd)?\s*\d+",
        "no interview claim": r"no\s+interview",
        "direct selection": r"direct\s+selection|direct\s+joining",
        "whatsapp/telegram only": r"whatsapp\s+only|telegram\s+only|contact\s+on\s+whatsapp",
        "guaranteed job/salary": r"guaranteed\s+(job|salary|income)",
        "urgent pressure": r"urgent|immediately|limited\s+seats|apply\s+immediately",
    }
    for label, pat in patterns.items():
        if re.search(pat, clean):
            fraud_hits.append(label)
    return list(dict.fromkeys(fraud_hits))