import os, pickle
from scipy.sparse import hstack, csr_matrix
from .features import clean_text, hand_features

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "models")

vectorizer = pickle.load(open(os.path.join(MODELS, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(MODELS, "model.pkl"), "rb"))

def transform(texts):
    cleaned = [clean_text(t) for t in texts]
    tfidf = vectorizer.transform(cleaned)
    extra = [hand_features(r, c) for r, c in zip(texts, cleaned)]
    return hstack([tfidf, csr_matrix(extra)])

def predict(text):
    X = transform([text])
    proba = model.predict_proba(X)[0]  # [legit_prob, fraud_prob]
    fraud_score = round(float(proba[1]) * 100, 1)
    legit_score = round(float(proba[0]) * 100, 1)
    if fraud_score >= 70:
        verdict = "Fraud"
    elif fraud_score >= 45:
        verdict = "Suspicious"
    else:
        verdict = "Legit"
    return verdict, fraud_score, legit_score

def predict_proba_fn(texts):
    return model.predict_proba(transform(texts))