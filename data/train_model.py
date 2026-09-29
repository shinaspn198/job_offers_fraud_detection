import re
import os
import pickle

import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


HERE = os.path.dirname(__file__)

OUT_DIR = os.path.join(
    HERE,
    "..",
    "app",
    "models"
)

os.makedirs(OUT_DIR, exist_ok=True)


FRAUD_WORDS = [
    "registration fee",
    "processing fee",
    "security deposit",
    "pay to join",
    "whatsapp only",
    "telegram only",
    "no interview",
    "guaranteed job",
    "guaranteed salary",
    "urgent hiring",
    "limited seats",
    "instant offer",
    "send money",
    "bank details",
    "advance payment",
    "easy money",
    "work from home earn",
    "no experience needed high salary",
]


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z0-9\s%$₹]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def hand_features(raw, clean):
    return [
        # 1. General fraud keyword count
        sum(1 for w in FRAUD_WORDS if w in clean),

        # 2. Uppercase word count
        sum(
            1 for w in raw.split()
            if w.isupper() and len(w) > 1
        ),

        # 3. Exclamation count
        raw.count("!"),

        # 4. WhatsApp / Telegram
        int(
            "whatsapp" in clean
            or "telegram" in clean
        ),

        # 5. Registration fee
        int("registration fee" in clean),

        # 6. Processing fee
        int("processing fee" in clean),

        # 7. Security deposit
        int("security deposit" in clean),

        # 8. Guaranteed job / placement
        int(
            "guaranteed job" in clean
            or "guaranteed placement" in clean
        ),

        # 9. Bank / financial information request
        int(
            "bank details" in clean
            or "account details" in clean
            or "bank account" in clean
        ),

        # 10. High salary / earning pattern
        int(
            bool(
                re.search(
                    r"(salary|stipend|earn)\s*(rs|inr|\$|₹)?\s*\d{4,}",
                    clean
                )
            )
        ),

        # 11. Urgency pattern
        int(
            bool(
                re.search(
                    r"urgent|immediately|hurry|act now",
                    clean
                )
            )
        ),
    ]


def build_features(texts, vectorizer, fit=False):

    cleaned = [
        clean_text(t)
        for t in texts
    ]

    if fit:
        tfidf = vectorizer.fit_transform(cleaned)
    else:
        tfidf = vectorizer.transform(cleaned)

    extra = [
        hand_features(raw, clean)
        for raw, clean in zip(texts, cleaned)
    ]

    return hstack([
        tfidf,
        csr_matrix(extra)
    ])


def main():

    # Load dataset
    dataset_path = os.path.join(
        HERE,
        "combined_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X_text = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()

    # Train/test split
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=2
    )

    Xtr = build_features(
        Xtr_txt,
        vectorizer,
        fit=True
    )

    Xte = build_features(
        Xte_txt,
        vectorizer,
        fit=False
    )

    print("\nTraining features:", Xtr.shape)
    print("Testing features:", Xte.shape)

    # Model
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0
    )

    model.fit(Xtr, ytr)

    # Evaluation
    preds = model.predict(Xte)

    print("\nClassification Report:")
    print(
        classification_report(
            yte,
            preds,
            target_names=["Legit", "Fraud"]
        )
    )

    print(
        "F1 (macro):",
        f1_score(
            yte,
            preds,
            average="macro"
        )
    )

    # Save model
    pickle.dump(
        vectorizer,
        open(
            os.path.join(
                OUT_DIR,
                "vectorizer.pkl"
            ),
            "wb"
        )
    )

    pickle.dump(
        model,
        open(
            os.path.join(
                OUT_DIR,
                "model.pkl"
            ),
            "wb"
        )
    )

    pickle.dump(
        {
            "fraud_words": FRAUD_WORDS
        },
        open(
            os.path.join(
                OUT_DIR,
                "meta.pkl"
            ),
            "wb"
        )
    )

    print("\nModel files saved successfully.")
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()