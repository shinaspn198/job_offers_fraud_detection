import re
import pickle
import os

HERE = os.path.dirname(__file__)

meta = pickle.load(
    open(os.path.join(HERE, "models", "meta.pkl"), "rb")
)

FRAUD_WORDS = meta["fraud_words"]


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z0-9\s%$â‚¹]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def hand_features(raw, clean):
    return [
        # 1. General fraud keyword count
        sum(1 for w in FRAUD_WORDS if w in clean),

        # 2. Uppercase word count
        sum(1 for w in raw.split() if w.isupper() and len(w) > 1),

        # 3. Exclamation count
        raw.count("!"),

        # 4. WhatsApp / Telegram
        int("whatsapp" in clean or "telegram" in clean),

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
        int(bool(
            re.search(
                r"(salary|stipend|earn)\s*(rs|inr|\$|â‚¹)?\s*\d{4,}",
                clean
            )
        )),

        # 11. Urgency pattern
        int(bool(
            re.search(
                r"urgent|immediately|hurry|act now",
                clean
            )
        )),
    ]


def matched_fraud_words(clean_text_):
    return [w for w in FRAUD_WORDS if w in clean_text_]