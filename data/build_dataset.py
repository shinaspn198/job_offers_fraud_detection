"""
Build a CLASS-BALANCED combined_dataset.csv (equal fraud/legit count)
Run: python build_dataset.py
"""
import pandas as pd
import re
import os

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "combined_dataset.csv")

def strip_html(t):
    t = re.sub(r"<[^>]+>", " ", str(t))
    return re.sub(r"\s+", " ", t).strip()

rows = []

# 1. New 10k fraud-only dataset -> sample as needed later (cap big first)
p = f"{RAW}/fake_job_postings.csv"
if os.path.exists(p):
    df = pd.read_csv(p).fillna("")
    df["text"] = (df["title"] + ". " + df["company_profile"] + " " +
                  df["description"] + " " + df["requirements"] + " " + df["benefits"]).apply(strip_html)
    df = df[df["text"].str.len() > 30].drop_duplicates(subset=["text"])
    for t in df["text"]:
        rows.append((t, 1))

# 2. dataset_500.xlsx -> both classes
p = f"{RAW}/dataset_500.xlsx"
if os.path.exists(p):
    d = pd.read_excel(p)
    for _, r in d.iterrows():
        rows.append((str(r["text"]), int(r["fraudulent"])))

# 3. dataset_new.xlsx -> both classes
p = f"{RAW}/dataset_new.xlsx"
if os.path.exists(p):
    d = pd.read_excel(p)
    for _, r in d.iterrows():
        if pd.notna(r["label"]):
            rows.append((str(r["text"]), int(r["label"])))

# 4. internship_fraud_dataset.csv -> both classes
p = f"{RAW}/internship_fraud_dataset.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        rows.append((str(r["job_text"]), int(r["label"])))

# 5. old EMSCAD 132-row slice (mostly legit, real text) -> rename to emscad_132.csv first
p = f"{RAW}/emscad_132.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        desc = str(r.get("description", ""))
        if len(desc) > 30:
            rows.append((desc, int(r["fraudulent"])))

df = pd.DataFrame(rows, columns=["text", "label"])
df = df.dropna(subset=["text"])
df = df[df["text"].str.len() > 15]
df = df.drop_duplicates(subset=["text"])

# --- BALANCE: undersample majority class to match minority ---
counts = df["label"].value_counts()
print("Before balance:", dict(counts))

min_count = counts.min()
balanced = pd.concat([
    df[df["label"] == 0].sample(n=min_count, random_state=42) if counts[0] > min_count else df[df["label"] == 0],
    df[df["label"] == 1].sample(n=min_count, random_state=42) if counts[1] > min_count else df[df["label"] == 1],
])
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

balanced.to_csv(OUT, index=False)
print("Saved:", OUT, "| rows:", len(balanced))
print("After balance:", balanced["label"].value_counts().to_dict())