import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) > 10
    ]


def retrieve_top_sentences(question, doc_text, k=6):
    sentences = split_sentences(doc_text)

    q_words = set(
        re.findall(r"[a-z]{3,}", question.lower())
    )

    scored = []

    for s in sentences:
        s_words = set(
            re.findall(r"[a-z]{3,}", s.lower())
        )

        score = len(q_words & s_words)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = [
        s for score, s in scored[:k]
        if score > 0
    ] or sentences[:k]

    return top


def rag_answer(question, document_text):

    if not document_text:
        return "No document loaded yet."

    context_sentences = retrieve_top_sentences(
        question,
        document_text
    )

    context = " ".join(context_sentences)

    if not API_KEY:
        return (
            "(no LLM key set — showing best-matching excerpt)\n\n"
            + context
        )

    from google import genai

    client = genai.Client(api_key=API_KEY)

    prompt = f"""You are a document assistant helping a user understand a job/internship offer.

Answer the question using ONLY the context below.
If the answer is not found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return resp.text.strip()