# backend/classify.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parents[1]
EXTRACT_FILE = BASE / "results" / "extractions.json"


# ---- STEP 1: Load sentences from extraction
def load_sentences_from_extraction():
    with open(EXTRACT_FILE, "r", encoding="utf8") as f:
        data = json.load(f)
    # Combine sentence with its extracted drugs/events for richer features
    sentences = []
    for item in data:
        drugs = " ".join(item.get("drugs", []))
        events = " ".join(item.get("events", []))
        combined = f"{item['sentence']} {drugs} {events}".strip()
        sentences.append(combined)
    return sentences


# ---- STEP 2: Demo training data ----
train_sentences = [
    "patient developed rash after aspirin",
    "subject took vitamin C",
    "he experienced nausea due to ibuprofen",
    "she is healthy and took paracetamol"
]
train_labels = [1, 0, 1, 0]   # 1 = ADE, 0 = Non-ADE


# ---- STEP 3: Train classifier ----
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

clf = LogisticRegression()
clf.fit(X_train, train_labels)


def classify_sentence(text: str):
    """Classify a sentence as ADE or Non-ADE"""
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    confidence = clf.predict_proba(X).max()
    return "ADE" if label == 1 else "Non-ADE", round(confidence, 3)


if __name__ == "__main__":
    sentences = load_sentences_from_extraction()
    for s in sentences:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")
