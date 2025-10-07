import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

EXTRACT_FILE = Path("dataset.json")

def load_sentences_from_person2():
    with open(EXTRACT_FILE, "r", encoding="utf8") as f:
        data = json.load(f)
    sentences = []
    for item in data:
        drugs = " ".join(item.get("drugs", []))
        events = " ".join(item.get("events", []))
        combined = f"{item['sentence']} {drugs} {events}".strip()
        sentences.append(combined)
    return sentences

train_sentences = [
    "patient developed rash after aspirin",
    "subject took vitamin C",
    "he experienced nausea due to ibuprofen",
    "she is healthy and took paracetamol"
]
train_labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

clf = LogisticRegression()
clf.fit(X_train, train_labels)


def classify_sentence(text: str):
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    confidence = clf.predict_proba(X).max()
    return "ADE" if label == 1 else "Non-ADE", round(confidence, 3)


if __name__ == "__main__":
    sentences = load_sentences_from_person2()
    for s in sentences:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")

