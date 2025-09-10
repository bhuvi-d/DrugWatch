# backend/classify.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Note: The original EXTRACT_FILE path has been changed to use the uploaded JSON file.
# Since the original file 'extractions.json' was not provided, we will use 'drug_to_effects.json'.
EXTRACT_FILE = Path("drug_to_effects.json")


# ---- STEP 1: Load sentences from the JSON file ----
def load_sentences_from_person2():
    sentences = []
    # The uploaded file has a dictionary structure, not a list of dictionaries.
    # We will extract all the effects listed in the values.
    with open(EXTRACT_FILE, "r", encoding="utf8") as f:
        data = json.load(f)
    for effects_list in data.values():
        for sentence in effects_list:
            # Filter out any empty strings that may be present in the lists.
            if sentence.strip():
                sentences.append(sentence.strip())
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


# ---- STEP 4: Example usage ----
if __name__ == "__main__":
    sentences = load_sentences_from_person2()
    # For demonstration, we'll only classify the first 10 sentences
    # to provide a concise output.
    for s in sentences[:10]:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")
    
