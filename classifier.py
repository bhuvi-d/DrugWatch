import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# STEP 1: Load sentences

def load_sentences_from_person1():
    with open("data/cleaned/cleaned_sentences.txt", "r") as f:
        sentences = [line.strip() for line in f.readlines()]
    return sentences

def load_sentences_from_person2():
    with open("results/extractions.json", "r") as f:
        data = json.load(f)
    # extract only the sentence text
    sentences = [item["sentence"] for item in data]
    return sentences


# ⚠️ Demo data:
train_sentences = [
    "patient developed rash after aspirin",
    "subject took vitamin C",
    "he experienced nausea due to ibuprofen",
    "she is healthy and took paracetamol"
]
train_labels = [1, 0, 1, 0]   # 1 = ADE, 0 = Non-ADE

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

clf = LogisticRegression()
clf.fit(X_train, train_labels)

def classify_sentence(text):
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    confidence = clf.predict_proba(X).max()
    return "ADE" if label == 1 else "Non-ADE", round(confidence, 3)

# STEP 4: Example usage

if __name__ == "__main__":
    #Person 1's cleaned sentences
    # sentences = load_sentences_from_person1()

    #use Person 2's NER sentences
    sentences = load_sentences_from_person2()

    for s in sentences:
        label, conf = classify_sentence(s)
        print(f"{s}  --> {label} (conf={conf})")
