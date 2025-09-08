# backend/extract.py
import json
from pathlib import Path
import spacy

BASE = Path(__file__).resolve().parents[1]
CLEAN_FILE = BASE / "data" / "cleaned" / "sentences.json"
OUT_FILE = BASE / "results" / "extractions.json"

# Try SciSpacy biomedical models first (if installed) else fallback.
MODEL_CANDIDATES = ["en_ner_bionlp13cg_md", "en_ner_bc5cdr_md", "en_core_sci_sm"]
nlp = None
for m in MODEL_CANDIDATES:
    try:
        nlp = spacy.load(m)
        print("Using model:", m)
        break
    except Exception:
        nlp = None

if nlp is None:
    nlp = spacy.load("en_core_web_sm")
    print("Using fallback model: en_core_web_sm")

# small dictionary fallback to catch common drug names/events quickly
DRUG_DICT = {"aspirin", "paracetamol", "ibuprofen"}
EVENT_DICT = {"bleeding", "rash", "nausea", "headache", "pain"}

def map_ent_label(label: str):
    label = label.upper()
    if label in {"CHEMICAL", "DRUG", "CHEM"}:
        return "drug"
    if label in {"DISEASE", "SYMPTOM", "CONDITION"}:
        return "event"
    return None

def extract_from_sentence(sent: str):
    doc = nlp(sent)
    drugs, events = [], []
    for ent in doc.ents:
        mapped = map_ent_label(ent.label_)
        if mapped == "drug":
            drugs.append(ent.text)
        elif mapped == "event":
            events.append(ent.text)
    # dictionary fallback if NER missed something
    s = sent.lower()
    for d in DRUG_DICT:
        if d in s and d not in drugs:
            drugs.append(d)
    for e in EVENT_DICT:
        if e in s and e not in events:
            events.append(e)
    return drugs, events

if __name__ == "__main__":
    records = json.loads(open(CLEAN_FILE, "r", encoding="utf8").read())
    out = []
    for r in records:
        sent = r["sentence"]
        drugs, events = extract_from_sentence(sent)
        if drugs or events:
            out.append({"file": r.get("file"), "sent_idx": r.get("sent_idx"), "sentence": sent, "drugs": drugs, "events": events})
    with open(OUT_FILE, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print("Saved", OUT_FILE)
