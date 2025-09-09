# backend/data_ingestion.py
import re, json, os
from pathlib import Path
import spacy

BASE = Path(_file_).resolve().parents[1]   # project root
RAW_DIR = BASE / "data" / "raw"
CLEAN_DIR = BASE / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("en_core_web_sm")  # sentence splitting

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\.\,\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_to_sentences(text: str):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def process_all_txt():
    out = []
    for p in RAW_DIR.glob("*.txt"):
        with open(p, "r", encoding="utf8") as f:
            text = f.read()
        cleaned = clean_text(text)
        sents = split_to_sentences(cleaned)
        # keep origin info optional
        for i, s in enumerate(sents):
            out.append({"file": p.name, "sent_idx": i, "sentence": s})
    return out

if _name_ == "_main_":
    sentences = process_all_txt()
    out_file = CLEAN_DIR / "sentences.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(sentences, f, indent=2)
    print("Saved", out_file)
