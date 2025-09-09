### *Person 1* 

* Collect a small dataset (FAERS CSV + PubMed abstracts).
* Write script to clean text (lowercase, remove symbols, sentence split with SpaCy).
* Store cleaned text in data/cleaned/.

---

### *Person 2*

* Implement NER (SciSpacy/Med7) to detect drugs and medical events.
* Output results as JSON: {drug: "X", event: "Y"}.
* Save outputs in results/extractions.json.

---

### *Person 3*

* Build a simple sentence classifier (ADE vs Non-ADE).
* Start with TF-IDF + logistic regression (scikit-learn).
* Expose it as a Python function: classify_sentence(text) -> label, confidence.

---

### *Person 4*

* Set up FastAPI backend.
* Create endpoints:

  * /extract → runs Person 2’s pipeline.
  * /classify → runs Person 3’s classifier.
* Make sure it returns JSON responses.

---

### *Person 5*

* Build a simple React/Next.js frontend.
* Text box → send text to backend /extract + /classify.
* Highlight drugs in blue, ADEs in red.

---

### *Person 6*

* Take Person 2’s extraction results.
* Compute basic stats:

  * Frequency of ADEs.
  * Top drugs with ADEs.
* Show results as bar chart / table in frontend.

---

### Sequence 

1. Person 1 finishes data prep.
2. Person 2 + Person 3 use cleaned data to build models.
3. Person 4 integrates extraction + classification into backend.
4. Person 5 connects frontend to backend.
5. Person 6 adds stats/visualization once extraction results are available.

---


### Final Flow
* Data → NER → Classification → API → UI → Stats.
