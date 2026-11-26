# Secret Wards — Complete Training & Unseen-Domain Adversarial Suite (`dataset-2-test`)

Built for **Track 3: Secret Wards** (`Reinforce-SST/Secret-Wards-ToolKit`) to both **train** your offline CPU-only models (NER CRF + Record Linkage + Quasi-Identifier Privacy Layer) and **stress-test** your full end-to-end pipeline against unseen schemas, homonyms, voter-list linkage attacks, and security exploits.

## Anti-Overfitting Design (Why This Improves Private Evaluation Scores)

The private evaluation runs on **3 hidden datasets** with unseen patients, unseen labels/headers, and unseen formats. To guarantee your pipeline generalizes instead of memorizing names:
- **`training_data/ner_bio_train.jsonl` (1,200 docs)** & **`linkage_pairs_train.csv` (3,000 pairs)** use one pool of Indian names, surnames, and villages.
- **`training_data/ner_bio_val.jsonl` (300 docs)** & **`linkage_pairs_val.csv` (800 pairs)** use a **strictly disjoint** pool of unseen names/villages.
- **`dev_unseen_domain/` (560 records + 26 quarantine attack files)** uses **unseen Hinglish CSV headers, unseen JSON key structures, unseen patients, and 26 security/boundary payloads**.

## Repository Layout

```text
.
├── generate_training_and_eval.py      # Deterministic generator (Python stdlib only)
├── training_data/
│   ├── ner_bio_train.jsonl            # 1,200 token+span BIO-tagged EN/HI/Hinglish docs (incl. hard clinical 'O' negatives)
│   ├── ner_bio_val.jsonl              # 300 disjoint held-out BIO docs
│   ├── linkage_pairs_train.csv        # 3,000 labeled record pairs (cross-script positives + homonym & household hard negatives)
│   ├── linkage_pairs_val.csv          # 800 disjoint held-out record pairs
│   └── qi_privacy_benchmark.csv       # 600 QI tuples with exact k-anonymity & singling-out risk labels
├── dev_unseen_domain/
│   ├── input/                         # 560 unseen-schema records (CSV, BOM CSV, CP1252 TSV, JSON, FHIR R4, TXT) + 26 quarantine attacks
│   ├── truth.json                     # Full ground truth (clusters, PII tokens, canaries, clinical values, LOS, age, quarantine)
│   └── attacker_known.json            # External voter list + known facts for linkage attack simulation
├── tools/
│   ├── check_stress_suite.py          # 100-point scorer, voter-list attacker, LOS/age-band verifier & security audit matrix
│   ├── train_crf_ner.py               # Ready-to-run offline CPU sklearn-crfsuite NER trainer (<50M, zero LLM)
│   └── sync_schema_with_kit.py        # Inspects official Secret-Wards-ToolKit/dev/ schema for 1:1 alignment
└── README.md
```

## Quick Start

```bash
# 1. Train & validate your offline multilingual CRF NER model on disjoint splits
pip install sklearn-crfsuite joblib
python3 tools/train_crf_ner.py \
  --train training_data/ner_bio_train.jsonl \
  --val training_data/ner_bio_val.jsonl

# 2. Run your full Secret Wards pipeline on the unseen-domain adversarial suite
RAKSHA_SECRET=stress_test_key python3 /path/to/your-repo/main.py \
  --input dev_unseen_domain/input \
  --output my_unseen_output

# 3. Run the 100-point scorer, LOS/age verifier & security audit
python3 tools/check_stress_suite.py \
  --input dev_unseen_domain/input \
  --output my_unseen_output \
  --truth dev_unseen_domain/truth.json \
  --attacker dev_unseen_domain/attacker_known.json \
  --show 20
```
