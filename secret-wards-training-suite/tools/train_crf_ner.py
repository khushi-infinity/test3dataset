#!/usr/bin/env python3
"""
Offline CPU-only Multilingual PII CRF Trainer (sklearn-crfsuite)
Trains on training_data/ner_bio_train.jsonl and evaluates on DISJOINT ner_bio_val.jsonl.
Complies 100% with Secret Wards Ground Rules (No LLMs, <50M params, offline CPU).

Install once during setup:
  pip install sklearn-crfsuite scikit-learn joblib
Run:
  python3 tools/train_crf_ner.py --train training_data/ner_bio_train.jsonl --val training_data/ner_bio_val.jsonl
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DIGIT_RE = re.compile(r"[\d०-९]")
ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e]")


def token_features(tokens: list[str], i: int) -> dict[str, object]:
    w = tokens[i]
    w_clean = ZW_RE.sub("", w)
    w_low = w_clean.lower()
    feats = {
        "bias": 1.0,
        "len": len(w_clean),
        "is_title": w_clean.istitle(),
        "is_upper": w_clean.isupper(),
        "has_devanagari": bool(DEVANAGARI_RE.search(w_clean)),
        "has_digit": bool(DIGIT_RE.search(w_clean)),
        "has_zw_evasion": bool(ZW_RE.search(w)),
        "has_at": "@" in w_clean,
        "has_hyphen": "-" in w_clean,
        "has_slash": "/" in w_clean,
        "prefix3": w_low[:3],
        "suffix3": w_low[-3:],
        "shape": re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"[\d०-९]", "d", w_clean)))[:12],
        # Contextual cue markers (teaches meaning trap & Indian honorifics!)
        "is_doctor_prefix": w_low in ("dr.", "dr", "consultant:", "physician:"),
        "is_rel_prefix": w_low in ("s/o", "w/o", "d/o", "c/o", "पिता/पति"),
        "is_honorific": w_low in ("ji", "sahab", "didi", "bhaiya", "smt.", "shri", "mr.", "ms.", "md.", "mohd"),
        "is_med_unit": w_low in ("mg", "ml", "bd", "od", "tds", "hs", "sos", "mmhg", "1-0-1", "1-0-0"),
    }
    if i > 0:
        prev = ZW_RE.sub("", tokens[i - 1]).lower()
        feats.update({
            "-1:lower": prev[:16],
            "-1:is_dr": prev in ("dr.", "dr"),
            "-1:is_co": prev in ("c/o", "s/o", "w/o", "d/o"),
            "-1:is_title": tokens[i - 1].istitle(),
        })
    else:
        feats["BOS"] = True

    if i < len(tokens) - 1:
        nxt = ZW_RE.sub("", tokens[i + 1]).lower()
        feats.update({
            "+1:lower": nxt[:16],
            "+1:is_honorific": nxt in ("ji", "sahab", "didi", "bhaiya", "ka", "ki", "ko"),
            "+1:is_med_unit": nxt in ("mg", "ml", "bd", "od", "tds", "disease", "syndrome", "palsy"),
        })
    else:
        feats["EOS"] = True
    return feats


def load_bio(path: Path):
    X, y = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        toks = obj["tokens"]
        X.append([token_features(toks, i) for i in range(len(toks))])
        y.append(obj["bio_tags"])
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("pii_crf_model.joblib"))
    args = ap.parse_args()

    try:
        import joblib
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install sklearn-crfsuite joblib") from exc

    X_train, y_train = load_bio(args.train)
    X_val, y_val = load_bio(args.val)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=120, all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_val)

    labels = sorted({t for seq in y_val for t in seq if t != "O"})
    print("=== HELD-OUT UNSEEN-DOMAIN BIO NER EVALUATION ===")
    print(metrics.flat_classification_report(y_val, y_pred, labels=labels, digits=4))
    joblib.dump(crf, args.out)
    print(f"[OK] Saved offline CRF model to {args.out}")


if __name__ == "__main__":
    main()
