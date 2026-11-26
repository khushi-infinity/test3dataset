#!/usr/bin/env python3
"""
Secret Wards — 100-Point Automated Scorer, Attacker Simulator, LOS/Age Verifier & Security Audit
Works on dev_unseen_domain/ (and any dataset following its truth.json schema).
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
DEVANAGARI_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")
CSV_FORMULA_START = ("=", "+", "-", "@", "\t=", "\r=")
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
AGE_BAND_RE = re.compile(r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\b")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_for_leak_scan(text: str) -> str:
    text = ZW_RE.sub("", text)
    text = text.translate(DEVANAGARI_TO_ASCII)
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def scan_json_keys(obj) -> list[str]:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(scan_json_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.extend(scan_json_keys(item))
    return keys


def verify_los_and_age_band(text: str, true_los: int | None, true_age: int | None) -> tuple[bool | None, bool | None]:
    """Checks if shifted ISO dates preserve exact LOS and if any age band contains true_age."""
    los_ok = None
    if true_los is not None:
        dates = []
        for m in ISO_DATE_RE.findall(text):
            try:
                dates.append(date.fromisoformat(m))
            except ValueError:
                pass
        if len(dates) >= 2:
            # Check if any consecutive or min/max pair matches exact true_los
            diffs = {abs((dates[j] - dates[i]).days) for i in range(len(dates)) for j in range(i + 1, len(dates))}
            los_ok = (true_los in diffs)
        else:
            los_ok = False

    age_ok = None
    if true_age is not None:
        bands = [(int(a), int(b)) for a, b in AGE_BAND_RE.findall(text) if int(a) <= int(b) <= 120]
        if bands:
            age_ok = any(lo <= true_age <= hi for lo, hi in bands)
        elif re.search(rf"\b{true_age}\b", text):
            age_ok = True  # Exact age kept (truthful, though hurts privacy score)
        else:
            age_ok = False
    return los_ok, age_ok


def compute_pairwise_linkage(true_map, pred_map, homonym_pairs) -> dict:
    true_clusters = defaultdict(list)
    pred_clusters = defaultdict(list)
    for k, t_pid in true_map.items():
        if t_pid is not None:
            true_clusters[t_pid].append(k)
    for k, p_pid in pred_map.items():
        if p_pid is not None and k in true_map:
            pred_clusters[p_pid].append(k)

    true_pairs = set()
    for recs in true_clusters.values():
        s = sorted(recs)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                true_pairs.add((s[i], s[j]))

    pred_pairs = set()
    for recs in pred_clusters.values():
        s = sorted(recs)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pred_pairs.add((s[i], s[j]))

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not true_pairs else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    homonym_set = {frozenset((a, b)) for a, b, _ in homonym_pairs}
    homonym_merges = 0
    for recs in pred_clusters.values():
        t_pids = list({true_map[r] for r in recs if true_map.get(r) is not None})
        for i in range(len(t_pids)):
            for j in range(i + 1, len(t_pids)):
                if frozenset((t_pids[i], t_pids[j])) in homonym_set:
                    homonym_merges += 1

    weak_overlinks = sum(1 for k, t_pid in true_map.items() if t_pid is None and pred_map.get(k) is not None)
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "homonym_false_merges": homonym_merges, "weak_overlinks": weak_overlinks,
    }


def run_voter_attack(sanitized_texts: dict[str, str], attacker_targets: list[dict]) -> dict:
    corpus_norm = {k: normalize_for_leak_scan(v) for k, v in sanitized_texts.items()}
    reidentified = []
    for tgt in attacker_targets:
        diag = tgt["known_diagnosis"].lower()
        village = tgt["known_village"].lower()
        pin = tgt["known_pin"].lower()
        age_str = str(tgt["known_age"])
        matched = []
        for fpath, body in corpus_norm.items():
            has_diag = diag in body
            has_loc = (village in body) or (pin in body)
            has_age = bool(re.search(rf"\b{re.escape(age_str)}\b", body))
            if (has_diag and has_loc) or (has_loc and has_age and tgt["risk_tier"] == "critical"):
                matched.append(fpath)
        if 1 <= len(matched) <= 3:
            reidentified.append({"true_patient_id": tgt["true_patient_id"], "files": matched[:3]})
    rate = len(reidentified) / max(len(attacker_targets), 1)
    return {
        "score": round(max(0.0, 25.0 * (1.0 - min(1.0, rate * 1.5))), 2),
        "targets_total": len(attacker_targets),
        "targets_reidentified": len(reidentified),
    }


def evaluate(input_dir: Path, output_dir: Path, truth_path: Path, attacker_path: Path, show_n: int) -> None:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    attacker = json.loads(attacker_path.read_text(encoding="utf-8"))
    sanitized_dir = output_dir / "sanitized"
    report_path = output_dir / "report.json"
    records_path = output_dir / "records.jsonl"

    if not (sanitized_dir.exists() and report_path.exists() and records_path.exists()):
        print("[CRITICAL] Missing sanitized/, report.json, or records.jsonl")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_files = {p.relative_to(input_dir).as_posix(): sha256_file(p) for p in input_dir.rglob("*") if p.is_file()}
    reported_by_sha = {e.get("input_sha256"): e for e in report.get("files", [])}

    failures = []
    penalties = 0.0
    vuln = defaultdict(lambda: {"pass": 0, "fail": 0})
    validity = 10.0

    skipped = [rel for rel, h in input_files.items() if h not in reported_by_sha]
    if skipped:
        penalties += 5.0 * len(skipped)
        validity -= min(4.0, 0.5 * len(skipped))
        failures.append(f"[SILENT SKIP] {len(skipped)} files absent from report.json: {skipped[:3]}")

    bad_q = 0
    for q in truth["expected_quarantine"]:
        rep = reported_by_sha.get(q["input_sha256"])
        tag = q["security_tag"]
        if not rep or rep.get("status") != "quarantined" or rep.get("output") is not None:
            bad_q += 1
            vuln[tag]["fail"] += 1
            failures.append(f"[QUARANTINE FAIL] {q['rel_path']} ({tag})")
        else:
            vuln[tag]["pass"] += 1
    if bad_q:
        penalties += 10.0 * bad_q
        validity -= min(4.0, bad_q * 1.0)

    sanitized_texts = {}
    formula_leaks = 0
    for sf in sanitized_dir.rglob("*"):
        if not sf.is_file():
            continue
        rel = sf.relative_to(sanitized_dir).as_posix()
        try:
            content = sf.read_text(encoding="utf-8")
            sanitized_texts[rel] = content
        except UnicodeDecodeError:
            validity -= 1.0
            failures.append(f"[INVALID UTF-8] {rel}")
            continue
        if sf.suffix == ".json":
            try:
                parsed = json.loads(content)
                if "fhir" in rel and parsed.get("resourceType") != "Bundle":
                    validity -= 0.5
                    failures.append(f"[INVALID FHIR] {rel}")
                keys_blob = " ".join(scan_json_keys(parsed)).lower()
                if "patient_" in keys_blob and "_audit" in keys_blob:
                    vuln["pii_in_json_key"]["fail"] += 1
                    failures.append(f"[JSON KEY LEAK] {rel}")
                else:
                    vuln["pii_in_json_key"]["pass"] += 1
            except Exception as exc:
                validity -= 1.0
                failures.append(f"[INVALID JSON] {rel}: {exc}")
        elif sf.suffix in (".csv", ".tsv"):
            delim = "\t" if sf.suffix == ".tsv" else ","
            for r_i, row in enumerate(csv.reader(content.splitlines(), delimiter=delim)):
                for cell in row:
                    if cell.startswith(CSV_FORMULA_START):
                        formula_leaks += 1
                        vuln["csv_formula_injection"]["fail"] += 1
                        failures.append(f"[CSV FORMULA LEAK] {rel} row:{r_i}: {cell[:25]}")
    if formula_leaks == 0:
        vuln["csv_formula_injection"]["pass"] += 1

    for orig in truth["filename_pii_files"]:
        if orig in sanitized_texts:
            vuln["pii_in_filename"]["fail"] += 1
            validity -= 0.5
            failures.append(f"[FILENAME PII LEAK] {orig}")
        else:
            vuln["pii_in_filename"]["pass"] += 1

    all_blob = normalize_for_leak_scan(
        report_path.read_text(errors="replace") + "\n" + records_path.read_text(errors="replace")
        + "\n" + "\n".join(sanitized_texts.keys()) + "\n" + "\n".join(sanitized_texts.values())
    )
    leaked_canaries = [c for c in sorted(set(truth["canaries"])) if c.lower() in all_blob]
    for c in leaked_canaries:
        penalties += 15.0
        vuln["canary_zero_leak"]["fail"] += 1
        failures.append(f"[CANARY LEAK -15pts] {c}")
    if not leaked_canaries:
        vuln["canary_zero_leak"]["pass"] += 1

    pii_tot = pii_leak = clin_tot = clin_kept = los_tot = los_ok_c = age_tot = age_ok_c = 0
    for rec in truth["records"]:
        rel = rec["rel_path"]
        body = sanitized_texts.get(rel)
        if body is None:
            out_rel = (reported_by_sha.get(input_files.get(rel), {}).get("output") or "").replace("sanitized/", "", 1)
            body = sanitized_texts.get(out_rel, "")
        body_norm = normalize_for_leak_scan(body)

        leaked = False
        for pii in rec["pii_must_redact"]:
            if not pii or len(pii.strip()) < 4:
                continue
            pii_tot += 1
            if normalize_for_leak_scan(pii) in body_norm:
                pii_leak += 1
                leaked = True
                if len(failures) < 500:
                    failures.append(f"[PII LEAK] {rel} ({rec['loc']}): {pii!r}")
        for tag in rec.get("security_tags", []):
            vuln[tag]["fail" if leaked else "pass"] += 1

        for clin in rec["clinical_must_keep"]:
            clin_tot += 1
            if clin.lower() in body.lower():
                clin_kept += 1
            elif len(failures) < 500:
                failures.append(f"[CLINICAL LOSS] {rel} ({rec['loc']}): {clin!r}")

        if rec["loc"] == "doc":
            l_ok, a_ok = verify_los_and_age_band(body, rec.get("true_los"), rec.get("true_age"))
            if l_ok is not None:
                los_tot += 1
                los_ok_c += int(l_ok)
            if a_ok is not None:
                age_tot += 1
                age_ok_c += int(a_ok)

    pii_score = round(20.0 * max(0.0, 1.0 - pii_leak / max(pii_tot, 1)), 2)
    clin_ratio = clin_kept / max(clin_tot, 1)
    los_ratio = (los_ok_c / los_tot) if los_tot else 1.0
    age_ratio = (age_ok_c / age_tot) if age_tot else 1.0
    med_score = round(14.0 * clin_ratio + 3.0 * los_ratio + 3.0 * age_ratio, 2)

    true_map = {(r["rel_path"], r["loc"]): r["true_patient_id"] for r in truth["records"]}
    pred_map = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            pred_map[(obj["file"].replace("sanitized/", "", 1), obj["loc"])] = obj.get("patient")
    lnk = compute_pairwise_linkage(true_map, pred_map, truth["homonym_pairs"])
    lnk_score = max(0.0, round(15.0 * lnk["f1"] - 1.5 * lnk["homonym_false_merges"] - 0.5 * lnk["weak_overlinks"], 2))

    atk = run_voter_attack(sanitized_texts, attacker["targets"])
    secs = report.get("timing", {}).get("seconds", 9999)
    rss = report.get("resources", {}).get("peak_rss_mb", 9999)
    spd = max(0.0, 10.0 - (6.0 if secs > 900 else (2.0 if secs > 120 else 0.0)) - (6.0 if rss > 4096 else (2.0 if rss > 1500 else 0.0)))
    validity = max(0.0, round(validity, 2))
    final = round(atk["score"] + pii_score + med_score + lnk_score + validity + spd - penalties, 2)

    print("=" * 72)
    print("  SECRET WARDS — UNSEEN-DOMAIN STRESS-TEST SCORECARD")
    print("=" * 72)
    print(f"  1. Privacy Against Attacks   : {atk['score']:5.2f} / 25.00  (Re-ID: {atk['targets_reidentified']}/{atk['targets_total']})")
    print(f"  2. Personal Data Detection   : {pii_score:5.2f} / 20.00  (Leaked: {pii_leak}/{pii_tot})")
    print(f"  3. Medical Data Preservation : {med_score:5.2f} / 20.00  (Clin:{clin_kept}/{clin_tot}, LOS:{los_ok_c}/{los_tot}, Age:{age_ok_c}/{age_tot})")
    print(f"  4. Patient Identity Linkage  : {lnk_score:5.2f} / 15.00  (F1={lnk['f1']}, HomonymMerges={lnk['homonym_false_merges']}, WeakOverlinks={lnk['weak_overlinks']})")
    print(f"  5. Output Quality & Validity : {validity:5.2f} / 10.00  (BadQ={bad_q}, FormulaLeaks={formula_leaks})")
    print(f"  6. Speed & Reliability       : {spd:5.2f} / 10.00  (Time={secs}s, PeakRSS={rss}MB)")
    print("-" * 72)
    print(f"  Hard Penalties               : -{penalties:5.2f}         (Canaries:{len(leaked_canaries)}, SilentSkips:{len(skipped)})")
    print(f"  FINAL SCORE                  : {final:6.2f} / 100.00")
    print("=" * 72)
    print("\n  SECURITY & VULNERABILITY AUDIT MATRIX")
    for tag in sorted(vuln.keys()):
        st = "OK" if vuln[tag]["fail"] == 0 else "VULNERABLE"
        print(f"  {tag:<34} PASS={vuln[tag]['pass']:<5} FAIL={vuln[tag]['fail']:<5} [{st}]")
    if failures and show_n > 0:
        print(f"\n  TOP {min(show_n, len(failures))} FAILURES (of {len(failures)}):")
        for f in failures[:show_n]:
            print(f"   - {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--attacker", required=True, type=Path)
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()
    evaluate(a.input, a.output, a.truth, a.attacker, a.show)
