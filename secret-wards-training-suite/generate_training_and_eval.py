#!/usr/bin/env python3
"""
Secret Wards — Complete ML Training & Adversarial Evaluation Suite Generator
============================================================================
Generates two strictly disjoint, 100% synthetic datasets (zero patient/name leakage
between train, val, and unseen-domain test):

1. training_data/
   - ner_bio_train.jsonl       (1,200 token/span BIO-labeled EN/HI/Hinglish clinical docs + hard clinical negatives)
   - ner_bio_val.jsonl         (300 held-out BIO-labeled docs from DISJOINT name/location pools)
   - linkage_pairs_train.csv   (3,000 labeled patient-record pairs: positive cross-script, homonym negatives, weak links)
   - linkage_pairs_val.csv     (800 held-out record pairs from DISJOINT patients)
   - qi_privacy_benchmark.csv  (600 patient QI tuples with k-anonymity & singling-out risk labels)

2. dev_unseen_domain/          (Simulates the Hidden Private Datasets: unseen column headers, unseen JSON keys,
                                unseen names/villages, + 26 security/boundary/quarantine attacks)
   - input/                    (560 patient records across CSV, TSV, JSON, FHIR R4, TXT + 26 adversarial quarantine files)
   - truth.json                (Full ground truth: exact PII spans, canaries, clinical values, LOS, ages, clusters, security tags)
   - attacker_known.json       (External voter register + known public-health facts for linkage attack simulation)

Zero external dependencies (Python 3.10+ stdlib only). Deterministic via seed.
"""

from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import random
import re
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

SEED = 20261126
DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

# =========================================================================
# STRICTLY DISJOINT ENTITY POOLS (Prevents overfitting / memorization!)
# =========================================================================
TRAIN_FIRST_M = [
    ("Aarav", "आरव"), ("Rohan", "रोहन"), ("Manoj", "मनोज"), ("Nitin", "नितिन"),
    ("Farooq", "फारूक"), ("Arjun", "अर्जुन"), ("Gaurav", "गौरव"), ("Rakesh", "राकेश"),
    ("Tariq", "तारिक"), ("Vivek", "विवेक"), ("Pradeep", "प्रदीप"), ("Ashok", "अशोक"),
    ("Shubham", "शुभम"), ("Pankaj", "पंकज"), ("Wasim", "वसीम"), ("Naveen", "नवीन"),
    ("Mahendra", "महेंद्र"), ("Dheeraj", "धीरज"), ("Sachin", "सचिन"), ("Karan", "करण"),
    ("Brijesh", "बृजेश"), ("Hemant", "हेमंत"), ("Javed", "जावेद"), ("Yogesh", "योगेश"),
]
TRAIN_FIRST_F = [
    ("Anjali", "अंजलि"), ("Rekha", "रेखा"), ("Suman", "सुमन"), ("Ritu", "रितु"),
    ("Shabnam", "शबनम"), ("Nisha", "निशा"), ("Seema", "सीमा"), ("Babita", "बबीता"),
    ("Kiran", "किरण"), ("Jyoti", "ज्योति"), ("Razia", "रज़िया"), ("Savita", "सविता"),
    ("Priyanka", "प्रियंका"), ("Mamta", "ममता"), ("Archana", "अर्चना"), ("Swati", "स्वाति"),
    ("Shalini", "शालिनी"), ("Poonam", "पूनम"), ("Roshni", "रोशनी"), ("Madhu", "मधु"),
    ("Vandana", "वंदना"), ("Sadhna", "साधना"), ("Farida", "फरीदा"), ("Chanda", "चंदा"),
]
TRAIN_LAST = [
    ("Saxena", "सक्सेना"), ("Tiwari", "तिवारी"), ("Dubey", "दुबे"), ("Pandey", "पांडेय"),
    ("Ansari", "अंसारी"), ("Chauhan", "चौहान"), ("Rastogi", "रस्तोगी"), ("Bhatt", "भट्ट"),
    ("Srivastava", "श्रीवास्तव"), ("Tripathi", "त्रिपाठी"), ("Qureshi", "कुरैशी"), ("Jain", "जैन"),
    ("Agarwal", "अग्रवाल"), ("Bansal", "बंसल"), ("Rawat", "रावत"), ("Negi", "नेगी"),
]
TRAIN_VILLAGES = [
    ("Sitapur Khas", "261001", "Sitapur", True),
    ("Malihabad", "226102", "Lucknow", True),
    ("Unnao Rural", "209801", "Unnao", True),
    ("Mohanlalganj", "226301", "Lucknow", True),
    ("Kanpur Central", "208001", "Kanpur Nagar", False),
    ("Agra Cantt", "282001", "Agra", False),
    ("Meerut City", "250001", "Meerut", False),
    ("Bareilly Urban", "243001", "Bareilly", False),
]

# Completely unseen names & villages for Validation & Unseen-Domain Test!
UNSEEN_FIRST_M = [
    ("Mohammad", "मोहम्मद"), ("Rajesh", "राजेश"), ("Suresh", "सुरेश"), ("Anil", "अनील"),
    ("Deepak", "दीपक"), ("Ramesh", "रमेश"), ("Imran", "इमरान"), ("Harish", "हरीश"),
    ("Vikram", "विक्रम"), ("Sanjay", "संजय"), ("Devendra", "देवेंद्र"), ("Kamlesh", "कमलेश"),
    ("Zafar", "ज़फ़र"), ("Gopal", "गोपाल"), ("Santosh", "संतोष"), ("Alok", "आलोक"),
]
UNSEEN_FIRST_F = [
    ("Sunita", "सुनीता"), ("Meera", "मीरा"), ("Priya", "प्रिया"), ("Fatima", "फातिमा"),
    ("Kavita", "कविता"), ("Lakshmi", "लक्ष्मी"), ("Geeta", "गीता"), ("Pooja", "पूजा"),
    ("Asha", "आशा"), ("Neha", "नेहा"), ("Kalyani", "कल्याणी"), ("Usha", "उषा"),
    ("Zoya", "ज़ोया"), ("Bhavna", "भावना"), ("Pushpa", "पुष्पा"), ("Sarita", "सरिता"),
]
UNSEEN_LAST = [
    ("Arif", "आरिफ"), ("Devi", "देवी"), ("Kumar", "कुमार"), ("Sharma", "शर्मा"),
    ("Nair", "नायर"), ("Yadav", "यादव"), ("Begum", "बेगम"), ("Verma", "वर्मा"),
    ("Patel", "पटेल"), ("Mishra", "मिश्रा"), ("Chaudhary", "चौधरी"), ("Iyer", "अय्यर"),
    ("Siddiqui", "सिद्दीकी"), ("Thakur", "ठाकुर"), ("Reddy", "रेड्डी"), ("Kulkarni", "कुलकर्णी"),
]
UNSEEN_VILLAGES = [
    ("Khera Khurd", "226101", "Lucknow", True),
    ("Rampur Maniharan", "247451", "Saharanpur", True),
    ("Bhitargaon", "209209", "Kanpur Nagar", True),
    ("Chauri Chaura", "273201", "Gorakhpur", True),
    ("Barabanki", "225001", "Barabanki", False),
    ("Varanasi Urban", "221001", "Varanasi", False),
    ("Prayagraj Civil Lines", "211001", "Prayagraj", False),
    ("Noida Sector 62", "201309", "Gautam Buddha Nagar", False),
]

# Clinical whitelists (MUST NEVER be redacted -> tagged 'O' in BIO data!)
DOCTORS = [
    "Dr. Ramesh Kulkarni", "Dr. Anita Deshmukh", "Dr. S. K. Sharma",
    "Dr. Farhan Qureshi", "Dr. Meenakshi Sundaram", "Dr. Rajeshwar Rao",
    "Dr. Arvind Kejriwal", "Dr. Nandini Menon", "Dr. P. V. Narasimhan",
]
EPONYMS = [
    "Parkinson's disease", "Hodgkin's lymphoma", "Bell's palsy",
    "Guillain-Barré syndrome", "Apgar score 9/10", "Babinski sign negative",
    "Ringer's lactate 500 mL IV", "Foley catheter 16Fr", "Addison's disease",
    "Cushing's syndrome", "Crohn's disease", "Ryles tube 14Fr",
]
INDIAN_MEDS = [
    "Dolo 650 mg BD x 5 days", "Ramipril 5 mg OD x 30 days", "Glycomet-GP 1 mg 1-0-1",
    "Telma 40 mg OD", "Augmentin 625 Duo TDS x 7 days", "Pan-D 40 mg 1-0-0 AC",
    "Azee 500 mg OD x 3 days", "Metformin 500 mg BD", "Atorva 20 mg HS",
    "Ecosprin 75 mg OD", "Crocin Advance 500 mg SOS", "Shelcal 500 mg OD",
    "Amlong 5 mg OD", "Montair-LC HS", "Thyronorm 50 mcg OD empty stomach",
]
COMMON_DIAGNOSES = [
    "Type 2 Diabetes Mellitus", "Essential Hypertension", "Dengue Fever",
    "Acute Viral Gastroenteritis", "Community Acquired Pneumonia",
    "Enteric Fever (Typhoid)", "Iron Deficiency Anemia", "Acute Bronchitis",
    "Urinary Tract Infection", "Hypothyroidism",
]
RARE_DIAGNOSES = [
    "Visceral Leishmaniasis (Kala-azar)", "Guillain-Barré syndrome",
    "Wilson's disease", "Takayasu arteritis", "Myasthenia gravis",
    "Scrub typhus with meningoencephalitis", "Porphyria cutanea tarda",
]

CSV_FORMULA_PAYLOADS = [
    "=CMD|' /C calc'!A0",
    "+SUM(1+1)*cmd|' /C calc'!A0",
    "-2+3+cmd|' /C calc'!A0",
    "@HYPERLINK(\"http://evil.invalid\",\"click\")",
    "\t=1+1",
]
INTEGER_BOUNDARY_VALUES = [
    "-1", "0", "150", "2147483647", "2147483648",
    "9223372036854775807", "9223372036854775808",
    "-9223372036854775809", "NaN", "Infinity", "0x7fffffff", "1e309",
]
SANITIZATION_INJECTIONS = [
    "'; DROP TABLE patients;--",
    "{\"$ne\": null}",
    "{{7*7}} ${jndi:ldap://127.0.0.1/a}",
    "<script>alert('xss')</script>",
    "%s%s%s%s%n",
    "Fever\\r\\n[INFO] SANITIZED_OK status=clean",
]

TOKEN_RE = re.compile(r"\S+")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inject_zwsp(text: str) -> str:
    if len(text) < 4:
        return text
    mid = len(text) // 2
    return text[:mid] + "\u200b" + text[mid:]


def synth_phone(rng: random.Random, style: int) -> tuple[str, str]:
    digits = rng.choice(["6", "7", "8", "9"]) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    if style == 0:
        return f"+91-{digits[:5]}-{digits[5:]}", digits
    if style == 1:
        return f"0{digits[:3]} {digits[3:6]} {digits[6:]}", digits
    if style == 2:
        return digits.translate(DEVANAGARI_DIGITS), digits
    if style == 3:
        return f"{digits[0]}.{digits[1:6]}E+09", digits
    if style == 4:
        return f"{digits[:4]}\u200b{digits[4:7]}\u200b{digits[7:]}", digits
    return digits, digits


def synth_ids(rng: random.Random, idx: int, prefix: str = "YP") -> dict[str, str]:
    """Generate permitted synthetic Indian healthcare & clinic IDs (strictly non-real)."""
    return {
        "abha": f"{rng.randint(10,99)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}",
        "pan_like": f"SYNTH{rng.randint(1000,9999)}Z",
        "uhid": f"UHID-{prefix}-{2020 + (idx % 6)}-{100000 + idx}",
        "mrn": f"MRN/{rng.choice(['H1','H2','H3','H4'])}/{50000 + idx}",
        "insurance_id": f"PMJAY-{prefix}-{rng.randint(10000000, 99999999)}",
        "voter_epic": f"YPX{rng.randint(1000000, 9999999)}",
    }


def make_aliases(first_en: str, last_en: str, gender: str) -> list[str]:
    hon = ["ji", "didi", "ben", "devi"] if gender == "F" else ["ji", "sahab", "bhaiya"]
    prefix = ["Smt.", "Ms.", "Km."] if gender == "F" else ["Shri", "Mr.", "Md."] if first_en == "Mohammad" else ["Shri", "Mr."]
    aliases = [
        f"{first_en[0]}. {last_en}",
        f"{last_en}, {first_en}",
        f"{first_en} {hon[0]}",
        f"{prefix[0]} {first_en} {last_en}",
        f"{first_en} {last_en[0]}.",
    ]
    if first_en == "Mohammad":
        aliases.extend([f"Md. {last_en}", f"Mohd {last_en}", f"Mohd. {last_en}"])
    return aliases


def build_patient_pool(rng: random.Random, n: int, first_m, first_f, last_pool, village_pool, id_prefix: str) -> list[dict]:
    patients = []
    for i in range(n):
        gender = "M" if i % 2 == 0 else "F"
        f_en, f_hi = rng.choice(first_m if gender == "M" else first_f)
        l_en, l_hi = rng.choice(last_pool)
        # Force intentional homonyms (same name, DIFFERENT person!) on every 9th patient
        if i >= 12 and i % 9 == 0:
            prev = patients[i - 10]
            f_en, l_en = prev["first"], prev["last"]
            f_hi, l_hi = prev["full_hi"].split()[0], prev["full_hi"].split()[-1]
            gender = prev["gender"]

        rel_en, rel_hi = rng.choice(first_m)
        rel_prefix = rng.choice(["W/o", "D/o", "C/O"]) if gender == "F" else rng.choice(["S/o", "C/O"])
        village, pin, district, is_small = rng.choice(village_pool)
        age = rng.randint(1, 92)
        is_rare = (i % 8 == 0)
        diagnosis = rng.choice(RARE_DIAGNOSES) if is_rare else rng.choice(COMMON_DIAGNOSES)
        admit = date(2024, 1, 1) + timedelta(days=rng.randint(0, 680))
        if i == 5:
            admit = date(2024, 2, 29)
        elif i == 6:
            admit = date(2025, 12, 30)
        los = rng.randint(1, 21)
        discharge = admit + timedelta(days=los)
        phone_raw, phone_digits = synth_phone(rng, i % 6)
        ids = synth_ids(rng, i, prefix=id_prefix)
        canary = f"CANARY-{id_prefix}-{i:04d}-{rng.randint(1000,9999)}" if (i % 10 == 0) else None

        patients.append({
            "true_patient_id": f"TP-{id_prefix}-{i:04d}",
            "first": f_en,
            "last": l_en,
            "full_en": f"{f_en} {l_en}",
            "full_hi": f"{f_hi} {l_hi}",
            "aliases": make_aliases(f_en, l_en, gender),
            "gender": gender,
            "age": age,
            "relative_prefix": rel_prefix,
            "relative_en": rel_en,
            "relative_hi": rel_hi,
            "village": village,
            "pin": pin,
            "district": district,
            "is_small_village": is_small,
            "diagnosis": diagnosis,
            "is_rare": is_rare,
            "admit_date": admit.isoformat(),
            "discharge_date": discharge.isoformat(),
            "los": los,
            "med": rng.choice(INDIAN_MEDS),
            "eponym": rng.choice(EPONYMS),
            "doctor": rng.choice(DOCTORS),
            "bp": f"{rng.randint(102, 174)}/{rng.randint(62, 106)}",
            "hba1c": f"{rng.uniform(5.1, 11.8):.1f}%",
            "phone_raw": phone_raw,
            "phone_digits": phone_digits,
            "email": f"{f_en.lower()}.{l_en.lower()}{i}@yugaantar-mail.invalid",
            "ids": ids,
            "canary": canary,
        })
    return patients


# =========================================================================
# PART 1: NER BIO TRAINING & VALIDATION GENERATOR (For CRF / spaCy)
# =========================================================================
def build_bio_example(rng: random.Random, p: dict, template_id: int) -> dict:
    """
    Constructs a clinical snippet from labeled segments so character spans and
    BIO token tags are 100% exact by construction (zero alignment bugs).
    Each segment is (text, label) where label is None for 'O' or entity type.
    """
    alias = rng.choice(p["aliases"])
    segs: list[tuple[str, str | None]] = []

    if template_id == 0:
        # English discharge snippet with Meaning Traps (Dr., C/O fever, eponym, Indian drug)
        segs = [
            ("Patient ", None), (p["full_en"], "PERSON"),
            (f" ({p['age']}{p['gender']}), {p['relative_prefix']} ", None), (p["relative_en"], "RELATIVE"),
            (", UHID ", None), (p["ids"]["uhid"], "ID"),
            (", Mob ", None), (p["phone_raw"], "PHONE"),
            (", resident of ", None), (f"{p['village']} {p['pin']}", "LOCATION"),
            (". Admitted on ", None), (p["admit_date"], "DATE"),
            (f" C/O fever and {p['diagnosis']} (r/o {p['eponym']}). BP {p['bp']} mmHg. Rx: {p['med']}. Seen by {p['doctor']}.", None),
        ]
    elif template_id == 1:
        # Devanagari Hindi OPD note
        segs = [
            ("मरीज़ ", None), (p["full_hi"], "PERSON"),
            (" (पिता/पति ", None), (p["relative_hi"], "RELATIVE"),
            (") पता ", None), (p["village"], "LOCATION"),
            (" फ़ोन ", None), (p["phone_digits"].translate(DEVANAGARI_DIGITS), "PHONE"),
            (" ABHA ", None), (p["ids"]["abha"], "ID"),
            (f" को {p['diagnosis']} की शिकायत है। {p['med']} चालू करें। {p['doctor']} द्वारा देखा गया।", None),
        ]
    elif template_id == 2:
        # Code-switched Hinglish nursing note
        segs = [
            (alias, "PERSON"),
            (f" ka BP {p['bp']} hai, HbA1c {p['hba1c']}. Unke relative ", None),
            (p["relative_en"], "RELATIVE"),
            (" ko phone ", None), (p["phone_raw"], "PHONE"),
            (" par inform kiya. Email ", None), (p["email"], "EMAIL"),
            (f". Diagnosis: {p['diagnosis']}; {p['eponym']}. Dawai: {p['med']}.", None),
        ]
    elif template_id == 3:
        # HARD NEGATIVE clinical-only snippet (0 PII entities — teaches model NOT to over-redact!)
        segs = [
            (f"Ward Round Note: C/O high fever, chills, and cough x 4 days. Rule out {p['eponym']} and {p['diagnosis']}. "
             f"Vitals: BP {p['bp']} mmHg, SpO2 97%, HbA1c {p['hba1c']}. Rx: {p['med']} and {rng.choice(INDIAN_MEDS)}. "
             f"Reviewed by {p['doctor']} and {rng.choice(DOCTORS)}.", None),
        ]
    elif template_id == 4:
        # Adversarial Unicode evasion (ZWSP + Devanagari digits + C/O collision)
        segs = [
            ("Reg ", None), (p["ids"]["mrn"], "ID"),
            (" | Name: ", None), (inject_zwsp(p["full_en"]), "PERSON"),
            (" | C/O ", None), (p["relative_en"], "RELATIVE"),
            (" | C/O chest pain and fever | PAN ", None), (p["ids"]["pan_like"], "ID"),
            (" | Ph ", None), (p["phone_raw"], "PHONE"),
            (" | DOA ", None), (p["admit_date"], "DATE"),
            (f" | Rx {p['med']} | {p['doctor']}", None),
        ]
    else:
        # Insurance + Voter EPIC + Email + Discharge Date
        segs = [
            ("Billing Desk: ", None), (p["full_hi"], "PERSON"),
            (" / ", None), (alias, "PERSON"),
            (" | PMJAY ", None), (p["ids"]["insurance_id"], "ID"),
            (" | EPIC ", None), (p["ids"]["voter_epic"], "ID"),
            (" | Mail ", None), (p["email"], "EMAIL"),
            (" | DOD ", None), (p["discharge_date"], "DATE"),
            (f" | {p['diagnosis']} | {p['doctor']}", None),
        ]

    # Assemble exact text + character spans
    text_parts = []
    entities = []
    cursor = 0
    for txt, label in segs:
        start = cursor
        text_parts.append(txt)
        cursor += len(txt)
        if label is not None:
            entities.append({"start": start, "end": cursor, "label": label, "text": txt})
    full_text = "".join(text_parts)

    # Whitespace-tokenized BIO tags aligned strictly to character spans
    tokens = []
    bio_tags = []
    for m in TOKEN_RE.finditer(full_text):
        t_start, t_end = m.start(), m.end()
        tok = m.group(0)
        tag = "O"
        for ent in entities:
            if t_start < ent["end"] and t_end > ent["start"]:
                prefix = "B-" if (not bio_tags or bio_tags[-1] not in (f"B-{ent['label']}", f"I-{ent['label']}")) else "I-"
                tag = f"{prefix}{ent['label']}"
                break
        tokens.append(tok)
        bio_tags.append(tag)

    return {
        "text": full_text,
        "entities": entities,
        "tokens": tokens,
        "bio_tags": bio_tags,
        "template_id": template_id,
    }


def generate_ner_splits(rng: random.Random, train_patients: list[dict], val_patients: list[dict], out_dir: Path) -> tuple[int, int]:
    train_path = out_dir / "ner_bio_train.jsonl"
    val_path = out_dir / "ner_bio_val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for i in range(1200):
            p = train_patients[i % len(train_patients)]
            ex = build_bio_example(rng, p, i % 6)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for i in range(300):
            p = val_patients[i % len(val_patients)]
            ex = build_bio_example(rng, p, i % 6)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return 1200, 300


# =========================================================================
# PART 2: RECORD LINKAGE PAIRWISE TRAINING & VAL DATA (For Splink / ML)
# =========================================================================
def typo_mutate(rng: random.Random, s: str) -> str:
    if len(s) < 5:
        return s
    i = rng.randint(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def generate_linkage_pairs(rng: random.Random, patients: list[dict], n_pairs: int, out_path: Path) -> int:
    headers = [
        "pair_id", "label", "pair_type",
        "rec_a_name", "rec_a_guardian", "rec_a_age", "rec_a_gender", "rec_a_phone", "rec_a_id", "rec_a_district",
        "rec_b_name", "rec_b_guardian", "rec_b_age", "rec_b_gender", "rec_b_phone", "rec_b_id", "rec_b_district",
    ]
    rows = []
    # Index homonyms for hard negative mining
    by_name: dict[str, list[dict]] = {}
    for p in patients:
        by_name.setdefault(p["full_en"], []).append(p)
    homonym_groups = [g for g in by_name.values() if len(g) > 1]

    for idx in range(n_pairs):
        kind = idx % 5
        if kind in (0, 1):
            # POSITIVE: Same patient across English <-> Hindi <-> Hinglish/Alias + noisy phone/missing ID
            p = rng.choice(patients)
            name_a = p["full_en"]
            name_b = p["full_hi"] if kind == 0 else rng.choice(p["aliases"])
            if rng.random() < 0.25:
                name_a = typo_mutate(rng, name_a)
            phone_b = p["phone_digits"].translate(DEVANAGARI_DIGITS) if rng.random() < 0.4 else ("" if rng.random() < 0.3 else p["phone_raw"])
            id_b = p["ids"]["abha"] if rng.random() < 0.5 else ""
            rows.append([
                f"PAIR-{idx:05d}", 1, "positive_cross_script" if kind == 0 else "positive_alias_noisy",
                name_a, p["relative_en"], p["age"], p["gender"], p["phone_raw"], p["ids"]["abha"], p["district"],
                name_b, p["relative_hi"] if kind == 0 else p["relative_en"], p["age"] + rng.choice([0, 0, 0, 1]), p["gender"], phone_b, id_b, p["district"],
            ])
        elif kind == 2 and homonym_groups:
            # HARD NEGATIVE 1: Exact same name, DIFFERENT real people (Homonym Collision Trap!)
            grp = rng.choice(homonym_groups)
            a, b = rng.sample(grp, 2)
            rows.append([
                f"PAIR-{idx:05d}", 0, "hard_negative_homonym",
                a["full_en"], a["relative_en"], a["age"], a["gender"], a["phone_raw"], a["ids"]["abha"], a["district"],
                b["full_hi"], b["relative_en"], b["age"], b["gender"], b["phone_raw"], b["ids"]["abha"], b["district"],
            ])
        elif kind == 3:
            # HARD NEGATIVE 2: Family members sharing phone & district but different first name/age
            a = rng.choice(patients)
            b = rng.choice([x for x in patients if x["true_patient_id"] != a["true_patient_id"]])
            rows.append([
                f"PAIR-{idx:05d}", 0, "hard_negative_same_household_phone",
                a["full_en"], a["relative_en"], a["age"], a["gender"], a["phone_raw"], a["ids"]["uhid"], a["district"],
                f"{b['first']} {a['last']}", a["relative_en"], (a["age"] + 23) % 85 + 2, b["gender"], a["phone_raw"], b["ids"]["uhid"], a["district"],
            ])
        else:
            # HARD NEGATIVE 3: Weak evidence / random non-match (should NOT link -> label 0)
            a, b = rng.sample(patients, 2)
            rows.append([
                f"PAIR-{idx:05d}", 0, "negative_weak_or_unrelated",
                f"{a['first']} ji", a["relative_en"], "", a["gender"], "", "", "",
                b["full_en"], b["relative_en"], b["age"], b["gender"], b["phone_raw"], b["ids"]["mrn"], b["district"],
            ])

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return len(rows)


# =========================================================================
# PART 3: QUASI-IDENTIFIER PRIVACY BENCHMARK TABLE (For k-anonymity tuning)
# =========================================================================
def generate_qi_benchmark(patients: list[dict], out_path: Path) -> int:
    # Compute exact equivalence class counts on (age, gender, village, diagnosis)
    qi_counts = Counter((p["age"], p["gender"], p["village"], p["diagnosis"]) for p in patients)
    coarse_counts = Counter((f"{(p['age']//10)*10}-{(p['age']//10)*10+9}", p["gender"], p["district"]) for p in patients)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id", "age", "age_band_10y", "gender", "village", "pin", "pin_3prefix",
            "district", "diagnosis", "is_rare_diagnosis", "admit_date", "discharge_date", "los",
            "raw_k_anonymity", "generalized_k_anonymity", "singling_out_risk"
        ])
        for p in patients:
            raw_k = qi_counts[(p["age"], p["gender"], p["village"], p["diagnosis"])]
            band = f"{(p['age']//10)*10}-{(p['age']//10)*10+9}"
            gen_k = coarse_counts[(band, p["gender"], p["district"])]
            risk = "critical" if (p["is_small_village"] and p["is_rare"]) else ("high" if raw_k == 1 else "moderate")
            w.writerow([
                p["true_patient_id"], p["age"], band, p["gender"], p["village"], p["pin"],
                p["pin"][:3] + "XXX", p["district"], p["diagnosis"], int(p["is_rare"]),
                p["admit_date"], p["discharge_date"], p["los"], raw_k, gen_k, risk
            ])
    return len(patients)


# =========================================================================
# PART 4: END-TO-END UNSEEN-DOMAIN PIPELINE EVAL SUITE (560 records + 26 quarantine)
# =========================================================================
def generate_unseen_domain_eval(rng: random.Random, patients: list[dict], out_root: Path) -> dict:
    in_root = out_root / "input"
    if out_root.exists():
        shutil.rmtree(out_root)

    # Note: Folder names & column names intentionally differ from typical practice data
    # to test schema generalization ("names, labels and formats never seen in practice data")!
    for sub in ["sys1_intake", "sys2_triage", "sys3_emr_json", "sys4_fhir_r4", "sys5_ward_notes", "sys6_quarantine_attacks"]:
        (in_root / sub).mkdir(parents=True, exist_ok=True)

    truth = {
        "meta": {
            "seed": SEED,
            "description": "Unseen-Domain Adversarial & Security Evaluation Suite (560 records + 26 quarantine files)",
        },
        "records": [],
        "canaries": [],
        "expected_quarantine": [],
        "filename_pii_files": [],
        "homonym_pairs": [],
    }

    by_name: dict[str, list[str]] = {}
    for p in patients:
        by_name.setdefault(p["full_en"], []).append(p["true_patient_id"])
    for name, pids in by_name.items():
        if len(pids) > 1:
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    truth["homonym_pairs"].append([pids[i], pids[j], name])

    # 1. SYS1 — Intake CSV (160 rows, UNSEEN headers + formula/int/null-byte injections) + Dirty BOM CSV (50 rows)
    csv1 = in_root / "sys1_intake" / "intake_register_utf8.csv"
    headers1 = [
        "reg_no", "mriz_ka_naam", "attendant_rel", "umar_yrs", "sex_cd", "mob_contact",
        "epic_ref", "health_id_abha", "niwas_sthan", "dak_pin", "bharti_tarikh",
        "chhutti_tarikh", "rog_diagnosis", "upchar_rx", "vital_readings", "chikitsak", "tippani"
    ]
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    w.writerow(headers1)
    for r_i in range(160):
        p = patients[r_i % len(patients)]
        tags = ["unseen_hinglish_headers", "csv_utf8"]
        age_val = str(p["age"])
        remarks = f"C/O fever and chills; {p['eponym']}; HbA1c {p['hba1c']}"
        if r_i < 12:
            remarks = f"{CSV_FORMULA_PAYLOADS[r_i % len(CSV_FORMULA_PAYLOADS)]} | {remarks}"
            tags.append("csv_formula_injection")
        if 12 <= r_i < 24:
            age_val = INTEGER_BOUNDARY_VALUES[r_i - 12]
            tags.append("integer_boundary_injection")
        if 24 <= r_i < 32:
            remarks = f"{remarks} | audit: {SANITIZATION_INJECTIONS[(r_i - 24) % len(SANITIZATION_INJECTIONS)]}"
            tags.append("input_sanitization_injection")
        if 32 <= r_i < 38:
            remarks = f"Ref: {p['relative_en']}\x00{p['phone_digits']} | {remarks}"
            tags.append("null_byte_injection")
        if p["canary"]:
            remarks += f" | REF:{p['canary']}"
            truth["canaries"].append(p["canary"])

        name_cell = p["full_hi"] if r_i % 4 == 0 else (rng.choice(p["aliases"]) if r_i % 3 == 0 else p["full_en"])
        rel_cell = f"{p['relative_prefix']} {p['relative_hi'] if r_i % 4 == 0 else p['relative_en']}"
        w.writerow([
            p["ids"]["uhid"], name_cell, rel_cell, age_val, p["gender"],
            p["phone_raw"], p["ids"]["voter_epic"], p["ids"]["abha"],
            f"Mohalla 2, {p['village']}, {p['district']}", p["pin"],
            p["admit_date"], p["discharge_date"], p["diagnosis"],
            p["med"], f"BP {p['bp']} mmHg", p["doctor"], remarks
        ])
        truth["records"].append({
            "rel_path": "sys1_intake/intake_register_utf8.csv",
            "loc": f"row:{r_i}",
            "true_patient_id": p["true_patient_id"],
            "pii_must_redact": [name_cell, p["full_en"], p["relative_en"], p["phone_digits"], p["ids"]["voter_epic"], p["ids"]["abha"], p["ids"]["uhid"]],
            "clinical_must_keep": [p["diagnosis"], p["med"], p["bp"], p["hba1c"], p["doctor"], p["eponym"]],
            "true_age": p["age"] if "integer_boundary_injection" not in tags else None,
            "true_los": p["los"],
            "security_tags": tags,
        })
    csv1.write_text(buf.getvalue(), encoding="utf-8")

    # Dirty BOM CSV (50 rows): PII inside a CSV header + ZWSP + Devanagari digits + shifted cols
    csv2 = in_root / "sys1_intake" / "legacy_bom_shifted.csv"
    lines = ["\ufeffmrn_code,naam_zwsp,phone_dev,abha_dev,gram_address,admit_dt,disch_dt,dx,rx_plan,consultant"]
    for idx in range(50):
        p = patients[(idx * 3) % len(patients)]
        tags = ["utf8_bom", "zwsp_pii_evasion", "devanagari_digits"]
        ev_name = inject_zwsp(p["full_en"])
        dev_ph = p["phone_digits"].translate(DEVANAGARI_DIGITS)
        dev_ab = p["ids"]["abha"].translate(DEVANAGARI_DIGITS)
        loc_cell = f"{p['village']}, Near Chowk" if idx % 6 == 0 else p["village"]
        if idx % 6 == 0:
            tags.append("shifted_csv_column")
        lines.append(
            f"{p['ids']['mrn']},{ev_name},{dev_ph},{dev_ab},{loc_cell},"
            f"{p['admit_date']},{p['discharge_date']},{p['diagnosis']},{p['med']},{p['doctor']}"
        )
        truth["records"].append({
            "rel_path": "sys1_intake/legacy_bom_shifted.csv",
            "loc": f"row:{idx}",
            "true_patient_id": p["true_patient_id"],
            "pii_must_redact": [p["full_en"], ev_name, p["phone_digits"], dev_ph, dev_ab, p["ids"]["mrn"]],
            "clinical_must_keep": [p["diagnosis"], p["med"], p["doctor"]],
            "true_age": p["age"],
            "true_los": p["los"],
            "security_tags": tags,
        })
    csv2.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. SYS2 — Legacy CP1252 Triage TSV (100 rows)
    tsv_path = in_root / "sys2_triage" / "triage_cp1252.tsv"
    tsv_lines = ["case_mrn\tpat_alias\tkin_info\tage_val\tcontact_num\tseen_on\tsymptoms\tmed_order\tseen_by"]
    for idx in range(100):
        p = patients[(idx + 17) % len(patients)]
        tags = ["legacy_cp1252_tsv", "meaning_trap"]
        alias = rng.choice(p["aliases"])
        ph_val, _ = synth_phone(rng, 3 if idx % 5 == 0 else 1)
        if idx % 5 == 0:
            tags.append("scientific_notation_phone")
        age_str = str(p["age"]) if idx % 14 != 0 else str(INTEGER_BOUNDARY_VALUES[idx % len(INTEGER_BOUNDARY_VALUES)])
        if idx % 14 == 0:
            tags.append("integer_boundary_injection")
        comp = f"C/O fever \u2013 4 days; {p['eponym'].replace('é','e')}; BP {p['bp']}"
        kin = f"C/O {p['relative_en']} (PAN: {p['ids']['pan_like']})"
        tsv_lines.append(f"{p['ids']['mrn']}\t{alias}\t{kin}\t{age_str}\t{ph_val}\t{p['admit_date']}\t{comp}\t{p['med']}\t{p['doctor']}")
        truth["records"].append({
            "rel_path": "sys2_triage/triage_cp1252.tsv",
            "loc": f"row:{idx}",
            "true_patient_id": p["true_patient_id"],
            "pii_must_redact": [alias, p["relative_en"], p["ids"]["pan_like"], p["ids"]["mrn"]],
            "clinical_must_keep": ["C/O fever", p["med"], p["bp"], p["doctor"]],
            "true_age": p["age"] if "integer_boundary_injection" not in tags else None,
            "true_los": None,
            "security_tags": tags,
        })
    tsv_path.write_bytes(("\n".join(tsv_lines) + "\n").encode("cp1252", errors="replace"))

    # 3. SYS3 — Nested EMR JSON (110 records, unseen top-level key 'encounters' + PII in keys + Bidi + 120KB field)
    json_path = in_root / "sys3_emr_json" / "emr_encounters.json"
    encounters = []
    for idx in range(110):
        p = patients[(idx * 2) % len(patients)]
        tags = ["nested_json", "unseen_json_schema"]
        ext = {}
        extra_pii = []
        if idx % 5 == 0:
            k_leak = f"patient_{p['first']}_{p['last']}_audit"
            ext[k_leak] = {"pmjay": p["ids"]["insurance_id"], "verified": True}
            extra_pii.extend([k_leak, p["ids"]["insurance_id"]])
            tags.append("pii_in_json_key")

        note_txt = f"{p['full_hi']} ({p['age']}{p['gender']}) को {p['diagnosis']} है। {p['med']} चालू करें। BP {p['bp']}."
        if idx % 6 == 0:
            bidi = f"\u202e{p['phone_digits'][::-1]}\u202c"
            note_txt += f" Ph: {bidi}"
            extra_pii.append(bidi)
            tags.append("bidi_rlo_evasion")
        if idx % 9 == 0:
            note_txt += f" | note: {SANITIZATION_INJECTIONS[idx % len(SANITIZATION_INJECTIONS)]}"
            tags.append("input_sanitization_injection")
        if idx == 42:
            c_big = "CANARY-BIGFIELD-9941"
            truth["canaries"].append(c_big)
            note_txt += " " + ("CLINICAL_PAD " * 4000) + f" {c_big} " + ("TAIL_PAD " * 3000)
            extra_pii.append(c_big)
            tags.append("large_buffer_field")

        encounters.append({
            "enc_uid": f"ENC-{idx:04d}",
            "demographics": {
                "full_name": p["full_en"] if idx % 2 == 0 else p["full_hi"],
                "kin": f"{p['relative_prefix']} {p['relative_en']}",
                "age_years": p["age"],
                "gender_code": p["gender"],
                "comms": {"tel": p["phone_raw"], "mail": p["email"]},
                "geo": {"gram": p["village"], "pincode": p["pin"], "zila": p["district"]},
                "ext_ids": {"abha_no": p["ids"]["abha"], "uhid_no": p["ids"]["uhid"]},
            },
            "clinical": {
                "doa": p["admit_date"],
                "dod": p["discharge_date"],
                "dx_text": p["diagnosis"],
                "rx_list": [p["med"]],
                "obs": {"bp": p["bp"], "hba1c": p["hba1c"]},
                "physician": p["doctor"],
                "ward_note": note_txt,
            },
            "custom_meta": ext,
        })
        truth["records"].append({
            "rel_path": "sys3_emr_json/emr_encounters.json",
            "loc": f"/encounters/{idx}",
            "true_patient_id": p["true_patient_id"],
            "pii_must_redact": [p["full_en"], p["full_hi"], p["email"], p["ids"]["abha"], p["ids"]["uhid"]] + extra_pii,
            "clinical_must_keep": [p["diagnosis"], p["med"], p["bp"], p["hba1c"], p["doctor"]],
            "true_age": p["age"],
            "true_los": p["los"],
            "security_tags": tags,
        })
    json_path.write_text(json.dumps({"encounters": encounters}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. SYS4 — FHIR R4 Bundles (60 files)
    for idx in range(60):
        p = patients[(idx * 2 + 1) % len(patients)]
        tags = ["fhir_r4"]
        fname = f"fhir_bundle_{idx:03d}.json"
        fpath = in_root / "sys4_fhir_r4" / fname
        extra_pii = []
        attach_txt = f"Ward slip for {p['full_en']} Ph:{p['phone_digits']} ABHA:{p['ids']['abha']}"
        if idx % 4 == 0:
            tags.append("fhir_base64_attachment_leak")
            extra_pii.append(p["phone_digits"])
        b64_data = base64.b64encode(attach_txt.encode("utf-8")).decode("ascii")
        bundle = {
            "resourceType": "Bundle",
            "id": f"yp-r4-{idx:03d}",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": f"p-{idx:03d}",
                        "text": {
                            "status": "generated",
                            "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">{p['full_hi']} ({p['relative_prefix']} {p['relative_en']}), {p['village']} {p['pin']}</div>"
                        },
                        "identifier": [
                            {"system": "https://healthid.ndhm.gov.in", "value": p["ids"]["abha"]},
                            {"system": "urn:yp:uhid", "value": p["ids"]["uhid"]},
                        ],
                        "name": [{"text": p["full_en"], "family": p["last"], "given": [p["first"]]}],
                        "telecom": [{"system": "phone", "value": p["phone_raw"]}],
                        "gender": "male" if p["gender"] == "M" else "female",
                        "birthDate": f"{2025 - p['age']}-06-15",
                        "address": [{"city": p["village"], "district": p["district"], "postalCode": p["pin"]}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Encounter",
                        "id": f"e-{idx:03d}",
                        "status": "finished",
                        "period": {"start": p["admit_date"], "end": p["discharge_date"]},
                        "participant": [{"individual": {"display": p["doctor"]}}],
                    }
                },
                {"resource": {"resourceType": "Condition", "id": f"c-{idx:03d}", "code": {"text": p["diagnosis"]}}},
                {"resource": {"resourceType": "MedicationRequest", "id": f"m-{idx:03d}", "status": "active", "intent": "order", "medicationCodeableConcept": {"text": p["med"]}}},
                {"resource": {"resourceType": "DocumentReference", "id": f"d-{idx:03d}", "status": "current", "content": [{"attachment": {"contentType": "text/plain", "data": b64_data}}]}},
            ],
        }
        fpath.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        truth["records"].append({
            "rel_path": f"sys4_fhir_r4/{fname}",
            "loc": "doc",
            "true_patient_id": p["true_patient_id"],
            "pii_must_redact": [p["full_en"], p["full_hi"], p["relative_en"], p["ids"]["abha"], p["ids"]["uhid"]] + extra_pii,
            "clinical_must_keep": [p["diagnosis"], p["med"], p["doctor"]],
            "true_age": p["age"],
            "true_los": p["los"],
            "security_tags": tags,
        })

    # 5. SYS5 — Multilingual Clinical Notes (80 TXT files: 12 PII filenames + 10 weak links)
    for idx in range(80):
        p = patients[(idx + 40) % len(patients)]
        tags = ["free_text_multilingual"]
        if idx < 12:
            fname = f"summary_{p['first']}_{p['last']}_{idx:03d}.txt"
            tags.append("pii_in_filename")
            truth["filename_pii_files"].append(f"sys5_ward_notes/{fname}")
        else:
            fname = f"ward_note_{idx:03d}.txt"
        fpath = in_root / "sys5_ward_notes" / fname
        is_weak = (60 <= idx < 70)
        if is_weak:
            tags.append("weak_identity_do_not_overlink")
            note = (
                f"OPD Slip\nPatient: {p['first']} ji, {p['relative_prefix']} {p['relative_en']}\n"
                f"C/O fever and cough. Rx: {p['med']}. BP {p['bp']}.\nSeen by {p['doctor']}.\n"
            )
            true_pid = None
            pii_list = [p["first"], p["relative_en"]]
        else:
            canary_line = ""
            if idx % 13 == 0:
                c_tok = f"CANARY-NOTE-{idx:03d}-LEAK"
                truth["canaries"].append(c_tok)
                canary_line = f"\nAudit Token: {c_tok}"
            note = (
                f"DISTRICT HOSPITAL — DISCHARGE SUMMARY\n"
                f"मरीज़ / Patient: {p['full_hi']} ({rng.choice(p['aliases'])})\n"
                f"Kin: {p['relative_prefix']} {p['relative_en']} | Mob: {p['phone_raw']}\n"
                f"UHID: {p['ids']['uhid']} | ABHA: {p['ids']['abha']} | Email: {p['email']}\n"
                f"Address: {p['village']}, PIN {p['pin']}, {p['district']} | Age/Sex: {p['age']}Y/{p['gender']}\n"
                f"DOA: {p['admit_date']} | DOD: {p['discharge_date']}\n"
                f"Chief Complaint: C/O high fever, {p['diagnosis']} (r/o {p['eponym']}).\n"
                f"Vitals: BP {p['bp']} mmHg, HbA1c {p['hba1c']}.\n"
                f"Rx (दवाइयाँ): {p['med']} — खाना खाने के बाद।\n"
                f"Consultant: {p['doctor']}{canary_line}\n"
            )
            true_pid = p["true_patient_id"]
            pii_list = [p["full_hi"], p["relative_en"], p["phone_digits"], p["ids"]["uhid"], p["ids"]["abha"], p["email"]]

        fpath.write_text(note, encoding="utf-8")
        truth["records"].append({
            "rel_path": f"sys5_ward_notes/{fname}",
            "loc": "doc",
            "true_patient_id": true_pid,
            "pii_must_redact": pii_list,
            "clinical_must_keep": [p["med"], p["bp"], p["doctor"]],
            "true_age": None if is_weak else p["age"],
            "true_los": None if is_weak else p["los"],
            "security_tags": tags,
        })

    # 6. SYS6 — Adversarial Security & Quarantine Attacks (26 files)
    q_dir = in_root / "sys6_quarantine_attacks"

    def add_q(name: str, data: bytes, reason: str, tag: str, canary: str | None = None) -> None:
        fp = q_dir / name
        fp.write_bytes(data)
        if canary:
            truth["canaries"].append(canary)
        truth["expected_quarantine"].append({
            "rel_path": f"sys6_quarantine_attacks/{name}",
            "input_sha256": sha256_file(fp),
            "expected_reason": reason,
            "security_tag": tag,
        })

    add_q("huge_single_line_overflow.txt", (("A" * 750_000) + " Patient: Mohammad Arif CANARY-BUF-7712 " + ("B" * 750_000)).encode(), "oversized_line_buffer", "buffer_overflow_stress", "CANARY-BUF-7712")
    add_q("redos_phone_backtrack.txt", (f"Phone: {'9'*45000}! Name: {'Dr. ' + 'a '*20000} CANARY-REDOS-8821").encode(), "redos_backtracking_bomb", "redos_backtracking_bomb", "CANARY-REDOS-8821")
    add_q("deep_nesting_bomb.json", (('{"a":' * 1200) + '"CANARY-DEEP-3319"' + ('}' * 1200)).encode(), "json_depth_exceeded", "json_depth_overflow", "CANARY-DEEP-3319")
    for i in range(4):
        c = f"CANARY-TRUNC-{i}-5501"
        add_q(f"truncated_fhir_{i}.json", f'{{"resourceType":"Bundle","patient":"Sunita Devi {c}","entry":['.encode(), "truncated_json", "truncated_json", c)
    for i in range(4):
        c = f"CANARY-BIN-{i}-4409"
        add_q(f"corrupt_binary_{i}.csv", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + c.encode() + b"\xff\xfe\xfd\x00", "binary_disguised_as_csv", "invalid_binary_as_csv", c)
    for uname, ubytes, ureason, ucan in [
        ("scan_Ramesh.pdf", b"%PDF-1.7\n%CANARY-PDF-1001\n%%EOF", "unsupported_pdf", "CANARY-PDF-1001"),
        ("export.xlsx", b"PK\x03\x04FAKE_XLSX_CANARY-XLSX-1002", "unsupported_xlsx", "CANARY-XLSX-1002"),
        ("xray.jpg", b"\xff\xd8\xff\xe0EXIF_CANARY-JPG-1003", "unsupported_image", "CANARY-JPG-1003"),
        ("dump.zip", b"PK\x03\x04ZIP_CANARY-ZIP-1004", "unsupported_archive", "CANARY-ZIP-1004"),
        ("ward.db", b"SQLite format 3\x00CANARY-DB-1005", "unsupported_db", "CANARY-DB-1005"),
        ("letter.docx", b"PK\x03\x04DOCX_CANARY-DOCX-1006", "unsupported_docx", "CANARY-DOCX-1006"),
    ]:
        add_q(uname, ubytes, ureason, "unsupported_format", ucan)
    add_q("empty_zero.csv", b"", "empty_file", "zero_byte_file")
    add_q("null_flood.txt", b"\x00" * 4096, "null_byte_stream", "null_byte_flood")
    add_q(".._.._etc_passwd_Arif.txt", b"Patient: Mohammad Arif, CANARY-TRAV-2001", "path_traversal_name", "path_traversal_filename", "CANARY-TRAV-2001")
    add_q("note;rm -rf tmp;.csv", b"name,age\nSunita Devi,47\nCANARY-SHELL-2002,10\n", "shell_metachar_name", "shell_metachar_filename", "CANARY-SHELL-2002")
    add_q("$(whoami)_opd.json", b'{"invalid": true, "leak": "CANARY-CMD-2003"', "cmd_sub_malformed", "cmd_sub_filename", "CANARY-CMD-2003")
    add_q("broken_utf16_surrogate.txt", b"\xff\xfe" + "Patient: Rajesh Kumar CANARY-UTF16-9001".encode("utf-16le") + b"\x00\xd8", "corrupt_utf16", "encoding_surrogate_corruption", "CANARY-UTF16-9001")
    add_q("json_nan_infinity.json", b'{"encounters":[{"age":NaN,"los":Infinity,"patient":"CANARY-NAN-7001"}]}', "nonstandard_json_nan", "json_integer_nan_injection", "CANARY-NAN-7001")
    add_q("xml_xxe_payload.xml", b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><note>CANARY-XXE-6001 &xxe;</note>', "unsupported_xml_xxe", "xxe_unsupported_xml", "CANARY-XXE-6001")
    add_q("huge_int_json_overflow.json", b'{"encounters":[{"age":' + (b"9" * 5000) + b',"patient":"CANARY-BIGINT-6002"}]}', "int_str_digits_limit", "integer_str_dos_overflow", "CANARY-BIGINT-6002")

    # 7. Attacker Knowledge
    attacker_known = {
        "description": "External voter register + known public-health facts for linkage attack simulation.",
        "targets": [
            {
                "true_patient_id": p["true_patient_id"],
                "voter_name": p["full_en"],
                "known_age": p["age"],
                "known_gender": p["gender"],
                "known_village": p["village"],
                "known_pin": p["pin"],
                "known_district": p["district"],
                "known_diagnosis": p["diagnosis"],
                "known_los": p["los"],
                "known_admit_year": int(p["admit_date"][:4]),
                "risk_tier": "critical" if (p["is_small_village"] and p["is_rare"]) else "high",
            }
            for p in patients if p["is_small_village"] or p["is_rare"]
        ],
    }

    (out_root / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "attacker_known.json").write_text(json.dumps(attacker_known, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input_files": sum(1 for p in in_root.rglob("*") if p.is_file()),
        "records": len(truth["records"]),
        "canaries": len(set(truth["canaries"])),
        "quarantines": len(truth["expected_quarantine"]),
        "targets": len(attacker_known["targets"]),
        "homonym_pairs": len(truth["homonym_pairs"]),
    }


def main() -> None:
    rng = random.Random(SEED)

    train_dir = Path("training_data")
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)

    # Disjoint patient pools: TRAIN uses TRAIN_* pools; VAL & UNSEEN_EVAL use UNSEEN_* pools
    train_patients = build_patient_pool(rng, 240, TRAIN_FIRST_M, TRAIN_FIRST_F, TRAIN_LAST, TRAIN_VILLAGES, "TR")
    val_patients = build_patient_pool(rng, 100, UNSEEN_FIRST_M[:8], UNSEEN_FIRST_F[:8], UNSEEN_LAST[:8], UNSEEN_VILLAGES[:4], "VA")
    eval_patients = build_patient_pool(rng, 140, UNSEEN_FIRST_M[8:], UNSEEN_FIRST_F[8:], UNSEEN_LAST[8:], UNSEEN_VILLAGES[4:] + UNSEEN_VILLAGES[:4], "EV")
    qi_patients = build_patient_pool(rng, 600, TRAIN_FIRST_M + UNSEEN_FIRST_M, TRAIN_FIRST_F + UNSEEN_FIRST_F, TRAIN_LAST + UNSEEN_LAST, TRAIN_VILLAGES + UNSEEN_VILLAGES, "QI")

    n_ner_tr, n_ner_va = generate_ner_splits(rng, train_patients, val_patients, train_dir)
    n_lnk_tr = generate_linkage_pairs(rng, train_patients, 3000, train_dir / "linkage_pairs_train.csv")
    n_lnk_va = generate_linkage_pairs(rng, val_patients, 800, train_dir / "linkage_pairs_val.csv")
    n_qi = generate_qi_benchmark(qi_patients, train_dir / "qi_privacy_benchmark.csv")
    eval_stats = generate_unseen_domain_eval(rng, eval_patients, Path("dev_unseen_domain"))

    print("=" * 68)
    print("  SECRET WARDS — TRAINING & ADVERSARIAL EVAL SUITE GENERATED")
    print("=" * 68)
    print(f"  [TRAIN] NER BIO train examples   : {n_ner_tr}  (training_data/ner_bio_train.jsonl)")
    print(f"  [VAL]   NER BIO val examples     : {n_ner_va}   (training_data/ner_bio_val.jsonl)")
    print(f"  [TRAIN] Linkage pairs            : {n_lnk_tr}  (training_data/linkage_pairs_train.csv)")
    print(f"  [VAL]   Linkage pairs            : {n_lnk_va}   (training_data/linkage_pairs_val.csv)")
    print(f"  [BENCH] QI privacy tuples        : {n_qi}   (training_data/qi_privacy_benchmark.csv)")
    print("-" * 68)
    print(f"  [EVAL]  Unseen-domain files      : {eval_stats['input_files']}   (dev_unseen_domain/input/)")
    print(f"  [EVAL]  Unseen-domain records    : {eval_stats['records']}")
    print(f"  [EVAL]  Planted canaries         : {eval_stats['canaries']}")
    print(f"  [EVAL]  Adversarial quarantines  : {eval_stats['quarantines']}")
    print(f"  [EVAL]  Voter-attack targets     : {eval_stats['targets']}")
    print(f"  [EVAL]  Homonym collision pairs  : {eval_stats['homonym_pairs']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
