"""
03_cosine_filter.py
Cosine dedup + per-vector thresholds + coverage report over all 32 taxonomy vectors.

Threshold strategy:
  - Large real vectors (>2000 rows): 0.65  — hammer near-duplicates hard
  - Medium real vectors (500–2000): 0.72
  - Synthetic vectors: 0.85              — more lenient (varied templates need to survive)
  - Default: 0.80
"""

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
BATCH_SIZE = 512
PRE_CAP = 8000   # max rows fed into the O(n²) loop per vector (speed guard)

# ── Per-vector thresholds (lower = stricter = more unique = fewer kept) ──────────
# Large real vectors that dominate without tightening:
TIGHT = {
    "Direct instruction override":               0.65,   # 4,856 raw → needs hard dedup
    "Indirect injection via context":             0.65,   # 2,577
    "Misinformation and hallucination manipulation": 0.65,  # 5,941
    "Toxic and abusive content":                  0.65,   # 7,270 (off-taxonomy but present)
    "Summarisation inference":                    0.68,   # 3,160 + weak labels
    "Cross-site scripting (XSS) generation":      0.70,   # 2,563
    "Encoded and obfuscated payload delivery":    0.70,   # 1,254
    "Role and persona assignment":                0.72,   # 1,143
    "Malicious URL generation":                   0.72,   # 1,347
}
# Synthetic vectors — more lenient so generated diversity survives
SYNTHETIC_THRESHOLD = 0.85
SYNTHETIC_VECTORS = {
    "Second-hop and forwarded injection",
    "Encoded and obfuscated injection",
    "Repetition and verbatim requests",
    "Format conversion extraction",
    "Completion-based extraction",
    "Behavioural inference and side-channel probing",
    "Direct questioning",
    "Jailbreak via code context",
    "Prompt-embedded payload delivery",
    "Cross-customer identity leakage (BOLA)",
    "Authentication credential extraction",
    "HTML and JavaScript injection in output",
    "Data exfiltration via output formatting",
    "Sycophantic manipulation and false premise confirmation",
    "Sponge attacks",
    "Recursive elaboration attacks",
    "Context window flooding",
    "Training data extraction",
    "Model inversion and fine-tuning probing",
    "Adversarial knowledge boundary exploitation",
    "Image-embedded instruction injection",
    "Document structure injection",
}
DEFAULT_THRESHOLD = 0.80

# ── Full taxonomy (32 vectors) ────────────────────────────────────────────────────
TAXONOMY = {
    "Prompt Injection": [
        "Direct instruction override",
        "Role and persona assignment",
        "Indirect injection via context",
        "Encoded and obfuscated injection",
        "Second-hop and forwarded injection",
    ],
    "System Prompt Extraction": [
        "Direct questioning",
        "Repetition and verbatim requests",
        "Format conversion extraction",
        "Summarisation inference",
        "Completion-based extraction",
        "Behavioural inference and side-channel probing",
    ],
    "Malicious Code and Payload Injection": [
        "Cross-site scripting (XSS) generation",
        "SQL injection payload generation",
        "Command injection payload generation",
        "Jailbreak via code context",
        "Prompt-embedded payload delivery",
        "Encoded and obfuscated payload delivery",
    ],
    "Privilege and Identity Attacks": [
        "Cross-customer identity leakage (BOLA)",
        "Authentication credential extraction",
    ],
    "Malicious Content in Output": [
        "HTML and JavaScript injection in output",
        "Malicious URL generation",
        "Data exfiltration via output formatting",
        "Misinformation and hallucination manipulation",
        "Sycophantic manipulation and false premise confirmation",
    ],
    "Denial of Service and Resource Exhaustion": [
        "Sponge attacks",
        "Recursive elaboration attacks",
        "Context window flooding",
    ],
    "Training Data and Knowledge Attacks": [
        "Training data extraction",
        "Model inversion and fine-tuning probing",
        "Adversarial knowledge boundary exploitation",
    ],
    "Multi-modal and Cross-modal Injection": [
        "Image-embedded instruction injection",
        "Document structure injection",
    ],
}

print("Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded\n")


def get_threshold(vector):
    if vector in TIGHT:
        return TIGHT[vector]
    if vector in SYNTHETIC_VECTORS:
        return SYNTHETIC_THRESHOLD
    return DEFAULT_THRESHOLD


def cosine_dedup(df):
    kept_frames = []
    for (parent, vector), chunk in df.groupby(["parent_category", "vector"], sort=False):
        chunk = chunk.reset_index(drop=True)
        thr = get_threshold(vector)

        # Pre-cap to avoid O(n²) blowup on very large vectors
        if len(chunk) > PRE_CAP:
            chunk = chunk.sample(PRE_CAP, random_state=42).reset_index(drop=True)

        n = len(chunk)
        print(f"  {vector[:38]:38s}  n={n:6d}  thr={thr}", flush=True)

        if n <= 1:
            kept_frames.append(chunk)
            continue

        emb = model.encode(
            chunk["prompt"].astype(str).tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        buf = np.empty((n, emb.shape[1]), dtype=np.float32)
        keep_idx, k = [], 0
        for i in range(n):
            if k == 0 or (buf[:k] @ emb[i]).max() < thr:
                buf[k] = emb[i]
                k += 1
                keep_idx.append(i)

        kept = chunk.iloc[keep_idx]
        print(f"    → kept {len(kept)}", flush=True)
        kept_frames.append(kept)

    return pd.concat(kept_frames, ignore_index=True)


def process(name, raw_path, out_path, apply_dedup=True):
    if not raw_path.exists():
        print(f"❌ {raw_path.name} not found")
        return None
    print(f"\n{'='*70}")
    print(f"PROCESSING: {name}")
    print(f"{'='*70}")
    df = pd.read_csv(raw_path)
    df = df[df["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
    print(f"Loaded {len(df)} rows (after exact-dedup)")
    if apply_dedup:
        df = cosine_dedup(df)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ {len(df)} rows → {out_path}")
    return df


# ── MERGE: real attack data + synthetic ──────────────────────────────────────────
print("Merging attack + synthetic sources...")
frames = []
for fname in ["attack_prompts_raw.csv", "synthetic_prompts.csv"]:
    p = OUTPUT_DIR / fname
    if p.exists():
        f = pd.read_csv(p)
        print(f"  {fname}: {len(f)} rows")
        frames.append(f)
    else:
        print(f"  ⚠️  {fname} not found (skipped)")

attack_merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
attack_merged = attack_merged[attack_merged["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print(f"Merged attack rows before dedup: {len(attack_merged)}\n")

# Save merged raw for reference
attack_merged.to_csv(OUTPUT_DIR / "attack_merged_raw.csv", index=False, quoting=csv.QUOTE_ALL)

# ── DEDUP ─────────────────────────────────────────────────────────────────────────
attack_merged.to_csv(OUTPUT_DIR / "_tmp_attack_merged.csv", index=False, quoting=csv.QUOTE_ALL)
attack = process("ATTACK (merged + deduped)", OUTPUT_DIR / "_tmp_attack_merged.csv", OUTPUT_DIR / "attack_prompts.csv")

# Benign — exact dedup only (no cosine, it's already diverse instruction text)
print("\n" + "="*70)
print("BENIGN — exact dedup only")
benign_path = OUTPUT_DIR / "benign_prompts_raw.csv"
if benign_path.exists():
    benign = pd.read_csv(benign_path)
    benign = benign[benign["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
    benign.to_csv(OUTPUT_DIR / "benign_prompts.csv", index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ {len(benign)} benign rows")
else:
    benign = pd.DataFrame()
    print("⚠️  benign_prompts_raw.csv not found")

# ── FINAL COMBINED ────────────────────────────────────────────────────────────────
combined_parts = [p for p in [attack, benign] if p is not None and len(p) > 0]
combined = pd.concat(combined_parts, ignore_index=True)
combined.to_csv(OUTPUT_DIR / "velyana_dataset_final.csv", index=False, quoting=csv.QUOTE_ALL)

# Cleanup temp
(OUTPUT_DIR / "_tmp_attack_merged.csv").unlink(missing_ok=True)

# ── COVERAGE REPORT ───────────────────────────────────────────────────────────────
present = combined["vector"].value_counts().to_dict()
report_rows = []
for cat, vecs in TAXONOMY.items():
    for v in vecs:
        cnt = present.get(v, 0)
        report_rows.append({
            "parent_category": cat,
            "vector": v,
            "rows": cnt if cnt else "N/A",
            "source": "synthetic" if v in SYNTHETIC_VECTORS and cnt > 0 else ("real" if cnt > 0 else "N/A"),
        })
pd.DataFrame(report_rows).to_csv(OUTPUT_DIR / "coverage_report.csv", index=False)

# ── SUMMARY PRINT ─────────────────────────────────────────────────────────────────
real_rows = (combined.get("source", pd.Series(dtype=str)) != "synthetic_llama3.2").sum() if "source" in combined.columns else len(combined)
syn_rows = len(combined) - real_rows if "source" in combined.columns else 0

print(f"\n{'='*70}")
print(f"FINAL DATASET: {len(combined)} rows")
print(f"  Attack:  {len(attack) if attack is not None else 0}")
print(f"  Benign:  {len(benign) if len(benign) > 0 else 0}")
covered = sum(1 for r in report_rows if r["rows"] != "N/A")
print(f"  Coverage: {covered}/32 vectors present, {32-covered}/32 N/A\n")

print(f"{'='*70}")
print("COVERAGE REPORT (all 32 taxonomy vectors):")
print(f"{'='*70}")
for cat, vecs in TAXONOMY.items():
    print(f"\n{cat}:")
    for v in vecs:
        cnt = present.get(v, 0)
        source_tag = "(synthetic)" if v in SYNTHETIC_VECTORS and cnt > 0 else ""
        print(f"  {'✅' if cnt else '❌'} {v[:50]:50s} {cnt if cnt else 'N/A':>6}  {source_tag}")

print(f"\n{'='*70}")
print("PER-VECTOR DISTRIBUTION (sorted by count):")
for v, cnt in sorted(present.items(), key=lambda x: -x[1]):
    bar = "█" * min(40, cnt // 100)
    print(f"  {v[:42]:42s} {cnt:6d}  {bar}")
