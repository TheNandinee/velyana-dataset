import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output"); BATCH_SIZE = 256
MAX_PER_VECTOR = 1500; MIN_PER_VECTOR = 100
THRESHOLDS = {"Prompt Injection": 0.65, "Safety": 0.70}
DEFAULT_THRESHOLD = 0.80

# Full Velyana taxonomy, categories 1-5 (24 vectors) — used for the N/A coverage report
TAXONOMY = {
 "Prompt Injection": ["Direct instruction override","Role and persona assignment","Indirect injection via context","Encoded and obfuscated injection","Second-hop and forwarded injection"],
 "System Prompt Extraction": ["Direct questioning","Repetition and verbatim requests","Format conversion extraction","Summarisation inference","Completion-based extraction","Behavioural inference and side-channel probing"],
 "Malicious Code and Payload Injection": ["Cross-site scripting (XSS) generation","SQL injection payload generation","Command injection payload generation","Jailbreak via code context","Prompt-embedded payload delivery","Encoded and obfuscated payload delivery"],
 "Privilege and Identity Attacks": ["Cross-customer identity leakage (BOLA)","Authentication credential extraction"],
 "Malicious Content in Output": ["HTML and JavaScript injection in output","Malicious URL generation","Data exfiltration via output formatting","Misinformation and hallucination manipulation","Sycophantic manipulation and false premise confirmation"],
}

print("Loading model..."); model = SentenceTransformer("all-MiniLM-L6-v2"); print("✅\n")

def cosine_dedup(df):
    kept = []
    for (parent, vector), chunk in df.groupby(["parent_category","vector"], sort=False):
        chunk = chunk.reset_index(drop=True); n = len(chunk)
        thr = THRESHOLDS.get(parent, DEFAULT_THRESHOLD)
        if n <= 1: kept.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(), batch_size=BATCH_SIZE,
                           show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        buf = np.empty((n, emb.shape[1]), dtype=np.float32); idx=[]; k=0
        for i in range(n):
            if k == 0 or (buf[:k] @ emb[i]).max() < thr:
                buf[k] = emb[i]; k += 1; idx.append(i)
        out = chunk.iloc[idx]; print(f"  dedup {vector[:34]:34s} {n:5d}→{len(out):5d}"); kept.append(out)
    return pd.concat(kept, ignore_index=True)

def balance(df):
    out = []
    for vec, g in df.groupby("vector"):
        if vec == "Safe": out.append(g); continue
        if len(g) < MIN_PER_VECTOR:
            print(f"  ⚠️  dropped '{vec}' — {len(g)} rows (<{MIN_PER_VECTOR}, will show N/A)"); continue
        out.append(g.sample(MAX_PER_VECTOR, random_state=42) if len(g) > MAX_PER_VECTOR else g)
    return pd.concat(out, ignore_index=True)

# ---- REAL data only ----
attack = pd.read_csv(OUTPUT_DIR/"attack_prompts_raw.csv")
attack = attack[attack["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("=== ATTACK (real only) ==="); attack = balance(cosine_dedup(attack))
attack.to_csv(OUTPUT_DIR/"attack_prompts.csv", index=False)

benign = pd.read_csv(OUTPUT_DIR/"benign_prompts_raw.csv")
benign = benign[benign["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("\n=== BENIGN ==="); benign = cosine_dedup(benign)
benign.to_csv(OUTPUT_DIR/"benign_prompts.csv", index=False)

combined = pd.concat([attack, benign], ignore_index=True)
combined.to_csv(OUTPUT_DIR/"velyana_dataset_final.csv", index=False)

# ---- COVERAGE REPORT: every taxonomy vector, real count or N/A (no fake rows in CSV) ----
present = combined["vector"].value_counts().to_dict()
report = []
for cat, vecs in TAXONOMY.items():
    for v in vecs:
        cnt = present.get(v, 0)
        report.append({"parent_category": cat, "vector": v, "rows": cnt if cnt else "N/A"})
rep = pd.DataFrame(report)
rep.to_csv(OUTPUT_DIR/"coverage_report.csv", index=False)

print(f"\nFINAL: {len(combined)} rows")
covered = sum(1 for r in report if r["rows"] != "N/A")
print(f"COVERAGE: {covered}/24 vectors have real data, {24-covered}/24 are N/A\n")
for cat, vecs in TAXONOMY.items():
    print(cat)
    for v in vecs:
        cnt = present.get(v, 0)
        print(f"    {v[:46]:46s} {cnt if cnt else 'N/A'}")
