import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output"); BATCH_SIZE = 256
MIN_PER_VECTOR = 1000      # floor: vectors below this drop to N/A
MAX_PER_VECTOR = None      # no ceiling — the 0.80 similarity filter is the limiter
SIM_THRESHOLD = 0.80       # keep a prompt only if <80% similar to everything kept

TAXONOMY = {
 "Prompt Injection": ["Direct instruction override","Role and persona assignment","Indirect injection via context","Encoded and obfuscated injection","Second-hop and forwarded injection"],
 "System Prompt Extraction": ["Direct questioning","Repetition and verbatim requests","Format conversion extraction","Summarisation inference","Completion-based extraction","Behavioural inference and side-channel probing"],
 "Malicious Code and Payload Injection": ["Cross-site scripting (XSS) generation","SQL injection payload generation","Command injection payload generation","Jailbreak via code context","Prompt-embedded payload delivery","Encoded and obfuscated payload delivery"],
 "Privilege and Identity Attacks": ["Cross-customer identity leakage (BOLA)","Authentication credential extraction"],
 "Malicious Content in Output": ["HTML and JavaScript injection in output","Malicious URL generation","Data exfiltration via output formatting","Misinformation and hallucination manipulation","Sycophantic manipulation and false premise confirmation"],
}

print("Loading model..."); model = SentenceTransformer("all-MiniLM-L6-v2"); print("✅\n")

PRE_CAP = 8000   # max rows per vector fed into dedup (plenty to survive down to 1000)

def cosine_dedup(df):
    kept = []
    for (parent, vector), chunk in df.groupby(["parent_category","vector"], sort=False):
        chunk = chunk.reset_index(drop=True)
        if len(chunk) > PRE_CAP:
            chunk = chunk.sample(PRE_CAP, random_state=42).reset_index(drop=True)
        n = len(chunk)
        print(f"  dedup {vector[:34]:34s} n={n} ...", flush=True)
        if n <= 1: kept.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(), batch_size=512,
                           show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        buf = np.empty((n, emb.shape[1]), dtype=np.float32); idx=[]; k=0
        for i in range(n):
            if k == 0 or (buf[:k] @ emb[i]).max() < SIM_THRESHOLD:
                buf[k] = emb[i]; k += 1; idx.append(i)
        out = chunk.iloc[idx]; print(f"     → kept {len(out)}", flush=True); kept.append(out)
    return pd.concat(kept, ignore_index=True)

def balance(df):
    out = []
    for vec, g in df.groupby("vector"):
        if vec == "Safe": out.append(g); continue
        if len(g) < MIN_PER_VECTOR:
            print(f"  ⚠️  dropped '{vec}' — {len(g)} rows (<{MIN_PER_VECTOR}, → N/A)"); continue
        if MAX_PER_VECTOR and len(g) > MAX_PER_VECTOR:
            g = g.sample(MAX_PER_VECTOR, random_state=42)
        out.append(g)
    return pd.concat(out, ignore_index=True)

attack = pd.read_csv(OUTPUT_DIR/"attack_prompts_raw.csv")
attack = attack[attack["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("=== ATTACK ==="); attack = balance(cosine_dedup(attack))
attack.to_csv(OUTPUT_DIR/"attack_prompts.csv", index=False, quoting=csv.QUOTE_ALL)

benign = pd.read_csv(OUTPUT_DIR/"benign_prompts_raw.csv")
benign = benign[benign["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("\n=== BENIGN ==="); benign = cosine_dedup(benign)
benign.to_csv(OUTPUT_DIR/"benign_prompts.csv", index=False, quoting=csv.QUOTE_ALL)

combined = pd.concat([attack, benign], ignore_index=True)
combined.to_csv(OUTPUT_DIR/"velyana_dataset_final.csv", index=False, quoting=csv.QUOTE_ALL)

present = combined["vector"].value_counts().to_dict()
report = [{"parent_category": cat, "vector": v, "rows": present.get(v, 0) or "N/A"}
          for cat, vecs in TAXONOMY.items() for v in vecs]
pd.DataFrame(report).to_csv(OUTPUT_DIR/"coverage_report.csv", index=False)

print(f"\nFINAL: {len(combined)} rows")
covered = sum(1 for r in report if r["rows"] != "N/A")
print(f"COVERAGE: {covered}/24 vectors ≥{MIN_PER_VECTOR}, {24-covered}/24 N/A\n")
for cat, vecs in TAXONOMY.items():
    print(cat)
    for v in vecs:
        print(f"    {v[:46]:46s} {present.get(v,0) or 'N/A'}")
