import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output"); BATCH = 512; PRE_CAP = 8000; SIM = 0.80
# SIM = keep a prompt only if it is < 80% cosine-similar to everything already kept.
# No MIN floor and no MAX cap: every vector with real data is kept at its real size.

TAXONOMY = {
 "Prompt Injection": ["Direct instruction override","Role and persona assignment","Indirect injection via context","Encoded and obfuscated injection","Second-hop and forwarded injection"],
 "System Prompt Extraction": ["Direct questioning","Repetition and verbatim requests","Format conversion extraction","Summarisation inference","Completion-based extraction","Behavioural inference and side-channel probing"],
 "Malicious Code and Payload Injection": ["Cross-site scripting (XSS) generation","SQL injection payload generation","Command injection payload generation","Jailbreak via code context","Prompt-embedded payload delivery","Encoded and obfuscated payload delivery"],
 "Privilege and Identity Attacks": ["Cross-customer identity leakage (BOLA)","Authentication credential extraction"],
 "Malicious Content in Output": ["HTML and JavaScript injection in output","Malicious URL generation","Data exfiltration via output formatting","Misinformation and hallucination manipulation","Sycophantic manipulation and false premise confirmation"],
 "Denial of Service and Resource Exhaustion": ["Sponge attacks","Recursive elaboration attacks","Context window flooding"],
 "Training Data and Knowledge Attacks": ["Training data extraction","Model inversion and fine-tuning probing","Adversarial knowledge boundary exploitation"],
 "Multi-modal and Cross-modal Injection": ["Image-embedded instruction injection","Document structure injection"],
}

print("Loading model..."); model = SentenceTransformer("all-MiniLM-L6-v2"); print("ready\n")

def cosine_dedup(df):
    kept = []
    for (parent, vector), chunk in df.groupby(["parent_category","vector"], sort=False):
        chunk = chunk.reset_index(drop=True)
        if len(chunk) > PRE_CAP:
            chunk = chunk.sample(PRE_CAP, random_state=42).reset_index(drop=True)
        n = len(chunk); print(f"  {vector[:34]:34s} n={n}", flush=True)
        if n <= 1: kept.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(), batch_size=BATCH,
                           show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        buf = np.empty((n, emb.shape[1]), dtype=np.float32); idx=[]; k=0
        for i in range(n):
            if k == 0 or (buf[:k] @ emb[i]).max() < SIM:
                buf[k] = emb[i]; k += 1; idx.append(i)
        kept.append(chunk.iloc[idx]); print(f"     -> {len(idx)}", flush=True)
    return pd.concat(kept, ignore_index=True)

# ATTACK: cosine-dedup every vector; keep all (no floor, no cap)
attack = pd.read_csv(OUTPUT_DIR/"attack_prompts_raw.csv")
attack = attack[attack["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("=== ATTACK ==="); attack = cosine_dedup(attack)
attack.to_csv(OUTPUT_DIR/"attack_prompts.csv", index=False, quoting=csv.QUOTE_ALL)

# BENIGN: exact-dedup only (keeps the full benign set; fast)
benign = pd.read_csv(OUTPUT_DIR/"benign_prompts_raw.csv")
benign = benign[benign["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
benign.to_csv(OUTPUT_DIR/"benign_prompts.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"\nbenign: {len(benign)}")

combined = pd.concat([attack, benign], ignore_index=True)
combined.to_csv(OUTPUT_DIR/"velyana_dataset_final.csv", index=False, quoting=csv.QUOTE_ALL)

# COVERAGE REPORT over all 31 taxonomy vectors (N/A where no real data; never written as fake rows)
present = combined["vector"].value_counts().to_dict()
report = [{"parent_category": c, "vector": v, "rows": present.get(v, 0) or "N/A"}
          for c, vs in TAXONOMY.items() for v in vs]
pd.DataFrame(report).to_csv(OUTPUT_DIR/"coverage_report.csv", index=False)

covered = sum(1 for r in report if r["rows"] != "N/A")
print(f"\nFINAL: {len(combined)} rows | {covered}/31 vectors present, {31-covered}/31 N/A")
print(f"attack: {len(attack)}  benign: {len(benign)}\n")
for c, vs in TAXONOMY.items():
    print(c)
    for v in vs:
        print(f"    {v[:46]:46s} {present.get(v,0) or 'N/A'}")
