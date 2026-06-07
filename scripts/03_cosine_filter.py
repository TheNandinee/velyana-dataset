import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output"); BATCH = 512; PRE_CAP = 8000
SIM = 0.75                 # nothing >= 75% cosine similar kept
SYNTHETIC_FLOOR = 1000     # synthetic vectors must not drop below this

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

def greedy_dedup(emb, threshold):
    """Return indices of kept rows (greedy, first-come)."""
    n = len(emb)
    buf = np.empty((n, emb.shape[1]), dtype=np.float32)
    idx = []; k = 0
    for i in range(n):
        if k == 0 or (buf[:k] @ emb[i]).max() < threshold:
            buf[k] = emb[i]; k += 1; idx.append(i)
    return idx

def cosine_dedup(df):
    kept = []
    for (parent, vector), chunk in df.groupby(["parent_category","vector"], sort=False):
        chunk = chunk.reset_index(drop=True)
        raw_n = len(chunk)
        is_synthetic = (chunk["source"] == "synthetic_llama3.2").all() if "source" in chunk.columns else False
        floor = SYNTHETIC_FLOOR if is_synthetic else 0

        if raw_n > PRE_CAP:
            chunk = chunk.sample(PRE_CAP, random_state=42).reset_index(drop=True)
        n = len(chunk)
        if n <= 1:
            kept.append(chunk); continue

        emb = model.encode(chunk["prompt"].astype(str).tolist(), batch_size=BATCH,
                           show_progress_bar=False, normalize_embeddings=True).astype(np.float32)

        # adaptive threshold: start strict (SIM), relax if synthetic floor isn't met
        used_sim = SIM
        for t in [SIM, SIM+0.05, SIM+0.10, SIM+0.15, 0.95]:
            idx = greedy_dedup(emb, t)
            used_sim = t
            if len(idx) >= floor or floor == 0:
                break

        kept.append(chunk.iloc[idx])
        tag = f" (relaxed to {used_sim:.2f} for floor)" if used_sim > SIM else ""
        print(f"  {vector[:40]:40s} {raw_n:>5d} -> {len(idx):>5d}  sim={used_sim:.2f}{tag}", flush=True)
    return pd.concat(kept, ignore_index=True)

# ATTACK: real (02) + synthetic (04, if present); cosine-dedup every vector
frames = [pd.read_csv(OUTPUT_DIR/"attack_prompts_raw.csv")]
syn = OUTPUT_DIR/"synthetic_prompts.csv"
if syn.exists():
    frames.append(pd.read_csv(syn)); print(f"merging synthetic: {len(frames[-1])} rows")
attack = pd.concat(frames, ignore_index=True)
attack = attack[attack["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
print("=== ATTACK ==="); attack = cosine_dedup(attack)
attack.to_csv(OUTPUT_DIR/"attack_prompts.csv", index=False, quoting=csv.QUOTE_ALL)

# BENIGN: exact-dedup only (fast)
benign = pd.read_csv(OUTPUT_DIR/"benign_prompts_raw.csv")
benign = benign[benign["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
benign.to_csv(OUTPUT_DIR/"benign_prompts.csv", index=False, quoting=csv.QUOTE_ALL)
print(f"\nbenign: {len(benign)}")

combined = pd.concat([attack, benign], ignore_index=True)
combined.to_csv(OUTPUT_DIR/"velyana_dataset_final.csv", index=False, quoting=csv.QUOTE_ALL)

# COVERAGE REPORT
present = combined["vector"].value_counts().to_dict()
report = [{"parent_category": c, "vector": v, "rows": present.get(v, 0) or "N/A"}
          for c, vs in TAXONOMY.items() for v in vs]
pd.DataFrame(report).to_csv(OUTPUT_DIR/"coverage_report.csv", index=False)

covered = sum(1 for r in report if r["rows"] != "N/A")
total = len(report)
print(f"\nFINAL: {len(combined)} rows | {covered}/{total} vectors present, {total-covered}/{total} N/A")
print(f"attack: {len(attack)}  benign: {len(benign)}\n")
for c, vs in TAXONOMY.items():
    print(c)
    for v in vs:
        print(f"    {v[:46]:46s} {present.get(v,0) or 'N/A'}")
