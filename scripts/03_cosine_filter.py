import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output"); BATCH_SIZE = 256; CAP_PER_CATEGORY = 5000
THRESHOLDS = {"Prompt Injection": 0.65, "Safety": 0.70}
DEFAULT_THRESHOLD = 0.80

print("Loading model..."); model = SentenceTransformer("all-MiniLM-L6-v2"); print("✅\n")

def cosine_dedup(df):
    kept = []
    for (parent, vector), chunk in df.groupby(["parent_category","vector"], sort=False):
        chunk = chunk.reset_index(drop=True); n = len(chunk)
        thr = THRESHOLDS.get(parent, DEFAULT_THRESHOLD)
        if n <= 1: kept.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(), batch_size=BATCH_SIZE,
                           show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
        buf = np.empty((n, emb.shape[1]), dtype=np.float32); idx = []; k = 0
        for i in range(n):
            if k == 0 or (buf[:k] @ emb[i]).max() < thr:
                buf[k] = emb[i]; k += 1; idx.append(i)
        out = chunk.iloc[idx]; print(f"  dedup {parent[:20]:20s}|{vector[:30]:30s} {n:5d}→{len(out):5d}")
        kept.append(out)
    return pd.concat(kept, ignore_index=True)

def cap_category(df, cap):
    counts = df["vector"].value_counts().to_dict()
    if sum(counts.values()) <= cap: return df
    alloc, rem, vs = {}, cap, sorted(counts, key=lambda v: counts[v]); n = len(vs)
    for v in vs:
        take = min(counts[v], rem // n); alloc[v] = take; rem -= take; n -= 1
    for v in sorted(counts, key=lambda v: counts[v], reverse=True):
        if rem <= 0: break
        extra = min(counts[v]-alloc[v], rem); alloc[v] += extra; rem -= extra
    return pd.concat([g.sample(alloc[v], random_state=42) if alloc[v] < len(g) else g
                      for v, g in df.groupby("vector")], ignore_index=True)

def process(name, raw, out, cap):
    if not raw.exists(): print(f"❌ {raw.name} missing"); return None
    print(f"\n=== {name} ===")
    df = pd.read_csv(raw)
    df = df[df["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
    df = cosine_dedup(df)
    if cap:
        df = pd.concat([cap_category(g, CAP_PER_CATEGORY) for _, g in df.groupby("parent_category")], ignore_index=True)
    df.to_csv(out, index=False); return df

attack = process("ATTACK", OUTPUT_DIR/"attack_prompts_raw.csv", OUTPUT_DIR/"attack_prompts.csv", True)
benign = process("BENIGN", OUTPUT_DIR/"benign_prompts_raw.csv", OUTPUT_DIR/"benign_prompts.csv", False)
if attack is not None and benign is not None:
    combined = pd.concat([attack, benign], ignore_index=True)
    combined.to_csv(OUTPUT_DIR/"velyana_dataset_final.csv", index=False)
    print(f"\nFINAL: {len(combined)} rows")
    for cat in sorted(combined["parent_category"].unique()):
        n = (combined["parent_category"]==cat).sum(); print(f"  {cat}: {n} ({n/len(combined)*100:.1f}%)")
