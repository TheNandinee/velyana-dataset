import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
COSINE_THRESHOLD = 0.80   # drop a prompt if >= this similar to one already kept
BATCH_SIZE = 256

print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded\n")


def cosine_dedup(df, threshold):
    kept_frames = []
    for vector in tqdm(df["vector"].unique(), desc="Dedup by vector"):
        chunk = df[df["vector"] == vector].reset_index(drop=True)
        if len(chunk) <= 1:
            kept_frames.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(),
                           batch_size=BATCH_SIZE, show_progress_bar=False,
                           normalize_embeddings=True).astype(np.float32)
        keep_idx, kept_emb = [0], emb[0:1]
        for i in range(1, len(chunk)):
            if (kept_emb @ emb[i]).max() < threshold:
                keep_idx.append(i)
                kept_emb = np.vstack([kept_emb, emb[i:i + 1]])
        kept = chunk.iloc[keep_idx]
        print(f"  {vector}: {len(chunk)} → {len(kept)}")
        kept_frames.append(kept)
    return pd.concat(kept_frames, ignore_index=True)


def process(name, raw, out):
    if not raw.exists():
        print(f"❌ {raw.name} missing"); return None
    print(f"\n=== {name} ===")
    df = pd.read_csv(raw)
    df = df[df["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
    df = cosine_dedup(df, COSINE_THRESHOLD)
    df.to_csv(out, index=False)
    print(f"✅ {len(df)} rows → {out}")
    return df


attack = process("ATTACK", OUTPUT_DIR / "attack_prompts_raw.csv", OUTPUT_DIR / "attack_prompts.csv")
benign = process("BENIGN", OUTPUT_DIR / "benign_prompts_raw.csv", OUTPUT_DIR / "benign_prompts.csv")

if attack is not None and benign is not None:
    combined = pd.concat([attack, benign], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "velyana_dataset_final.csv", index=False)
    print(f"\n✅ Combined: {len(combined)} rows")
    for cat in sorted(combined["parent_category"].unique()):
        n = (combined["parent_category"] == cat).sum()
        print(f"  {cat}: {n} ({n/len(combined)*100:.1f}%)")
