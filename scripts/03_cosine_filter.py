import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
COSINE_THRESHOLD = 0.80   # drop a prompt if it's >= this similar to one already kept
BATCH_SIZE = 256

print("=" * 80)
print("COSINE SIMILARITY DEDUPLICATION (no cap)")
print("=" * 80)

print("\nLoading sentence transformer model (one-time, ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded\n")


def cosine_dedup(df, threshold):
    """Greedy near-duplicate removal within each vector.
    Keeps a prompt only if its max similarity to all already-kept
    prompts in the same vector is below `threshold`."""
    kept_frames = []

    for vector in tqdm(df["vector"].unique(), desc="Dedup by vector"):
        chunk = df[df["vector"] == vector].reset_index(drop=True)

        if len(chunk) <= 1:
            kept_frames.append(chunk)
            continue

        prompts = chunk["prompt"].astype(str).tolist()
        # normalize so a plain dot product == cosine similarity
        emb = model.encode(
            prompts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        keep_idx = [0]
        kept_emb = emb[0:1]  # running matrix of kept embeddings

        for i in range(1, len(chunk)):
            sims = kept_emb @ emb[i]          # cosine sims to everything kept so far
            if sims.max() < threshold:        # not too similar to anything → keep
                keep_idx.append(i)
                kept_emb = np.vstack([kept_emb, emb[i:i + 1]])

        kept = chunk.iloc[keep_idx]
        print(f"  {vector}: {len(chunk)} → {len(kept)}")
        kept_frames.append(kept)

    return pd.concat(kept_frames, ignore_index=True)


def process(name, raw_path, out_path):
    if not raw_path.exists():
        print(f"❌ {raw_path.name} not found. Run 02 first.")
        return None

    print("=" * 80)
    print(f"PROCESSING {name}")
    print("=" * 80)

    df = pd.read_csv(raw_path)
    print(f"  Raw: {len(df)} rows")

    df = df[df["prompt"].astype(str).str.len() > 5]
    df = df.drop_duplicates(subset=["prompt"])     # exact-dup removal first (cheap)
    print(f"  After exact-dup cleanup: {len(df)} rows\n")

    df = cosine_dedup(df, COSINE_THRESHOLD)
    print(f"\n  After cosine dedup: {len(df)} rows")

    df.to_csv(out_path, index=False)
    print(f"✅ Saved → {out_path}\n")
    return df


attack = process("ATTACK DATASET",
                 OUTPUT_DIR / "attack_prompts_raw.csv",
                 OUTPUT_DIR / "attack_prompts.csv")

benign = process("BENIGN DATASET",
                 OUTPUT_DIR / "benign_prompts_raw.csv",
                 OUTPUT_DIR / "benign_prompts.csv")

# ── COMBINED ──────────────────────────────────────────────────────
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

if attack is not None and benign is not None:
    combined = pd.concat([attack, benign], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "velyana_dataset_final.csv", index=False)

    print(f"\n  Attack rows: {len(attack)}")
    print(f"  Benign rows: {len(benign)}")
    print(f"  Total rows:  {len(combined)}\n")
    print("Breakdown by parent_category:")
    for cat in sorted(combined["parent_category"].unique()):
        n = (combined["parent_category"] == cat).sum()
        print(f"  {cat}: {n} ({n/len(combined)*100:.1f}%)")

print("\n✅ COMPLETE")
