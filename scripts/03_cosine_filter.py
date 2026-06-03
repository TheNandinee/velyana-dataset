import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
BATCH_SIZE = 256
CAP_PER_CATEGORY = 5000          # hard ceiling per parent_category

# Per-category dedup aggressiveness. LOWER = stricter = fewer, more diverse rows.
THRESHOLDS = {
    "Prompt Injection": 0.65,
    "Safety":           0.70,
}
DEFAULT_THRESHOLD = 0.80

print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded\n")


def cosine_dedup(df):
    kept_frames = []
    for (parent, vector), chunk in df.groupby(["parent_category", "vector"], sort=False):
        chunk = chunk.reset_index(drop=True)
        n = len(chunk)
        thr = THRESHOLDS.get(parent, DEFAULT_THRESHOLD)
        if n <= 1:
            kept_frames.append(chunk); continue
        emb = model.encode(chunk["prompt"].astype(str).tolist(),
                           batch_size=BATCH_SIZE, show_progress_bar=False,
                           normalize_embeddings=True).astype(np.float32)
        kept = np.empty((n, emb.shape[1]), dtype=np.float32)
        keep_idx, kcount = [], 0
        for i in range(n):
            if kcount == 0 or (kept[:kcount] @ emb[i]).max() < thr:
                kept[kcount] = emb[i]; kcount += 1; keep_idx.append(i)
        out = chunk.iloc[keep_idx]
        print(f"  dedup {parent[:20]:20s} | {vector[:32]:32s} {n:6d} → {len(out):5d}")
        kept_frames.append(out)
    return pd.concat(kept_frames, ignore_index=True)


def cap_category(df, cap):
    """Trim a category to `cap` rows, fairly across vectors (water-filling):
    small vectors keep everything, only oversized vectors get sampled down."""
    counts = df["vector"].value_counts().to_dict()
    if sum(counts.values()) <= cap:
        return df
    alloc, remaining, vs = {}, cap, sorted(counts, key=lambda v: counts[v])  # smallest first
    n_left = len(vs)
    for v in vs:
        take = min(counts[v], remaining // n_left)
        alloc[v] = take; remaining -= take; n_left -= 1
    # hand any leftover budget to the largest vectors that still have room
    for v in sorted(counts, key=lambda v: counts[v], reverse=True):
        if remaining <= 0: break
        extra = min(counts[v] - alloc[v], remaining)
        alloc[v] += extra; remaining -= extra
    parts = []
    for v, g in df.groupby("vector"):
        parts.append(g.sample(alloc[v], random_state=42) if alloc[v] < len(g) else g)
    return pd.concat(parts, ignore_index=True)


def process(name, raw, out, apply_cap):
    if not raw.exists():
        print(f"❌ {raw.name} missing"); return None
    print(f"\n=== {name} ===")
    df = pd.read_csv(raw)
    df = df[df["prompt"].astype(str).str.len() > 5].drop_duplicates(subset=["prompt"])
    df = cosine_dedup(df)
    if apply_cap:
        capped = []
        for parent, grp in df.groupby("parent_category"):
            c = cap_category(grp, CAP_PER_CATEGORY)
            if len(c) < len(grp):
                print(f"  cap   {parent}: {len(grp)} → {len(c)}")
            capped.append(c)
        df = pd.concat(capped, ignore_index=True)
    df.to_csv(out, index=False)
    return df


attack = process("ATTACK", OUTPUT_DIR / "attack_prompts_raw.csv",
                 OUTPUT_DIR / "attack_prompts.csv", apply_cap=True)
benign = process("BENIGN", OUTPUT_DIR / "benign_prompts_raw.csv",
                 OUTPUT_DIR / "benign_prompts.csv", apply_cap=False)  # only 150, no cap

if attack is not None and benign is not None:
    combined = pd.concat([attack, benign], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "velyana_dataset_final.csv", index=False)
    print("\n" + "=" * 60)
    print(f"FINAL: {len(combined)} rows")
    for cat in sorted(combined["parent_category"].unique()):
        n = (combined["parent_category"] == cat).sum()
        print(f"  {cat}: {n} ({n/len(combined)*100:.1f}%)")
