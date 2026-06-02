import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = Path("output")
COSINE_THRESHOLD = 0.80
CAP_PER_PARENT = 1000
BATCH_SIZE = 256

print("=" * 80)
print("COSINE SIMILARITY FILTERING & FINALIZATION")
print("=" * 80)
print()

print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded\n")

def cosine_filter(df, threshold):
    kept = []
    for sub in tqdm(df["subcategory"].unique(), desc="Filtering subcategories"):
        chunk = df[df["subcategory"] == sub].reset_index(drop=True)
        
        if len(chunk) <= 1:
            kept.append(chunk)
            continue

        prompts = chunk["prompt"].tolist()
        embeddings = model.encode(prompts, batch_size=BATCH_SIZE, show_progress_bar=False)
        embeddings = np.array(embeddings)

        keep_mask = np.ones(len(chunk), dtype=bool)

        for i in range(len(chunk)):
            if not keep_mask[i]:
                continue
            later = np.where(keep_mask)[0]
            later = later[later > i]
            if len(later) == 0:
                continue
            sims = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[later])[0]
            too_sim = later[sims >= threshold]
            keep_mask[too_sim] = False

        kept.append(chunk[keep_mask])

    return pd.concat(kept, ignore_index=True)

def cap_samples(df, cap):
    frames = []
    for parent, grp in df.groupby("parent_category"):
        if len(grp) > cap:
            grp = grp.sample(cap, random_state=42)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)

# ATTACK
attack_raw = OUTPUT_DIR / "attack_prompts_raw.csv"
if attack_raw.exists():
    print(f"Loading {attack_raw.name}...")
    attack_df = pd.read_csv(attack_raw)
    print(f"  Raw: {len(attack_df)} rows\n")

    print("Filtering attack dataset...")
    attack_filtered = cosine_filter(attack_df, COSINE_THRESHOLD)
    print(f"After filter: {len(attack_filtered)} rows\n")

    print("Capping per parent_category...")
    attack_final = cap_samples(attack_filtered, CAP_PER_PARENT)

    attack_out = OUTPUT_DIR / "attack_prompts.csv"
    attack_final.to_csv(attack_out, index=False)
    print(f"✅ Saved → {attack_out}")
    print(f"\nFinal breakdown:")
    for cat in sorted(attack_final["parent_category"].unique()):
        count = len(attack_final[attack_final["parent_category"] == cat])
        print(f"  {cat}: {count}")

# BENIGN
benign_raw = OUTPUT_DIR / "benign_prompts_raw.csv"
if benign_raw.exists():
    print(f"\nLoading {benign_raw.name}...")
    benign_df = pd.read_csv(benign_raw)
    print(f"  Raw: {len(benign_df)} rows\n")

    print("Filtering benign dataset...")
    benign_filtered = cosine_filter(benign_df, COSINE_THRESHOLD)

    benign_out = OUTPUT_DIR / "benign_prompts.csv"
    benign_filtered.to_csv(benign_out, index=False)
    print(f"✅ Saved → {benign_out}")

print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)