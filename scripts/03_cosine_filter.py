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

print("Loading sentence transformer model (one-time, ~90MB)...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

def cosine_filter(df, threshold):
    """Remove duplicates within each vector based on cosine similarity."""
    kept = []
    
    for vector in tqdm(df["vector"].unique(), desc="Filtering by cosine similarity"):
        chunk = df[df["vector"] == vector].reset_index(drop=True)
        
        if len(chunk) <= 1:
            kept.append(chunk)
            continue

        prompts = chunk["prompt"].tolist()
        
        # Encode all prompts
        embeddings = model.encode(prompts, batch_size=BATCH_SIZE, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Mark which rows to keep
        keep_mask = np.ones(len(chunk), dtype=bool)

        for i in range(len(chunk)):
            if not keep_mask[i]:
                continue
            
            # Find all rows after i that are still marked as keep
            later_indices = np.where(keep_mask)[0]
            later_indices = later_indices[later_indices > i]
            
            if len(later_indices) == 0:
                continue
            
            # Calculate cosine similarity between row i and all later rows
            sims = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[later_indices])[0]
            
            # Mark similar rows as remove
            too_similar = later_indices[sims >= threshold]
            keep_mask[too_similar] = False

        kept.append(chunk[keep_mask])

    return pd.concat(kept, ignore_index=True)


def cap_samples(df, cap):
    """Cap samples per parent_category."""
    frames = []
    for parent, grp in df.groupby("parent_category"):
        if len(grp) > cap:
            grp = grp.sample(cap, random_state=42)
            print(f"  Capped '{parent}': {len(grp)} → {cap} rows")
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


# ── ATTACK DATASET ────────────────────────────────────────────────
attack_raw = OUTPUT_DIR / "attack_prompts_raw.csv"

if attack_raw.exists():
    print("=" * 80)
    print("PROCESSING ATTACK DATASET")
    print("=" * 80)
    
    print(f"\nLoading {attack_raw.name}...")
    attack_df = pd.read_csv(attack_raw)
    print(f"  Raw: {len(attack_df)} rows")
    
    # Basic cleanup
    attack_df = attack_df[attack_df["prompt"].str.len() > 5]
    attack_df = attack_df.drop_duplicates(subset=["prompt"])
    print(f"  After basic cleanup: {len(attack_df)} rows\n")

    print("Applying cosine similarity filter (threshold: 0.80)...")
    attack_filtered = cosine_filter(attack_df, COSINE_THRESHOLD)
    print(f"  After cosine filter: {len(attack_filtered)} rows\n")

    print("Capping per parent_category (max 1000 each)...")
    attack_final = cap_samples(attack_filtered, CAP_PER_PARENT)
    print(f"  Final total: {len(attack_final)} rows\n")

    attack_out = OUTPUT_DIR / "attack_prompts.csv"
    attack_final.to_csv(attack_out, index=False)
    print(f"✅ Saved → {attack_out}\n")
    
    print("Breakdown by parent_category:")
    for cat in sorted(attack_final["parent_category"].unique()):
        count = len(attack_final[attack_final["parent_category"] == cat])
        print(f"  {cat}: {count}")
else:
    print(f"❌ {attack_raw.name} not found. Run 02_build_datasets_APPEND.py first.")

# ── BENIGN DATASET ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PROCESSING BENIGN DATASET")
print("=" * 80)

benign_raw = OUTPUT_DIR / "benign_prompts_raw.csv"

if benign_raw.exists():
    print(f"\nLoading {benign_raw.name}...")
    benign_df = pd.read_csv(benign_raw)
    print(f"  Raw: {len(benign_df)} rows\n")

    print("Applying cosine similarity filter (threshold: 0.80)...")
    benign_filtered = cosine_filter(benign_df, COSINE_THRESHOLD)
    print(f"  After filter: {len(benign_filtered)} rows\n")

    benign_out = OUTPUT_DIR / "benign_prompts.csv"
    benign_filtered.to_csv(benign_out, index=False)
    print(f"✅ Saved → {benign_out}")
else:
    print(f"❌ {benign_raw.name} not found. Run 02_build_datasets_APPEND.py first.")

# ── COMBINED DATASET ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

attack_final_path = OUTPUT_DIR / "attack_prompts.csv"
benign_final_path = OUTPUT_DIR / "benign_prompts.csv"

if attack_final_path.exists() and benign_final_path.exists():
    attack = pd.read_csv(attack_final_path)
    benign = pd.read_csv(benign_final_path)
    
    combined = pd.concat([attack, benign], ignore_index=True)
    combined_out = OUTPUT_DIR / "velyana_dataset_final.csv"
    combined.to_csv(combined_out, index=False)
    
    print(f"\n✅ Combined dataset → {combined_out}")
    print(f"\nStats:")
    print(f"  Attack rows: {len(attack)}")
    print(f"  Benign rows: {len(benign)}")
    print(f"  Total rows: {len(combined)}")
    print(f"\nBreakdown by parent_category:")
    for cat in sorted(combined["parent_category"].unique()):
        count = len(combined[combined["parent_category"] == cat])
        pct = (count / len(combined)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\nFinal files ready in output/:")
    print("  • attack_prompts.csv")
    print("  • benign_prompts.csv")
    print("  • velyana_dataset_final.csv")
else:
    print("❌ Missing input files. Make sure you ran 02_build_datasets_APPEND.py first.")
