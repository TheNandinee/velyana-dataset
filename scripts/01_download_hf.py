import pandas as pd
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import kagglehub
import shutil

DOWNLOAD_DIR = Path("downloaded")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# HUGGINGFACE DATASETS
# ═══════════════════════════════════════════════════════════════════

HF_DATASETS = [
    ("Lakera/gandalf_ignore_instructions", "lakera_gandalf", "train"),
    ("deepset/prompt-injections", "deepset_pi", "train"),
    ("microsoft/llmail-inject-challenge", "microsoft_llmail", "train"),
    ("qxcv/tensor-trust", "tensor_trust", "train"),
    ("hackaprompt/hackaprompt-dataset", "hackaprompt", "train"),
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "trustairlab_jailbreak", "train"),
    ("reshabhs/SPML_Chatbot_Prompt_Injection", "spml_chatbot_pi", "train"),
    ("xTRam1/safe-guard-prompt-injection", "xtram1_safeguard", "train"),
    ("jayavibhav/prompt-injection", "jayavibhav_pi", "train"),
    ("qualifire/prompt-injections-benchmark", "qualifire_pi", "train"),
    ("gabrielchua/system-prompt-leakage", "gabrielchua_spe", "train"),
    ("Waiper/ExploitDB_DataSet", "waiper_exploitdb", "train"),
    ("ai4privacy/pii-masking-400k", "ai4privacy_400k", "train"),
    ("nvidia/Nemotron-PII", "nvidia_nemotron_pii", "train"),
    ("Isotonic/pii-masking-200k", "isotonic_pii_200k", "train"),
    ("ai4privacy/open-pii-masking-500k-ai4privacy", "ai4privacy_500k", "train"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "nvidia_aegis", "train"),
    ("toxigen/toxigen-data", "toxigen", "train"),
    ("google/civil_comments", "google_civil_comments", "train"),
    ("ucberkeley-dlab/measuring-hate-speech", "ucb_hate_speech", "train"),
]

# ═══════════════════════════════════════════════════════════════════
# KAGGLE DATASETS (using kagglehub)
# ═══════════════════════════════════════════════════════════════════

KAGGLE_DATASETS = [
    ("syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning", "kaggle_xss.csv"),
    ("sajid576/sql-injection-dataset", "kaggle_sqli.csv"),
    ("sid321axn/malicious-urls-dataset", "kaggle_malicious_urls.csv"),
]

print("=" * 80)
print("DOWNLOADING HUGGINGFACE & KAGGLE DATASETS")
print("=" * 80)
print()

# ──────────────────────────────────────────────────────────────────
# HUGGINGFACE
# ──────────────────────────────────────────────────────────────────
print("📥 HUGGINGFACE DATASETS\n")

hf_success = 0
hf_failed = 0

for repo, name, split in tqdm(HF_DATASETS, desc="Downloading HF"):
    out_path = DOWNLOAD_DIR / f"{name}.parquet"
    
    if out_path.exists():
        hf_success += 1
        continue
    
    try:
        # Load without trust_remote_code (deprecated)
        ds = load_dataset(repo, split=split)
        df = ds.to_pandas()
        df.to_parquet(out_path, index=False)
        hf_success += 1
    except Exception as e:
        error_msg = str(e)[:50]
        # Skip datasets that have loading script issues
        if "loading script" in error_msg or "trust_remote_code" in error_msg:
            print(f"  ⚠️  {repo}: Dataset requires custom loading (skipping)")
        else:
            print(f"  ❌ {repo}: {error_msg}")
        hf_failed += 1

print(f"\n✅ HuggingFace: {hf_success}/{len(HF_DATASETS)} downloaded\n")

# ──────────────────────────────────────────────────────────────────
# KAGGLE (using kagglehub)
# ──────────────────────────────────────────────────────────────────
print("📥 KAGGLE DATASETS\n")

kaggle_success = 0
kaggle_failed = 0

for dataset_id, csv_name in tqdm(KAGGLE_DATASETS, desc="Downloading Kaggle"):
    out_path = DOWNLOAD_DIR / csv_name
    
    if out_path.exists():
        kaggle_success += 1
        continue
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(dataset_id)
        
        # Find the CSV file in the downloaded folder
        kaggle_path = Path(path)
        csv_files = list(kaggle_path.glob("*.csv"))
        
        if csv_files:
            # Copy first CSV found to downloaded/ with correct name
            shutil.copy(csv_files[0], out_path)
            kaggle_success += 1
        else:
            print(f"  ⚠️  {dataset_id}: No CSV found in downloaded folder")
            kaggle_failed += 1
            
    except Exception as e:
        print(f"  ❌ {dataset_id}: {str(e)[:50]}")
        kaggle_failed += 1

print(f"\n✅ Kaggle: {kaggle_success}/{len(KAGGLE_DATASETS)} downloaded\n")

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"HuggingFace: {hf_success}/{len(HF_DATASETS)} ✅")
print(f"Kaggle: {kaggle_success}/{len(KAGGLE_DATASETS)} ✅")
print(f"Total: {hf_success + kaggle_success} datasets ready")
print()
print("Downloaded files in downloaded/ folder:")
for f in sorted(DOWNLOAD_DIR.glob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name} ({size_mb:.1f} MB)")
print()
print("=" * 80)
print("NEXT: Run  python scripts/02_build_datasets_APPEND.py")
print("=" * 80)
