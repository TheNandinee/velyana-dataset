import pandas as pd
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

DOWNLOAD_DIR = Path("downloaded")
DOWNLOAD_DIR.mkdir(exist_ok=True)

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

print("=" * 80)
print("DOWNLOADING HUGGINGFACE DATASETS")
print("=" * 80)
print()

hf_success = 0
hf_failed = 0

for repo, name, split in tqdm(HF_DATASETS, desc="Downloading HF"):
    out_path = DOWNLOAD_DIR / f"{name}.parquet"
    
    if out_path.exists():
        hf_success += 1
        continue
    
    try:
        ds = load_dataset(repo, split=split)
        df = ds.to_pandas()
        df.to_parquet(out_path, index=False)
        hf_success += 1
    except Exception as e:
        error_msg = str(e)[:60]
        if "loading script" in error_msg.lower() or "trust_remote_code" in error_msg.lower():
            print(f"  ⚠️  {repo}: Dataset requires custom loading (skipping)")
        else:
            print(f"  ❌ {repo}: {error_msg}")
        hf_failed += 1

print(f"\n✅ HuggingFace: {hf_success}/{len(HF_DATASETS)} downloaded")
print(f"⚠️  Failed: {hf_failed}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total HuggingFace datasets: {hf_success}/{len(HF_DATASETS)}")
print()
print("Downloaded files:")
for f in sorted(DOWNLOAD_DIR.glob("*.parquet")):
    size_mb = f.stat().st_size / (1024**2)
    print(f"  ✅ {f.name} ({size_mb:.1f} MB)")
print()
print("⚠️  KAGGLE DATASETS:")
print("  To add Kaggle CSVs, download them manually and upload to your repo:")
print("  - kaggle_xss.csv")
print("  - kaggle_sqli.csv")
print("  - kaggle_malicious_urls.csv")
print("=" * 80)
print("NEXT: Run  python scripts/02_build_datasets_APPEND.py")
print("=" * 80)
