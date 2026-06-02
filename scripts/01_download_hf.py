import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
from datasets import load_dataset
import io

DOWNLOAD_DIR = Path("downloaded")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# DIRECT PARQUET URLs (from HuggingFace convert/parquet endpoints)
# ═══════════════════════════════════════════════════════════════════════════════════

PARQUET_URLS = {
    # microsoft/llmail-inject-challenge (multiple files)
    "microsoft_llmail": [
        "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default/Phase1/0000.parquet",
        "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default/Phase1/0001.parquet",
        "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default/Phase1/0002.parquet",
        "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default/Phase2/0000.parquet",
    ],
    
    # qxcv/tensor-trust
    "tensor_trust": [
        "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ],
    
    # TrustAIRLab/in-the-wild-jailbreak-prompts (use largest config)
    "trustairlab_jailbreak": [
        "https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/resolve/refs%2Fconvert%2Fparquet/regular_2023_12_25/train/0000.parquet",
    ],
    
    # Bordair/bordair-multimodal
    "bordair_multimodal": [
        "https://huggingface.co/datasets/Bordair/bordair-multimodal/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ],
    
    # truongp/web-attack-detection
    "truongp_web_attack": [
        "https://huggingface.co/datasets/truongp/web-attack-detection/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ],
    
    # Antijection/prompt-injection-dataset-v1
    "antijection_mco": [
        "https://huggingface.co/datasets/Antijection/prompt-injection-dataset-v1/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ],
}

print("=" * 90)
print("DOWNLOADING DATASETS VIA DIRECT PARQUET URLs (NO AUTH)")
print("=" * 90)
print()

success = 0
failed = 0

for dataset_name, urls in PARQUET_URLS.items():
    out_path = DOWNLOAD_DIR / f"{dataset_name}.parquet"
    
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024**2)
        print(f"  ✅ {dataset_name}: already exists ({size_mb:.1f} MB)")
        success += 1
        continue
    
    try:
        print(f"  📥 {dataset_name}: downloading {len(urls)} file(s)...", end=" ", flush=True)
        
        frames = []
        for url in urls:
            try:
                # Download with timeout
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Read parquet directly from response
                df = pd.read_parquet(io.BytesIO(response.content))
                frames.append(df)
            except Exception as e:
                print(f"\n     ⚠️  Skipped file (error: {str(e)[:40]})")
                continue
        
        if frames:
            # Combine all parquet files
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            
            size_mb = out_path.stat().st_size / (1024**2)
            print(f"✅ {len(combined)} rows ({size_mb:.1f} MB)")
            success += 1
        else:
            print(f"❌ No files downloaded")
            failed += 1
            
    except Exception as e:
        print(f"❌ {str(e)[:60]}")
        failed += 1

# ═══════════════════════════════════════════════════════════════════════════════════
# ALSO DOWNLOAD THE NORMAL ONES THAT WORK
# ═══════════════════════════════════════════════════════════════════════════════════

print()
print("=" * 90)
print("DOWNLOADING STANDARD DATASETS (NORMAL LOAD_DATASET)")
print("=" * 90)
print()

NORMAL_DATASETS = [
    ("Lakera/gandalf_ignore_instructions", "lakera_gandalf", "train"),
    ("deepset/prompt-injections", "deepset_pi", "train"),
    ("hackaprompt/hackaprompt-dataset", "hackaprompt", "train"),
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

for repo, name, split in tqdm(NORMAL_DATASETS, desc="Standard datasets"):
    out_path = DOWNLOAD_DIR / f"{name}.parquet"
    
    if out_path.exists():
        continue
    
    try:
        ds = load_dataset(repo, split=split, trust_remote_code=False)
        df = ds.to_pandas()
        df.to_parquet(out_path, index=False)
        success += 1
    except Exception as e:
        error_msg = str(e)[:60]
        print(f"  ⚠️  {repo}: {error_msg}")
        failed += 1

print()
print("=" * 90)
print(f"✅ SUCCESS: {success} datasets")
print(f"❌ FAILED: {failed} datasets")
print("=" * 90)
print()

print("Downloaded files in downloaded/:")
total_size = 0
for f in sorted(DOWNLOAD_DIR.glob("*.parquet")):
    size_mb = f.stat().st_size / (1024**2)
    total_size += size_mb
    print(f"  ✅ {f.name:40s} {size_mb:8.1f} MB")

print(f"\nTotal: {total_size:.1f} MB")
