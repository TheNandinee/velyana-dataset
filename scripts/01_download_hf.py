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

print("=" * 70)
print("DOWNLOADING HUGGINGFACE DATASETS")
print("=" * 70)

success = 0
failed = 0

for repo, name, split in tqdm(HF_DATASETS, desc="Downloading"):
    out_path = DOWNLOAD_DIR / f"{name}.parquet"
    
    if out_path.exists():
        success += 1
        continue
    
    try:
        ds = load_dataset(repo, split=split, trust_remote_code=True)
        df = ds.to_pandas()
        df.to_parquet(out_path, index=False)
        success += 1
    except Exception as e:
        print(f"❌ {repo}: {str(e)[:50]}")
        failed += 1

print("\n" + "=" * 70)
print(f"✅ Downloaded {success}/{len(HF_DATASETS)} datasets")
print("=" * 70)
