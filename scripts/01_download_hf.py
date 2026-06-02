import pandas as pd
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

DOWNLOAD_DIR = Path("downloaded")
DOWNLOAD_DIR.mkdir(exist_ok=True)

HF_DATASETS = [
    ("microsoft/llmail-inject-challenge", "microsoft_llmail", "train"),
    ("qxcv/tensor-trust", "tensor_trust", "train"),
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "trustairlab_jailbreak", "train"),
    ("reshabhs/SPML_Chatbot_Prompt_Injection", "spml_chatbot_pi", "train"),
    ("jayavibhav/prompt-injection", "jayavibhav_pi", "train"),
    ("qualifire/prompt-injections-benchmark", "qualifire_pi", "train"),
    ("gabrielchua/system-prompt-leakage", "gabrielchua_spe", "train"),
    ("Bordair/bordair-multimodal", "bordair_multimodal", "train"),
    ("truongp/web-attack-detection", "truongp_web_attack", "train"),
    ("Antijection/prompt-injection-dataset-v1", "antijection_mco", "train"),
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