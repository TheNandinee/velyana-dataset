import os
import io
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

DOWNLOAD_DIR = Path("downloaded")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Token used only as a header / load_dataset arg — never calls /whoami,
# so it won't trip the rate limit that broke `hf auth login`.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset={repo}"

MS = "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default"

# name -> spec. `urls` = hardcoded fallback, `configs` = restrict to these configs.
DATASETS = {
    "lakera_gandalf":        dict(repo="Lakera/gandalf_ignore_instructions"),
    "deepset_pi":            dict(repo="deepset/prompt-injections"),
    "hackaprompt":           dict(repo="hackaprompt/hackaprompt-dataset"),
    "spml_chatbot_pi":       dict(repo="reshabhs/SPML_Chatbot_Prompt_Injection"),
    "xtram1_safeguard":      dict(repo="xTRam1/safe-guard-prompt-injection"),
    "jayavibhav_pi":         dict(repo="jayavibhav/prompt-injection"),
    "qualifire_pi":          dict(repo="qualifire/prompt-injections-benchmark"),  # only 'test'
    "gabrielchua_spe":       dict(repo="gabrielchua/system-prompt-leakage"),
    "waiper_exploitdb":      dict(repo="Waiper/ExploitDB_DataSet"),
    "ai4privacy_400k":       dict(repo="ai4privacy/pii-masking-400k"),
    "nvidia_nemotron_pii":   dict(repo="nvidia/Nemotron-PII"),
    "isotonic_pii_200k":     dict(repo="Isotonic/pii-masking-200k"),
    "ai4privacy_500k":       dict(repo="ai4privacy/open-pii-masking-500k-ai4privacy"),
    "nvidia_aegis":          dict(repo="nvidia/Aegis-AI-Content-Safety-Dataset-1.0"),
    "toxigen":               dict(repo="toxigen/toxigen-data"),
    "google_civil_comments": dict(repo="google/civil_comments"),
    "ucb_hate_speech":       dict(repo="ucberkeley-dlab/measuring-hate-speech"),

    # ---- tricky ones ----
    "microsoft_llmail": dict(
        repo="microsoft/llmail-inject-challenge",
        urls=[f"{MS}/Phase1/0000.parquet", f"{MS}/Phase1/0001.parquet",
              f"{MS}/Phase1/0002.parquet", f"{MS}/Phase2/0000.parquet"],
    ),
    "tensor_trust":       dict(repo="qxcv/tensor-trust"),
    "bordair_multimodal": dict(repo="Bordair/bordair-multimodal"),
    "truongp_web_attack": dict(repo="truongp/web-attack-detection"),
    # jailbreak configs ONLY — the "regular_*" configs are benign prompts
    "trustairlab_jailbreak": dict(
        repo="TrustAIRLab/in-the-wild-jailbreak-prompts",
        configs=["jailbreak_2023_05_07", "jailbreak_2023_12_25"],
    ),
}


def _download_urls(urls):
    frames = []
    for u in urls:
        resp = requests.get(u, headers=HEADERS, timeout=300)
        resp.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(resp.content)))
    return pd.concat(frames, ignore_index=True) if frames else None


def _discover(repo, configs=None):
    r = requests.get(PARQUET_API.format(repo=repo), headers=HEADERS, timeout=60)
    r.raise_for_status()
    files = r.json().get("parquet_files", [])
    if configs:
        files = [f for f in files if f["config"] in configs]
    return [f["url"] for f in files]


def _load_standard(repo, configs=None):
    token = HF_TOKEN or None
    if configs:                                   # config-restricted fallback
        frames = []
        for c in configs:
            try:
                dd = load_dataset(repo, c, token=token)
                frames.append(concatenate_datasets([dd[s] for s in dd]).to_pandas())
            except Exception:
                pass
        if frames:
            return pd.concat(frames, ignore_index=True)
    try:
        return load_dataset(repo, split="train", token=token).to_pandas()
    except Exception:
        dd = load_dataset(repo, token=token)       # split-robust: grab all splits
        return concatenate_datasets([dd[s] for s in dd]).to_pandas()


def fetch(spec):
    # 1) hardcoded URLs
    if spec.get("urls"):
        try:
            df = _download_urls(spec["urls"])
            if df is not None and len(df):
                return df, "urls"
        except Exception as e:
            print(f"     urls failed: {str(e)[:55]}")
    # 2) parquet-API discovery
    try:
        urls = _discover(spec["repo"], spec.get("configs"))
        if urls:
            df = _download_urls(urls)
            if df is not None and len(df):
                return df, "api"
    except Exception as e:
        print(f"     api failed: {str(e)[:55]}")
    # 3) load_dataset
    return _load_standard(spec["repo"], spec.get("configs")), "load_dataset"


success, failed = 0, 0
print("=" * 80)
print("DOWNLOADING DATASETS")
print("=" * 80)

for name, spec in tqdm(DATASETS.items(), desc="Datasets"):
    out = DOWNLOAD_DIR / f"{name}.parquet"
    if out.exists():
        success += 1
        continue
    try:
        df, method = fetch(spec)
        df.to_parquet(out, index=False)
        print(f"  ✅ {name}: {len(df)} rows (via {method})")
        success += 1
    except Exception as e:
        msg = str(e)[:70]
        tag = "🔒 gated/private" if ("401" in msg or "403" in msg) else "❌"
        print(f"  {tag} {name}: {msg}")
        failed += 1

print("\n" + "=" * 80)
print(f"✅ {success}   ❌ {failed}")
for f in sorted(DOWNLOAD_DIR.glob("*.parquet")):
    print(f"  {f.name:42s} {f.stat().st_size/1024**2:8.1f} MB")
