import os, io, requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

DOWNLOAD_DIR = Path("downloaded"); DOWNLOAD_DIR.mkdir(exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset={repo}"
MS = "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/refs%2Fconvert%2Fparquet/default"

DATASETS = {
    # Prompt Injection
    "lakera_gandalf":        dict(repo="Lakera/gandalf_ignore_instructions"),
    "deepset_pi":            dict(repo="deepset/prompt-injections"),
    "hackaprompt":           dict(repo="hackaprompt/hackaprompt-dataset"),
    "spml_chatbot_pi":       dict(repo="reshabhs/SPML_Chatbot_Prompt_Injection"),
    "xtram1_safeguard":      dict(repo="xTRam1/safe-guard-prompt-injection"),
    "jayavibhav_pi":         dict(repo="jayavibhav/prompt-injection"),
    "qualifire_pi":          dict(repo="qualifire/prompt-injections-benchmark"),
    "neuralchemy_pi":        dict(repo="neuralchemy/Prompt-injection-dataset", configs=["full"]),
    "microsoft_llmail":      dict(repo="microsoft/llmail-inject-challenge",
                                  urls=[f"{MS}/Phase1/0000.parquet", f"{MS}/Phase1/0001.parquet",
                                        f"{MS}/Phase1/0002.parquet", f"{MS}/Phase2/0000.parquet"]),
    "tensor_trust":          dict(repo="qxcv/tensor-trust"),
    "trustairlab_jailbreak": dict(repo="TrustAIRLab/in-the-wild-jailbreak-prompts",
                                  configs=["jailbreak_2023_05_07", "jailbreak_2023_12_25"]),
    # System Prompt Extraction
    "gabrielchua_spe":       dict(repo="gabrielchua/system-prompt-leakage"),
    "bordair_multimodal":    dict(repo="Bordair/bordair-multimodal"),
    # Malicious Code
    "waiper_exploitdb":      dict(repo="Waiper/ExploitDB_DataSet"),
    "truongp_web_attack":    dict(repo="truongp/web-attack-detection"),
    # Malicious Content in Output (cat 5)
    "meg_tong_sycophancy":   dict(repo="meg-tong/sycophancy-eval"),
    # Safety baseline
    "nvidia_aegis":          dict(repo="nvidia/Aegis-AI-Content-Safety-Dataset-1.0"),
    "toxigen":               dict(repo="toxigen/toxigen-data"),
    "google_civil_comments": dict(repo="google/civil_comments"),
    "ucb_hate_speech":       dict(repo="ucberkeley-dlab/measuring-hate-speech"),
}


def _download_urls(urls):
    frames = []
    for u in urls:
        r = requests.get(u, headers=HEADERS, timeout=300); r.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(r.content)))
    return pd.concat(frames, ignore_index=True) if frames else None


def _discover(repo, configs=None):
    r = requests.get(PARQUET_API.format(repo=repo), headers=HEADERS, timeout=60); r.raise_for_status()
    files = r.json().get("parquet_files", [])
    if configs: files = [f for f in files if f["config"] in configs]
    return [f["url"] for f in files]


def _load_standard(repo, configs=None):
    token = HF_TOKEN or None
    if configs:
        frames = []
        for c in configs:
            try:
                dd = load_dataset(repo, c, token=token)
                frames.append(concatenate_datasets([dd[s] for s in dd]).to_pandas())
            except Exception: pass
        if frames: return pd.concat(frames, ignore_index=True)
    try:
        return load_dataset(repo, split="train", token=token).to_pandas()
    except Exception:
        dd = load_dataset(repo, token=token)
        return concatenate_datasets([dd[s] for s in dd]).to_pandas()


def fetch(spec):
    if spec.get("urls"):
        try:
            df = _download_urls(spec["urls"])
            if df is not None and len(df): return df, "urls"
        except Exception as e: print(f"     urls failed: {str(e)[:55]}")
    try:
        urls = _discover(spec["repo"], spec.get("configs"))
        if urls:
            df = _download_urls(urls)
            if df is not None and len(df): return df, "api"
    except Exception as e: print(f"     api failed: {str(e)[:55]}")
    return _load_standard(spec["repo"], spec.get("configs")), "load_dataset"


success = failed = 0
print("=" * 80); print("DOWNLOADING DATASETS"); print("=" * 80)
for name, spec in tqdm(DATASETS.items(), desc="Datasets"):
    out = DOWNLOAD_DIR / f"{name}.parquet"
    if out.exists(): success += 1; continue
    try:
        df, method = fetch(spec); df.to_parquet(out, index=False)
        print(f"  ✅ {name}: {len(df)} rows (via {method})"); success += 1
    except Exception as e:
        msg = str(e)[:70]
        tag = "🔒 gated/private" if ("401" in msg or "403" in msg) else "❌"
        print(f"  {tag} {name}: {msg}"); failed += 1
print("\n" + "=" * 80); print(f"✅ {success}   ❌ {failed}")
for f in sorted(DOWNLOAD_DIR.glob("*.parquet")):
    print(f"  {f.name:42s} {f.stat().st_size/1024**2:8.1f} MB")
