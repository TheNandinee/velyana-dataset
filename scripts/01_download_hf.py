import os, io, time, requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DOWNLOAD_DIR = Path("downloaded"); DOWNLOAD_DIR.mkdir(exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset={repo}"

# repo, config-filter (None = all configs). Parquet-direct only (no load_dataset -> no 429 storm).
DATASETS = {
    # ---- Prompt Injection ----
    "lakera_gandalf":        ("Lakera/gandalf_ignore_instructions", None),
    "deepset_pi":            ("deepset/prompt-injections", None),
    "hackaprompt":           ("hackaprompt/hackaprompt-dataset", None),
    "jayavibhav_pi":         ("jayavibhav/prompt-injection", None),
    "neuralchemy_pi":        ("neuralchemy/Prompt-injection-dataset", ["full"]),
    "microsoft_llmail":      ("microsoft/llmail-inject-challenge", None),
    "trustairlab_jailbreak": ("TrustAIRLab/in-the-wild-jailbreak-prompts",
                              ["jailbreak_2023_05_07", "jailbreak_2023_12_25"]),
    # ---- System Prompt Extraction ----
    "gabrielchua_spe":       ("gabrielchua/system-prompt-leakage", None),
    # ---- Malicious Code ----
    "truongp_web_attack":    ("truongp/web-attack-detection", None),
    "waiper_exploitdb":      ("Waiper/ExploitDB_DataSet", None),
    "rmcbench":              ("zhongqy/RMCBench", None),   # -> Jailbreak via code context
    # ---- Malicious Content in Output ----
    "phishing_urls":         ("shawhin/phishing-site-classification", None),
    "fake_news":             ("GonzaloA/fake_news", None),
    # ---- Safety baseline (NOT part of the 31-vector taxonomy; optional) ----
    "nvidia_aegis":          ("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", None),
    "toxigen":               ("toxigen/toxigen-data", None),
    "google_civil_comments": ("google/civil_comments", None),
    "ucb_hate_speech":       ("ucberkeley-dlab/measuring-hate-speech", None),
    # ---- Benign (fixes attack/benign imbalance) ----
    "dolly":                 ("databricks/databricks-dolly-15k", None),
    "alpaca":                ("tatsu-lab/alpaca", None),
}


def get(url, tries=5):
    delay = 5
    for _ in range(tries):
        r = requests.get(url, headers=HEADERS, timeout=300)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", delay))); delay = min(delay*2, 60); continue
        r.raise_for_status(); return r
    r.raise_for_status()


def discover(repo, configs=None, tries=5):
    delay = 5
    for _ in range(tries):
        r = requests.get(PARQUET_API.format(repo=repo), headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(delay); delay = min(delay*2, 60); continue
        r.raise_for_status()
        files = r.json().get("parquet_files", [])
        if configs:
            files = [f for f in files if f["config"] in configs]
        return [f["url"] for f in files]
    return []


def fetch(repo, configs):
    urls = discover(repo, configs)
    if not urls:
        return None
    return pd.concat([pd.read_parquet(io.BytesIO(get(u).content)) for u in urls], ignore_index=True)


success = failed = 0
print("=" * 80); print("DOWNLOADING DATASETS (parquet-direct, 429-aware)"); print("=" * 80)
for name, (repo, configs) in tqdm(DATASETS.items(), desc="Datasets"):
    out = DOWNLOAD_DIR / f"{name}.parquet"
    if out.exists(): success += 1; continue
    try:
        df = fetch(repo, configs)
        if df is None or len(df) == 0:
            print(f"  WARN {name}: no parquet (skipped)"); failed += 1; continue
        df.to_parquet(out, index=False); print(f"  OK   {name}: {len(df)} rows"); success += 1
    except Exception as e:
        print(f"  FAIL {name}: {str(e)[:70]}"); failed += 1
    time.sleep(2)

print("=" * 80); print(f"OK {success}   FAIL {failed}")
for f in sorted(DOWNLOAD_DIR.glob("*.parquet")):
    print(f"  {f.name:42s} {f.stat().st_size/1024**2:8.1f} MB")
