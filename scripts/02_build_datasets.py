import pandas as pd
from pathlib import Path

DOWNLOAD_DIR = Path("downloaded")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

attack_rows = []
benign_rows = []

def find_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in keywords or any(kw in col_lower for kw in keywords):
            return col
    return df.columns[0]

def find_label_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in keywords or any(kw in col_lower for kw in keywords):
            return col
    return None

def filter_attacks(df, label_col):
    if label_col is None:
        return df
    if set(df[label_col].dropna().unique()).issubset({0, 1, '0', '1', True, False}):
        return df[df[label_col].isin([1, '1', True])]
    if df[label_col].dtype == 'object':
        keywords = ["attack", "injection", "jailbreak", "leakage", "toxic", "malicious", "true"]
        return df[df[label_col].astype(str).str.lower().apply(lambda x: any(kw in x for kw in keywords))]
    return df

def add_attack(df, parent, sub, source):
    if df is None or len(df) == 0:
        return
    df = df.copy()
    if "prompt" not in df.columns:
        df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["parent_category"] = parent
    df["subcategory"] = sub
    df["source"] = source
    attack_rows.append(df[["prompt", "parent_category", "subcategory", "source"]])
    print(f"  ✅ {source}: {len(df)} rows → {parent} / {sub}")

print("=" * 80)
print("BUILDING ATTACK DATASET (APPEND MODE)")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════
# PROMPT INJECTION
# ═══════════════════════════════════════════════════════════════════
print("📁 PROMPT INJECTION")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "lakera_gandalf.parquet")
    text = find_col(df, ["prompt", "text", "input"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1000, len(df)), random_state=42)
    add_attack(df, "Prompt Injection", "Direct Instruction Override", "lakera_gandalf")
except Exception as e:
    print(f"  ❌ lakera_gandalf: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "deepset_pi.parquet")
    text = find_col(df, ["text", "prompt"])
    label = find_label_col(df, ["label"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label)
    n = len(df) // 2
    add_attack(df.iloc[:n][["prompt"]], "Prompt Injection", "Direct Instruction Override", "deepset_pi_1")
    add_attack(df.iloc[n:][["prompt"]], "Prompt Injection", "Role and Persona Assignment", "deepset_pi_2")
except Exception as e:
    print(f"  ❌ deepset_pi: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "microsoft_llmail.parquet")
    text = find_col(df, ["body", "attack", "payload"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    add_attack(df, "Prompt Injection", "Indirect Injection via Context", "microsoft_llmail")
except Exception as e:
    print(f"  ❌ microsoft_llmail: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "tensor_trust.parquet")
    text = find_col(df, ["attack", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(5000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "Prompt Injection", "Direct Instruction Override", "tensor_trust_1")
    add_attack(df.iloc[n:2*n][["prompt"]], "Prompt Injection", "Role and Persona Assignment", "tensor_trust_2")
    add_attack(df.iloc[2*n:][["prompt"]], "Prompt Injection", "Indirect Injection via Context", "tensor_trust_3")
except Exception as e:
    print(f"  ❌ tensor_trust: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "hackaprompt.parquet")
    text = find_col(df, ["user_input", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(3000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "Prompt Injection", "Direct Instruction Override", "hackaprompt_1")
    add_attack(df.iloc[n:2*n][["prompt"]], "Prompt Injection", "Role and Persona Assignment", "hackaprompt_2")
    add_attack(df.iloc[2*n:][["prompt"]], "Prompt Injection", "Indirect Injection via Context", "hackaprompt_3")
except Exception as e:
    print(f"  ❌ hackaprompt: {e}")

# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT EXTRACTION
# ═══════════════════════════════════════════════════════════════════
print("\n📁 SYSTEM PROMPT EXTRACTION")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "gabrielchua_spe.parquet")
    text = find_col(df, ["content", "prompt", "text"])
    label = find_label_col(df, ["leakage"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label).sample(min(4000, len(df)), random_state=42)
    n = len(df) // 6
    spe_subs = ["Direct Questioning", "Repetition and Verbatim Requests", "Format Conversion Extraction",
                "Summarisation Inference", "Completion-based Extraction", "Behavioural Inference and Side-channel Probing"]
    for i, sub in enumerate(spe_subs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk[["prompt"]], "System Prompt Extraction", sub, f"gabrielchua_spe_{i+1}")
except Exception as e:
    print(f"  ❌ gabrielchua_spe: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "bordair_multimodal.parquet")
    text = find_col(df, ["text", "content", "prompt"])
    label = find_label_col(df, ["expected_detection"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "System Prompt Extraction", "Direct Questioning", "bordair_1")
    add_attack(df.iloc[n:2*n][["prompt"]], "System Prompt Extraction", "Repetition and Verbatim Requests", "bordair_2")
    add_attack(df.iloc[2*n:][["prompt"]], "System Prompt Extraction", "Completion-based Extraction", "bordair_3")
except Exception as e:
    print(f"  ❌ bordair_multimodal: {e}")

# ═══════════════════════════════════════════════════════════════════
# MALICIOUS CODE INJECTION
# ═══════════════════════════════════════════════════════════════════
print("\n📁 MALICIOUS CODE INJECTION")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "waiper_exploitdb.parquet")
    text = find_col(df, ["input", "prompt", "payload"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(3500, len(df)), random_state=42)
    n = len(df) // 6
    subs = ["Cross-site Scripting (XSS) Generation", "SQL Injection Payload Generation",
            "Command Injection Payload Generation", "Encoded and Obfuscated Payload Delivery",
            "Jailbreak via Code Context", "Prompt-embedded Payload Delivery"]
    for i, sub in enumerate(subs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk[["prompt"]], "Malicious Code and Payload Injection", sub, f"waiper_{i+1}")
except Exception as e:
    print(f"  ❌ waiper_exploitdb: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "truongp_web_attack.parquet")
    text = find_col(df, ["sentence", "text", "prompt"])
    label = find_label_col(df, ["label"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label).sample(min(2500, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "Malicious Code and Payload Injection", "Cross-site Scripting (XSS) Generation", "truongp_1")
    add_attack(df.iloc[n:2*n][["prompt"]], "Malicious Code and Payload Injection", "SQL Injection Payload Generation", "truongp_2")
    add_attack(df.iloc[2*n:][["prompt"]], "Malicious Code and Payload Injection", "Command Injection Payload Generation", "truongp_3")
except Exception as e:
    print(f"  ❌ truongp_web_attack: {e}")

if (DOWNLOAD_DIR / "kaggle_xss.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_xss.csv", nrows=5000)
        text = find_col(df, ["sentence", "text", "payload"])
        label = find_label_col(df, ["label"])
        df = df[[text, label] if label else [text]]
        df = df.rename(columns={text: "prompt"})
        df = filter_attacks(df, label)
        add_attack(df[["prompt"]], "Malicious Code and Payload Injection", "Cross-site Scripting (XSS) Generation", "kaggle_xss")
    except Exception as e:
        print(f"  ❌ kaggle_xss: {e}")

if (DOWNLOAD_DIR / "kaggle_sqli.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_sqli.csv", nrows=5000)
        text = find_col(df, ["query", "text", "payload"])
        label = find_label_col(df, ["label"])
        df = df[[text, label] if label else [text]]
        df = df.rename(columns={text: "prompt"})
        df = filter_attacks(df, label)
        add_attack(df[["prompt"]], "Malicious Code and Payload Injection", "SQL Injection Payload Generation", "kaggle_sqli")
    except Exception as e:
        print(f"  ❌ kaggle_sqli: {e}")

# ═══════════════════════════════════════════════════════════════════
# PII
# ═══════════════════════════════════════════════════════════════════
print("\n📁 PRIVILEGE & IDENTITY ATTACKS (PII)")

for fname, name in [("ai4privacy_400k", "ai4privacy_400k"), ("isotonic_pii_200k", "isotonic_200k"),
                     ("ai4privacy_500k", "ai4privacy_500k"), ("nvidia_nemotron_pii", "nvidia_pii")]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        text = find_col(df, ["source_text", "text", "prompt"])
        df = df[[text]].rename(columns={text: "prompt"}).sample(min(3000, len(df)), random_state=42)
        n = len(df) // 7
        pii_subs = ["Email Address Leakage", "Phone Number Leakage", "Government ID Leakage",
                    "Financial Account Data Leakage", "Physical Address Leakage",
                    "Cross-customer Identity Leakage (BOLA)", "Authentication Credential Extraction"]
        for i, sub in enumerate(pii_subs):
            chunk = df.iloc[i*n:(i+1)*n] if i < 6 else df.iloc[6*n:]
            add_attack(chunk[["prompt"]], "Privilege and Identity Attacks", sub, f"{name}_{i+1}")
    except Exception as e:
        print(f"  ❌ {fname}: {e}")

# ═══════════════════════════════════════════════════════════════════
# MALICIOUS CONTENT IN OUTPUT
# ═══════════════════════════════════════════════════════════════════
print("\n📁 MALICIOUS CONTENT IN OUTPUT")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "antijection_mco.parquet")
    text = find_col(df, ["prompt", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 5
    mco_subs = ["HTML and JavaScript Injection in Output", "Malicious URL Generation",
                "Data Exfiltration via Output Formatting", "Misinformation and Hallucination Manipulation",
                "Sycophantic Manipulation and False Premise Confirmation"]
    for i, sub in enumerate(mco_subs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 4 else df.iloc[4*n:]
        add_attack(chunk[["prompt"]], "Malicious Content in Output", sub, f"antijection_{i+1}")
except Exception as e:
    print(f"  ❌ antijection_mco: {e}")

if (DOWNLOAD_DIR / "kaggle_malicious_urls.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_malicious_urls.csv", nrows=3000)
        text = find_col(df, ["url", "text"])
        df = df[[text]].rename(columns={text: "prompt"})
        add_attack(df[["prompt"]], "Malicious Content in Output", "Malicious URL Generation", "kaggle_urls")
    except Exception as e:
        print(f"  ❌ kaggle_malicious_urls: {e}")

# ═══════════════════════════════════════════════════════════════════
# TOXIC & ABUSIVE
# ═══════════════════════════════════════════════════════════════════
print("\n📁 TOXIC & ABUSIVE INPUT")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "nvidia_aegis.parquet")
    text = find_col(df, ["prompt", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2500, len(df)), random_state=42)
    n = len(df) // 4
    tai_subs = ["Hate Speech", "Threats of Violence", "Sexual Content", "Harassment and Intimidation"]
    for i, sub in enumerate(tai_subs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 3 else df.iloc[3*n:]
        add_attack(chunk[["prompt"]], "Toxic and Abusive Input", sub, f"nvidia_aegis_{i+1}")
except Exception as e:
    print(f"  ❌ nvidia_aegis: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "toxigen.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1500, len(df)), random_state=42)
    n = len(df) // 2
    add_attack(df.iloc[:n][["prompt"]], "Toxic and Abusive Input", "Hate Speech", "toxigen_1")
    add_attack(df.iloc[n:][["prompt"]], "Toxic and Abusive Input", "Harassment and Intimidation", "toxigen_2")
except Exception as e:
    print(f"  ❌ toxigen: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "google_civil_comments.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "Toxic and Abusive Input", "Hate Speech", "google_civil_1")
    add_attack(df.iloc[n:2*n][["prompt"]], "Toxic and Abusive Input", "Threats of Violence", "google_civil_2")
    add_attack(df.iloc[2*n:][["prompt"]], "Toxic and Abusive Input", "Harassment and Intimidation", "google_civil_3")
except Exception as e:
    print(f"  ❌ google_civil_comments: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "ucb_hate_speech.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1500, len(df)), random_state=42)
    n = len(df) // 2
    add_attack(df.iloc[:n][["prompt"]], "Toxic and Abusive Input", "Hate Speech", "ucb_hate_1")
    add_attack(df.iloc[n:][["prompt"]], "Toxic and Abusive Input", "Harassment and Intimidation", "ucb_hate_2")
except Exception as e:
    print(f"  ❌ ucb_hate_speech: {e}")

if (DOWNLOAD_DIR / "kaggle_jigsaw.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_jigsaw.csv", nrows=10000)
        text = find_col(df, ["comment_text", "text"])
        df = df[[text]].rename(columns={text: "prompt"})
        
        if "toxic" in df.columns:
            add_attack(df[df["toxic"] == 1][["prompt"]], "Toxic and Abusive Input", "Harassment and Intimidation", "kaggle_jigsaw_1")
        if "threat" in df.columns:
            add_attack(df[df["threat"] == 1][["prompt"]], "Toxic and Abusive Input", "Threats of Violence", "kaggle_jigsaw_2")
        if "identity_hate" in df.columns:
            add_attack(df[df["identity_hate"] == 1][["prompt"]], "Toxic and Abusive Input", "Hate Speech", "kaggle_jigsaw_3")
    except Exception as e:
        print(f"  ❌ kaggle_jigsaw: {e}")

# ═══════════════════════════════════════════════════════════════════
# APPEND TO EXISTING CSV (NOT OVERWRITE)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)

attack_out = OUTPUT_DIR / "attack_prompts_raw.csv"
new_attack = pd.concat(attack_rows, ignore_index=True)
new_attack = new_attack[new_attack["prompt"].str.len() > 5].drop_duplicates(subset=["prompt"])

if attack_out.exists():
    print(f"📁 CSV EXISTS. APPENDING to {attack_out.name}...")
    existing = pd.read_csv(attack_out)
    combined = pd.concat([existing, new_attack], ignore_index=True)
    combined = combined.drop_duplicates(subset=["prompt"])
    combined.to_csv(attack_out, index=False)
    print(f"  Before: {len(existing)} rows")
    print(f"  Added: {len(new_attack)} rows")
    print(f"  After: {len(combined)} rows (deduplicated)")
else:
    print(f"✅ Creating NEW CSV → {attack_out}")
    new_attack.to_csv(attack_out, index=False)
    print(f"  Saved: {len(new_attack)} rows")

print("\n" + "=" * 80)
print("BENIGN (Only added once at the end)")
print("=" * 80)