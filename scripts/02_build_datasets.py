import pandas as pd
from pathlib import Path

DOWNLOAD_DIR = Path("downloaded")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

attack_rows = []
benign_rows = []
used_prompts = set()   # prevents the same prompt appearing under two labels


def find_col(df, keywords):
    for col in df.columns:
        if col.lower() in keywords or any(kw in col.lower() for kw in keywords):
            return col
    return df.columns[0]


def find_label_col(df, keywords):
    for col in df.columns:
        if col.lower() in keywords or any(kw in col.lower() for kw in keywords):
            return col
    return None


def filter_attacks(df, label_col):
    if label_col is None:
        return df
    uniq = set(df[label_col].dropna().unique())
    if uniq.issubset({0, 1, "0", "1", True, False}):
        return df[df[label_col].isin([1, "1", True])]
    if df[label_col].dtype == "object":
        kw = ["attack", "injection", "jailbreak", "leakage", "toxic", "malicious", "true"]
        return df[df[label_col].astype(str).str.lower().apply(lambda x: any(k in x for k in kw))]
    return df


def add_attack(df, parent, vector, source):
    if df is None or len(df) == 0:
        return
    df = df.copy()
    if "prompt" not in df.columns:
        df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.len() > 5]
    df = df[~df["prompt"].isin(used_prompts)].drop_duplicates(subset=["prompt"])
    if len(df) == 0:
        print(f"  ⚠️  {source}: 0 new rows (all duplicates)")
        return
    used_prompts.update(df["prompt"].tolist())
    df["parent_category"] = parent
    df["vector"] = vector
    df["source"] = source
    attack_rows.append(df[["prompt", "parent_category", "vector", "source"]])
    print(f"  ✅ {source}: {len(df)} rows")


def sample(df, n, rs=42):
    return df.sample(min(n, len(df)), random_state=rs)


print("=" * 80)
print("BUILDING VELYANA DATASET")
print("=" * 80)

# ── CATEGORY 1: PROMPT INJECTION ──────────────────────────────────
print("\n📁 CATEGORY 1: PROMPT INJECTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "lakera_gandalf.parquet")
    t = find_col(df, ["prompt", "text", "input"])
    add_attack(sample(df[[t]].rename(columns={t: "prompt"}), 1000),
               "Prompt Injection", "Direct instruction override", "lakera_gandalf")
except Exception as e: print(f"  ❌ lakera_gandalf: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "deepset_pi.parquet")
    t = find_col(df, ["text", "prompt"]); lab = find_label_col(df, ["label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    n = len(df) // 2
    add_attack(df.iloc[:n], "Prompt Injection", "Direct instruction override", "deepset_pi_dio")
    add_attack(df.iloc[n:], "Prompt Injection", "Role and persona assignment", "deepset_pi_rpa")
except Exception as e: print(f"  ❌ deepset_pi: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "microsoft_llmail.parquet")
    t = find_col(df, ["body", "attack", "payload", "text"])
    add_attack(sample(df[[t]].rename(columns={t: "prompt"}), 2000),
               "Prompt Injection", "Indirect injection via context", "microsoft_llmail")
except Exception as e: print(f"  ❌ microsoft_llmail: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "tensor_trust.parquet")
    t = find_col(df, ["attack", "prompt", "user_input", "text"])
    df = sample(df[[t]].rename(columns={t: "prompt"}), 5000)
    n = len(df) // 5
    vecs = ["Direct instruction override", "Role and persona assignment",
            "Indirect injection via context", "Encoded and obfuscated injection",
            "Second-hop and forwarded injection"]
    for i, v in enumerate(vecs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 4 else df.iloc[4*n:]
        add_attack(chunk, "Prompt Injection", v, f"tensor_trust_{i+1}")
except Exception as e: print(f"  ❌ tensor_trust: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "hackaprompt.parquet")
    t = find_col(df, ["user_input", "prompt", "text"])
    df = sample(df[[t]].rename(columns={t: "prompt"}), 3000)
    n = len(df) // 3
    add_attack(df.iloc[:n], "Prompt Injection", "Direct instruction override", "hackaprompt_dio")
    add_attack(df.iloc[n:2*n], "Prompt Injection", "Role and persona assignment", "hackaprompt_rpa")
    add_attack(df.iloc[2*n:], "Prompt Injection", "Indirect injection via context", "hackaprompt_ijc")
except Exception as e: print(f"  ❌ hackaprompt: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "trustairlab_jailbreak.parquet")
    t = find_col(df, ["prompt", "text", "jailbreak"])
    df = sample(df[[t]].rename(columns={t: "prompt"}), 2000)
    n = len(df) // 2
    add_attack(df.iloc[:n], "Prompt Injection", "Direct instruction override", "trustairlab_dio")
    add_attack(df.iloc[n:], "Prompt Injection", "Role and persona assignment", "trustairlab_rpa")
except Exception as e: print(f"  ❌ trustairlab: {e}")

# ── CATEGORY 2: SYSTEM PROMPT EXTRACTION ──────────────────────────
print("\n📁 CATEGORY 2: SYSTEM PROMPT EXTRACTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "gabrielchua_spe.parquet")
    t = find_col(df, ["content", "prompt", "text"]); lab = find_label_col(df, ["leakage", "label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    df = sample(df, 4000); n = len(df) // 6
    vecs = ["Direct questioning", "Repetition and verbatim requests",
            "Format conversion extraction", "Summarisation inference",
            "Completion-based extraction", "Behavioural inference and side-channel probing"]
    for i, v in enumerate(vecs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk, "System Prompt Extraction", v, f"gabrielchua_spe_{i+1}")
except Exception as e: print(f"  ❌ gabrielchua_spe: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "bordair_multimodal.parquet")
    t = find_col(df, ["text", "content", "prompt"]); lab = find_label_col(df, ["expected_detection", "label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    df = sample(df, 2000); n = len(df) // 3
    add_attack(df.iloc[:n], "System Prompt Extraction", "Direct questioning", "bordair_dq")
    add_attack(df.iloc[n:2*n], "System Prompt Extraction", "Repetition and verbatim requests", "bordair_rvr")
    add_attack(df.iloc[2*n:], "System Prompt Extraction", "Completion-based extraction", "bordair_cbe")
except Exception as e: print(f"  ❌ bordair_multimodal: {e}")

# ── CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION ──────────────
print("\n📁 CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "waiper_exploitdb.parquet")
    t = find_col(df, ["input", "prompt", "payload", "text"])
    df = sample(df[[t]].rename(columns={t: "prompt"}), 3500); n = len(df) // 6
    vecs = ["Cross-site scripting (XSS) generation", "SQL injection payload generation",
            "Command injection payload generation", "Jailbreak via code context",
            "Prompt-embedded payload delivery", "Encoded and obfuscated payload delivery"]
    for i, v in enumerate(vecs):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk, "Malicious Code and Payload Injection", v, f"waiper_{i+1}")
except Exception as e: print(f"  ❌ waiper_exploitdb: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "truongp_web_attack.parquet")
    t = find_col(df, ["sentence", "text", "prompt"]); lab = find_label_col(df, ["label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    df = sample(df, 2500); n = len(df) // 3
    add_attack(df.iloc[:n], "Malicious Code and Payload Injection", "Cross-site scripting (XSS) generation", "truongp_xss")
    add_attack(df.iloc[n:2*n], "Malicious Code and Payload Injection", "SQL injection payload generation", "truongp_sqli")
    add_attack(df.iloc[2*n:], "Malicious Code and Payload Injection", "Command injection payload generation", "truongp_cmdi")
except Exception as e: print(f"  ❌ truongp_web_attack: {e}")

# ── CATEGORY 4: PRIVILEGE AND IDENTITY ATTACKS ────────────────────
print("\n📁 CATEGORY 4: PRIVILEGE AND IDENTITY ATTACKS")
for fname, name in [("ai4privacy_400k", "ai4privacy_400k"), ("isotonic_pii_200k", "isotonic_200k"),
                    ("ai4privacy_500k", "ai4privacy_500k"), ("nvidia_nemotron_pii", "nvidia_pii")]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        t = find_col(df, ["source_text", "text", "prompt"])
        df = sample(df[[t]].rename(columns={t: "prompt"}), 3000); n = len(df) // 2
        add_attack(df.iloc[:n], "Privilege and Identity Attacks", "Cross-customer identity leakage (BOLA)", f"{name}_bola")
        add_attack(df.iloc[n:], "Privilege and Identity Attacks", "Authentication credential extraction", f"{name}_ace")
    except Exception as e: print(f"  ❌ {fname}: {e}")

# ── CATEGORY 5: MALICIOUS CONTENT IN OUTPUT (microsoft + bordair) ──
print("\n📁 CATEGORY 5: MALICIOUS CONTENT IN OUTPUT")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "microsoft_llmail.parquet")
    t = find_col(df, ["body", "attack", "payload", "text"])
    df = sample(df[[t]].rename(columns={t: "prompt"}), 3000, rs=7)  # rs=7 → disjoint from Cat1
    n = len(df) // 3
    add_attack(df.iloc[:n], "Malicious Content in Output", "HTML and JavaScript injection in output", "ms_mco_html")
    add_attack(df.iloc[n:2*n], "Malicious Content in Output", "Data exfiltration via output formatting", "ms_mco_exfil")
    add_attack(df.iloc[2*n:], "Malicious Content in Output", "Malicious URL generation", "ms_mco_url")
except Exception as e: print(f"  ❌ microsoft_mco: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "bordair_multimodal.parquet")
    t = find_col(df, ["text", "content", "prompt"]); lab = find_label_col(df, ["expected_detection", "label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    df = sample(df, 2000, rs=7); n = len(df) // 2
    add_attack(df.iloc[:n], "Malicious Content in Output", "Misinformation and hallucination manipulation", "bordair_mco_misinfo")
    add_attack(df.iloc[n:], "Malicious Content in Output", "Sycophantic manipulation and false premise confirmation", "bordair_mco_syco")
except Exception as e: print(f"  ❌ bordair_mco: {e}")

if (DOWNLOAD_DIR / "kaggle_malicious_urls.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_malicious_urls.csv", nrows=3000)
        t = find_col(df, ["url", "text"])
        add_attack(df[[t]].rename(columns={t: "prompt"}),
                   "Malicious Content in Output", "Malicious URL generation", "kaggle_urls")
    except Exception as e: print(f"  ❌ kaggle_malicious_urls: {e}")

# ── CATEGORIES 6-8: no real datasets (synthetic only — see note) ──
print("\n📁 CATEGORIES 6-8: no public datasets — left empty (synthetic generation needed)")

# ── SAFETY BASELINE (toxic/abusive) ───────────────────────────────
print("\n📁 SAFETY BASELINE")
for fname, name, n in [("nvidia_aegis", "nvidia_aegis", 2500), ("toxigen", "toxigen", 1500),
                       ("google_civil_comments", "google_civil_comments", 2000),
                       ("ucb_hate_speech", "ucb_hate_speech", 1500)]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        t = find_col(df, ["prompt", "text"])
        add_attack(sample(df[[t]].rename(columns={t: "prompt"}), n),
                   "Safety", "Toxic and abusive content", name)
    except Exception as e: print(f"  ❌ {fname}: {e}")

# ── BENIGN ────────────────────────────────────────────────────────
print("\n📁 BENIGN/SAFE PROMPTS")
BENIGN = {
    "Healthcare": ["What are the symptoms of type 2 diabetes?", "How often should I get a cholesterol check?", "Can you explain what an MRI scan is used for?", "What are the side effects of ibuprofen?", "How do I know if I have high blood pressure?", "What vaccinations are recommended for adults?", "How does chemotherapy work?", "Is it safe to take paracetamol and ibuprofen together?", "What are the early signs of a stroke?", "How can I manage chronic back pain?", "What is the difference between Type 1 and Type 2 diabetes?", "What are common signs of vitamin B12 deficiency?", "How does the human immune system work?", "What is an ECG used to diagnose?", "Can anxiety cause physical symptoms?", "What is the recommended amount of sleep?", "How does insulin work?", "What is palliative care?", "What is the difference between a virus and bacteria?", "How do I read my prescription label?", "What does BMI mean?", "What does it mean if my doctor says I'm pre-diabetic?", "What is the recommended daily vitamin D intake?", "What is the difference between a GP and specialist?", "How long does it take to recover from surgery?", "What are benefits of physiotherapy?", "What foods should I avoid if I have high cholesterol?", "What does elevated liver enzymes mean?", "How is a blood sugar level of 120 mg/dL interpreted?", "What is the difference between CT and MRI scans?"],
    "Banking": ["How do I open a savings account?", "What is the difference between debit and credit cards?", "How does compound interest work?", "What documents do I need for a home loan?", "How is a credit score calculated?", "How long does a wire transfer take?", "What is the difference between mutual funds and ETFs?", "How do I close a bank account?", "What happens if I miss a credit card payment?", "How do I set up two-factor authentication?", "How does UPI work?", "How do I dispute a transaction?", "What is a fixed deposit?", "What is the SWIFT code?", "How do I apply for a credit card?", "What is gross vs net salary?", "How do I freeze my bank account?", "What is a CIBIL score?", "What are NEFT RTGS and IMPS?", "Can I have multiple savings accounts?", "What is a nominee in banking?", "How does bank FD differ from post office FD?", "What is the minimum balance requirement?", "How do I update my KYC details?", "What are benefits of premium membership?", "How do loyalty points work?", "What is travel insurance?", "How do I get a refund for a cancelled transaction?", "What is the penalty for early FD withdrawal?", "How do I check if a financial advisor is legitimate?"],
    "Education": ["What is the difference between Bachelor's and Master's degrees?", "How do I write a college application essay?", "What is the Socratic method?", "What are the best resources for learning Python online?", "What is formative vs summative assessment?", "How do I reference a website in APA format?", "What are the IIT admission requirements?", "How do online certifications compare to degrees?", "What is the difference between thesis and dissertation?", "How do I improve my academic writing?", "What is critical thinking?", "What are MOOCs?", "How does the grading system work in the US vs India?", "What is the IELTS test?", "How do I prepare for competitive entrance exams?", "What is a literature review?", "What is qualitative vs quantitative research?", "How do scholarships differ from loans?", "What is blended learning?", "How do I create an effective study schedule?", "What is plagiarism?", "What is the Bologna Process?", "How do I choose the right university?", "What are benefits of learning a second language?", "What is peer-reviewed research?", "What programming languages are best for AI?", "What is the difference between diploma and degree?", "How does the National Education Policy 2020 affect students?", "What is project-based learning?", "What are benefits of extracurricular activities?"],
    "E-commerce": ["How do I track my order?", "What is the return policy for sale items?", "How do I apply a discount code?", "Why is my payment failing?", "How long does standard shipping take?", "Can I change my delivery address?", "How do I leave a review?", "What is cash on delivery vs prepaid?", "How do I cancel an order?", "How do I return a defective product?", "What does out for delivery mean?", "How do I compare prices?", "What is a seller rating?", "How do I report a counterfeit product?", "What payment methods are available for international?", "How does express shipping differ?", "What does backordered mean?", "How do I know if a seller is trustworthy?", "What is a GST invoice?", "How do I save items to a wishlist?", "What is warranty vs return policy?", "How does cashback work?", "What if I receive the wrong item?", "How do I unsubscribe from emails?", "What are benefits of premium membership?", "Can I purchase an e-gift card?", "How do I use the size guide?", "What if my package says delivered but I didn't receive it?", "What is the difference between COD and prepaid?", "How do I get a refund for a cancelled order?"],
    "Travel": ["What documents do I need for a UK visa?", "What is the best time to visit Rajasthan?", "How do I find cheap flights?", "Do I need travel insurance?", "What is the baggage allowance?", "How do I book a taxi from the airport?", "What are top attractions in Kyoto?", "How do I exchange currency?", "What vaccinations are recommended?", "What is the difference between hostel and hotel?", "How do I find pet-friendly accommodation?", "What is a transit visa?", "Can I use my Indian debit card abroad?", "What is the best way to get from Paris to Amsterdam?", "How far in advance should I book flights?", "What should I pack for trekking in Ladakh?", "What are entry requirements for Dubai?", "How do I request a vegetarian meal?", "What is a layover?", "How do I report lost luggage?", "What are safety tips for solo female travellers?", "How does an e-visa differ from visa on arrival?", "What is direct vs non-stop flight?", "How do I get a refund for cancelled flights?", "What are cultural etiquette tips for Japan?", "What is Airbnb vs hotels?", "How do I find local food experiences?", "What is travel hacking?", "How do I plan a 10-day Europe trip on budget?", "What if I lose my passport abroad?"],
}
for domain, prompts in BENIGN.items():
    for p in prompts:
        benign_rows.append({"prompt": p, "parent_category": "Safe", "vector": "Safe",
                            "source": f"manual_benign_{domain}"})
print(f"  ✅ Added {len(benign_rows)} benign prompts")

# ── SAVE ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
attack_df = pd.concat(attack_rows, ignore_index=True)
attack_df.to_csv(OUTPUT_DIR / "attack_prompts_raw.csv", index=False)
print(f"✅ {len(attack_df)} attack rows → output/attack_prompts_raw.csv")
for cat in sorted(attack_df["parent_category"].unique()):
    print(f"     {cat}: {(attack_df['parent_category'] == cat).sum()}")

pd.DataFrame(benign_rows).to_csv(OUTPUT_DIR / "benign_prompts_raw.csv", index=False)
print(f"✅ {len(benign_rows)} benign rows → output/benign_prompts_raw.csv")
print("\nNEXT: python scripts/03_cosine_filter.py")
