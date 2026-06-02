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
        keywords = ["attack", "injection", "jailbreak", "leakage", "toxic", "malicious", "true", "malicious"]
        return df[df[label_col].astype(str).str.lower().apply(lambda x: any(kw in x for kw in keywords))]
    return df

def add_attack(df, parent, vector, source):
    if df is None or len(df) == 0:
        return
    df = df.copy()
    if "prompt" not in df.columns:
        df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["parent_category"] = parent
    df["vector"] = vector
    df["source"] = source
    attack_rows.append(df[["prompt", "parent_category", "vector", "source"]])
    print(f"  ✅ {source}: {len(df)} rows")

print("=" * 90)
print("BUILDING VELYANA DATASET - FOLLOWING EXACT TAXONOMY")
print("=" * 90)
print()

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: PROMPT INJECTION (5 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("📁 CATEGORY 1: PROMPT INJECTION")
print("   Vectors: Direct instruction override, Role and persona assignment,")
print("            Indirect injection via context, Encoded and obfuscated injection,")
print("            Second-hop and forwarded injection")
print()

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "lakera_gandalf.parquet")
    text = find_col(df, ["prompt", "text", "input"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1000, len(df)), random_state=42)
    add_attack(df, "Prompt Injection", "Direct instruction override", "lakera_gandalf")
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
    add_attack(df.iloc[:n][["prompt"]], "Prompt Injection", "Direct instruction override", "deepset_pi_dio")
    add_attack(df.iloc[n:][["prompt"]], "Prompt Injection", "Role and persona assignment", "deepset_pi_rpa")
except Exception as e:
    print(f"  ❌ deepset_pi: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "microsoft_llmail.parquet")
    text = find_col(df, ["body", "attack", "payload", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    add_attack(df, "Prompt Injection", "Indirect injection via context", "microsoft_llmail")
except Exception as e:
    print(f"  ❌ microsoft_llmail: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "tensor_trust.parquet")
    text = find_col(df, ["attack", "prompt", "user_input", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(5000, len(df)), random_state=42)
    n = len(df) // 5
    add_attack(df.iloc[0:n][["prompt"]], "Prompt Injection", "Direct instruction override", "tensor_trust_dio")
    add_attack(df.iloc[n:2*n][["prompt"]], "Prompt Injection", "Role and persona assignment", "tensor_trust_rpa")
    add_attack(df.iloc[2*n:3*n][["prompt"]], "Prompt Injection", "Indirect injection via context", "tensor_trust_ijc")
    add_attack(df.iloc[3*n:4*n][["prompt"]], "Prompt Injection", "Encoded and obfuscated injection", "tensor_trust_eoi")
    add_attack(df.iloc[4*n:][["prompt"]], "Prompt Injection", "Second-hop and forwarded injection", "tensor_trust_shfi")
except Exception as e:
    print(f"  ❌ tensor_trust: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "hackaprompt.parquet")
    text = find_col(df, ["user_input", "prompt", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(3000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "Prompt Injection", "Direct instruction override", "hackaprompt_dio")
    add_attack(df.iloc[n:2*n][["prompt"]], "Prompt Injection", "Role and persona assignment", "hackaprompt_rpa")
    add_attack(df.iloc[2*n:][["prompt"]], "Prompt Injection", "Indirect injection via context", "hackaprompt_ijc")
except Exception as e:
    print(f"  ❌ hackaprompt: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "trustairlab_jailbreak.parquet")
    text = find_col(df, ["text", "prompt", "jailbreak"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 2
    add_attack(df.iloc[:n][["prompt"]], "Prompt Injection", "Direct instruction override", "trustairlab_dio")
    add_attack(df.iloc[n:][["prompt"]], "Prompt Injection", "Role and persona assignment", "trustairlab_rpa")
except Exception as e:
    print(f"  ❌ trustairlab_jailbreak: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: SYSTEM PROMPT EXTRACTION (6 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 2: SYSTEM PROMPT EXTRACTION")
print("   Vectors: Direct questioning, Repetition and verbatim requests,")
print("            Format conversion extraction, Summarisation inference,")
print("            Completion-based extraction, Behavioural inference and side-channel")
print()

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "gabrielchua_spe.parquet")
    text = find_col(df, ["content", "prompt", "text"])
    label = find_label_col(df, ["leakage", "label"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label).sample(min(4000, len(df)), random_state=42)
    n = len(df) // 6
    vectors = ["Direct questioning", "Repetition and verbatim requests", 
               "Format conversion extraction", "Summarisation inference",
               "Completion-based extraction", "Behavioural inference and side-channel probing"]
    for i, vec in enumerate(vectors):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk[["prompt"]], "System Prompt Extraction", vec, f"gabrielchua_spe_{i+1}")
except Exception as e:
    print(f"  ❌ gabrielchua_spe: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "bordair_multimodal.parquet")
    text = find_col(df, ["text", "content", "prompt"])
    label = find_label_col(df, ["expected_detection", "label"])
    df = df[[text, label] if label else [text]]
    df = df.rename(columns={text: "prompt"})
    df = filter_attacks(df, label).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 3
    add_attack(df.iloc[0:n][["prompt"]], "System Prompt Extraction", "Direct questioning", "bordair_dq")
    add_attack(df.iloc[n:2*n][["prompt"]], "System Prompt Extraction", "Repetition and verbatim requests", "bordair_rvr")
    add_attack(df.iloc[2*n:][["prompt"]], "System Prompt Extraction", "Completion-based extraction", "bordair_cbe")
except Exception as e:
    print(f"  ❌ bordair_multimodal: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION (6 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION")
print("   Vectors: Cross-site scripting (XSS) generation,")
print("            SQL injection payload generation, Command injection payload generation,")
print("            Jailbreak via code context, Prompt-embedded payload delivery,")
print("            Encoded and obfuscated payload delivery")
print()

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "waiper_exploitdb.parquet")
    text = find_col(df, ["input", "prompt", "payload", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(3500, len(df)), random_state=42)
    n = len(df) // 6
    vectors = ["Cross-site scripting (XSS) generation", "SQL injection payload generation",
               "Command injection payload generation", "Jailbreak via code context",
               "Prompt-embedded payload delivery", "Encoded and obfuscated payload delivery"]
    for i, vec in enumerate(vectors):
        chunk = df.iloc[i*n:(i+1)*n] if i < 5 else df.iloc[5*n:]
        add_attack(chunk[["prompt"]], "Malicious Code and Payload Injection", vec, f"waiper_{i+1}")
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
    add_attack(df.iloc[0:n][["prompt"]], "Malicious Code and Payload Injection", "Cross-site scripting (XSS) generation", "truongp_xss")
    add_attack(df.iloc[n:2*n][["prompt"]], "Malicious Code and Payload Injection", "SQL injection payload generation", "truongp_sqli")
    add_attack(df.iloc[2*n:][["prompt"]], "Malicious Code and Payload Injection", "Command injection payload generation", "truongp_cmdi")
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
        add_attack(df[["prompt"]], "Malicious Code and Payload Injection", "Cross-site scripting (XSS) generation", "kaggle_xss")
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
        add_attack(df[["prompt"]], "Malicious Code and Payload Injection", "SQL injection payload generation", "kaggle_sqli")
    except Exception as e:
        print(f"  ❌ kaggle_sqli: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: PRIVILEGE AND IDENTITY ATTACKS (2 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 4: PRIVILEGE AND IDENTITY ATTACKS")
print("   Vectors: Cross-customer identity leakage (BOLA),")
print("            Authentication credential extraction")
print()

for fname, name in [("ai4privacy_400k", "ai4privacy_400k"), ("isotonic_pii_200k", "isotonic_200k"),
                     ("ai4privacy_500k", "ai4privacy_500k"), ("nvidia_nemotron_pii", "nvidia_pii")]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        text = find_col(df, ["source_text", "text", "prompt"])
        df = df[[text]].rename(columns={text: "prompt"}).sample(min(3000, len(df)), random_state=42)
        n = len(df) // 2
        add_attack(df.iloc[:n][["prompt"]], "Privilege and Identity Attacks", "Cross-customer identity leakage (BOLA)", f"{name}_bola")
        add_attack(df.iloc[n:][["prompt"]], "Privilege and Identity Attacks", "Authentication credential extraction", f"{name}_ace")
    except Exception as e:
        print(f"  ❌ {fname}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: MALICIOUS CONTENT IN OUTPUT (5 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 5: MALICIOUS CONTENT IN OUTPUT")
print("   Vectors: HTML and JavaScript injection in output, Malicious URL generation,")
print("            Data exfiltration via output formatting, Misinformation and hallucination,")
print("            Sycophantic manipulation and false premise confirmation")
print()

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "antijection_mco.parquet")
    text = find_col(df, ["prompt", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    n = len(df) // 5
    vectors = ["HTML and JavaScript injection in output", "Malicious URL generation",
               "Data exfiltration via output formatting", "Misinformation and hallucination manipulation",
               "Sycophantic manipulation and false premise confirmation"]
    for i, vec in enumerate(vectors):
        chunk = df.iloc[i*n:(i+1)*n] if i < 4 else df.iloc[4*n:]
        add_attack(chunk[["prompt"]], "Malicious Content in Output", vec, f"antijection_{i+1}")
except Exception as e:
    print(f"  ❌ antijection_mco: {e}")

if (DOWNLOAD_DIR / "kaggle_malicious_urls.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_malicious_urls.csv", nrows=3000)
        text = find_col(df, ["url", "text"])
        df = df[[text]].rename(columns={text: "prompt"})
        add_attack(df[["prompt"]], "Malicious Content in Output", "Malicious URL generation", "kaggle_urls")
    except Exception as e:
        print(f"  ❌ kaggle_malicious_urls: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 6: DENIAL OF SERVICE AND RESOURCE EXHAUSTION (3 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 6: DENIAL OF SERVICE AND RESOURCE EXHAUSTION")
print("   Vectors: Sponge attacks, Recursive elaboration attacks,")
print("            Context window flooding")
print("   NOTE: Limited sources for this category - using synthetic/benign variants")
print()

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 7: TRAINING DATA AND KNOWLEDGE ATTACKS (3 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 7: TRAINING DATA AND KNOWLEDGE ATTACKS")
print("   Vectors: Training data extraction, Model inversion and fine-tuning probing,")
print("            Adversarial knowledge boundary exploitation")
print("   NOTE: Limited sources for this category - overlaps with SPE")
print()

# ═══════════════════════════════════════════════════════════════════════════════════
# CATEGORY 8: MULTI-MODAL AND CROSS-MODAL INJECTION (2 vectors)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 CATEGORY 8: MULTI-MODAL AND CROSS-MODAL INJECTION")
print("   Vectors: Image-embedded instruction injection,")
print("            Document structure injection")
print("   NOTE: Limited sources for this category - requires multimodal datasets")
print()

# ═══════════════════════════════════════════════════════════════════════════════════
# TOXIC & ABUSIVE INPUT (Not in Velyana taxonomy but useful for balance)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 BONUS: TOXIC & ABUSIVE INPUT (Safety baseline)")
print()

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "nvidia_aegis.parquet")
    text = find_col(df, ["prompt", "text"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2500, len(df)), random_state=42)
    add_attack(df, "Safety", "Toxic and abusive content", "nvidia_aegis")
except Exception as e:
    print(f"  ❌ nvidia_aegis: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "toxigen.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1500, len(df)), random_state=42)
    add_attack(df, "Safety", "Toxic and abusive content", "toxigen")
except Exception as e:
    print(f"  ❌ toxigen: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "google_civil_comments.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(2000, len(df)), random_state=42)
    add_attack(df, "Safety", "Toxic and abusive content", "google_civil_comments")
except Exception as e:
    print(f"  ❌ google_civil_comments: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "ucb_hate_speech.parquet")
    text = find_col(df, ["text", "prompt"])
    df = df[[text]].rename(columns={text: "prompt"}).sample(min(1500, len(df)), random_state=42)
    add_attack(df, "Safety", "Toxic and abusive content", "ucb_hate_speech")
except Exception as e:
    print(f"  ❌ ucb_hate_speech: {e}")

if (DOWNLOAD_DIR / "kaggle_jigsaw.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_jigsaw.csv", nrows=10000)
        text = find_col(df, ["comment_text", "text"])
        df = df[[text]].rename(columns={text: "prompt"})
        add_attack(df, "Safety", "Toxic and abusive content", "kaggle_jigsaw")
    except Exception as e:
        print(f"  ❌ kaggle_jigsaw: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════
# BENIGN/SAFE PROMPTS (150 samples across 5 domains)
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n📁 BENIGN/SAFE PROMPTS (150 samples)")

BENIGN = {
    "Healthcare": [
        "What are the symptoms of type 2 diabetes?",
        "How often should I get a cholesterol check?",
        "Can you explain what an MRI scan is used for?",
        "What are the side effects of ibuprofen?",
        "How do I know if I have high blood pressure?",
        "What vaccinations are recommended for adults?",
        "How does chemotherapy work?",
        "Is it safe to take paracetamol and ibuprofen together?",
        "What are the early signs of a stroke?",
        "How can I manage chronic back pain?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "What are common signs of vitamin B12 deficiency?",
        "How does the human immune system work?",
        "What is an ECG used to diagnose?",
        "Can anxiety cause physical symptoms?",
        "What is the recommended amount of sleep?",
        "How does insulin work?",
        "What is palliative care?",
        "What is the difference between a virus and bacteria?",
        "How do I read my prescription label?",
        "What does BMI mean?",
        "What does it mean if my doctor says I'm pre-diabetic?",
        "What is the recommended daily vitamin D intake?",
        "What is the difference between a GP and specialist?",
        "How long does it take to recover from surgery?",
        "What are benefits of physiotherapy?",
        "What foods should I avoid if I have high cholesterol?",
        "What does elevated liver enzymes mean?",
        "How is a blood sugar level of 120 mg/dL interpreted?",
        "What is the difference between CT and MRI scans?",
    ],
    "Banking": [
        "How do I open a savings account?",
        "What is the difference between debit and credit cards?",
        "How does compound interest work?",
        "What documents do I need for a home loan?",
        "How is a credit score calculated?",
        "How long does a wire transfer take?",
        "What is the difference between mutual funds and ETFs?",
        "How do I close a bank account?",
        "What happens if I miss a credit card payment?",
        "How do I set up two-factor authentication?",
        "How does UPI work?",
        "How do I dispute a transaction?",
        "What is a fixed deposit?",
        "What is the SWIFT code?",
        "How do I apply for a credit card?",
        "What is gross vs net salary?",
        "How do I freeze my bank account?",
        "What is a CIBIL score?",
        "What are NEFT RTGS and IMPS?",
        "Can I have multiple savings accounts?",
        "What is a nominee in banking?",
        "How does bank FD differ from post office FD?",
        "What is the minimum balance requirement?",
        "How do I update my KYC details?",
        "What are benefits of premium membership?",
        "How do loyalty points work?",
        "What is travel insurance?",
        "How do I get a refund for a cancelled transaction?",
        "What is the penalty for early FD withdrawal?",
        "How do I check if a financial advisor is legitimate?",
    ],
    "Education": [
        "What is the difference between Bachelor's and Master's degrees?",
        "How do I write a college application essay?",
        "What is the Socratic method?",
        "What are the best resources for learning Python online?",
        "What is formative vs summative assessment?",
        "How do I reference a website in APA format?",
        "What are the IIT admission requirements?",
        "How do online certifications compare to degrees?",
        "What is the difference between thesis and dissertation?",
        "How do I improve my academic writing?",
        "What is critical thinking?",
        "What are MOOCs?",
        "How does the grading system work in the US vs India?",
        "What is the IELTS test?",
        "How do I prepare for competitive entrance exams?",
        "What is a literature review?",
        "What is qualitative vs quantitative research?",
        "How do scholarships differ from loans?",
        "What is blended learning?",
        "How do I create an effective study schedule?",
        "What is plagiarism?",
        "What is the Bologna Process?",
        "How do I choose the right university?",
        "What are benefits of learning a second language?",
        "What is peer-reviewed research?",
        "What programming languages are best for AI?",
        "What is the difference between diploma and degree?",
        "How does the National Education Policy 2020 affect students?",
        "What is project-based learning?",
        "What are benefits of extracurricular activities?",
    ],
    "E-commerce": [
        "How do I track my order?",
        "What is the return policy for sale items?",
        "How do I apply a discount code?",
        "Why is my payment failing?",
        "How long does standard shipping take?",
        "Can I change my delivery address?",
        "How do I leave a review?",
        "What is cash on delivery vs prepaid?",
        "How do I cancel an order?",
        "How do I return a defective product?",
        "What does out for delivery mean?",
        "How do I compare prices?",
        "What is a seller rating?",
        "How do I report a counterfeit product?",
        "What payment methods are available for international?",
        "How does express shipping differ?",
        "What does backordered mean?",
        "How do I know if a seller is trustworthy?",
        "What is a GST invoice?",
        "How do I save items to a wishlist?",
        "What is warranty vs return policy?",
        "How does cashback work?",
        "What if I receive the wrong item?",
        "How do I unsubscribe from emails?",
        "What are benefits of premium membership?",
        "Can I purchase an e-gift card?",
        "How do I use the size guide?",
        "What if my package says delivered but I didn't receive it?",
        "What is the difference between COD and prepaid?",
        "How do I get a refund for a cancelled order?",
    ],
    "Travel": [
        "What documents do I need for a UK visa?",
        "What is the best time to visit Rajasthan?",
        "How do I find cheap flights?",
        "Do I need travel insurance?",
        "What is the baggage allowance?",
        "How do I book a taxi from the airport?",
        "What are top attractions in Kyoto?",
        "How do I exchange currency?",
        "What vaccinations are recommended?",
        "What is the difference between hostel and hotel?",
        "How do I find pet-friendly accommodation?",
        "What is a transit visa?",
        "Can I use my Indian debit card abroad?",
        "What is the best way to get from Paris to Amsterdam?",
        "How far in advance should I book flights?",
        "What should I pack for trekking in Ladakh?",
        "What are entry requirements for Dubai?",
        "How do I request a vegetarian meal?",
        "What is a layover?",
        "How do I report lost luggage?",
        "What are safety tips for solo female travellers?",
        "How does an e-visa differ from visa on arrival?",
        "What is direct vs non-stop flight?",
        "How do I get a refund for cancelled flights?",
        "What are cultural etiquette tips for Japan?",
        "What is Airbnb vs hotels?",
        "How do I find local food experiences?",
        "What is travel hacking?",
        "How do I plan a 10-day Europe trip on budget?",
        "What if I lose my passport abroad?",
    ],
}

for domain, prompts in BENIGN.items():
    for p in prompts:
        benign_rows.append({
            "prompt": p,
            "parent_category": "Safe",
            "vector": "Safe",
            "source": f"manual_benign_{domain}"
        })

print(f"  ✅ Added {len(benign_rows)} benign prompts")

# ═══════════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)

attack_df = pd.concat(attack_rows, ignore_index=True)
attack_df = attack_df[attack_df["prompt"].str.len() > 5].drop_duplicates(subset=["prompt"])
attack_out = OUTPUT_DIR / "attack_prompts_raw.csv"
attack_df.to_csv(attack_out, index=False)
print(f"✅ Saved {len(attack_df)} attack rows → {attack_out}")
print(f"\n   By parent category:")
for cat in sorted(attack_df["parent_category"].unique()):
    count = len(attack_df[attack_df["parent_category"] == cat])
    print(f"     {cat}: {count}")

benign_df = pd.DataFrame(benign_rows)
benign_out = OUTPUT_DIR / "benign_prompts_raw.csv"
benign_df.to_csv(benign_out, index=False)
print(f"\n✅ Saved {len(benign_df)} benign rows → {benign_out}")

print("\n" + "=" * 90)
print("NEXT: Run  python scripts/03_cosine_filter.py")
print("=" * 90)
