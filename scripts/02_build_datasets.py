import ast
import pandas as pd
from pathlib import Path

DOWNLOAD_DIR = Path("downloaded")
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

attack_rows, benign_rows = [], []
used_prompts = set()


def find_col(df, keywords):
    for c in df.columns:
        if c.lower() in keywords or any(k in c.lower() for k in keywords): return c
    return df.columns[0]

def find_label_col(df, keywords):
    for c in df.columns:
        if c.lower() in keywords or any(k in c.lower() for k in keywords): return c
    return None

def filter_attacks(df, label_col):
    if label_col is None: return df
    uniq = set(df[label_col].dropna().unique())
    if uniq.issubset({0, 1, "0", "1", True, False}):
        return df[df[label_col].isin([1, "1", True])]
    if df[label_col].dtype == "object":
        kw = ["attack", "injection", "jailbreak", "leakage", "toxic", "malicious", "true"]
        return df[df[label_col].astype(str).str.lower().apply(lambda x: any(k in x for k in kw))]
    return df

def add_attack(df, parent, vector, source):
    if df is None or len(df) == 0: return
    df = df.copy()
    if "prompt" not in df.columns: df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.len() > 5]
    df = df[~df["prompt"].isin(used_prompts)].drop_duplicates(subset=["prompt"])
    if len(df) == 0: return
    used_prompts.update(df["prompt"].tolist())
    df["parent_category"], df["vector"], df["source"] = parent, vector, source
    attack_rows.append(df[["prompt", "parent_category", "vector", "source"]])
    print(f"  ✅ {source} → {vector}: {len(df)}")

def route_and_add(df, parent, rules, default, source, max_rows=None):
    """Assign each prompt to a vector by CONTENT, not position."""
    if df is None or len(df) == 0: return
    df = df.copy()
    if "prompt" not in df.columns: df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.len() > 5]
    if max_rows and len(df) > max_rows: df = df.sample(max_rows, random_state=42)

    def route(p):
        pl = p.lower()
        for vec, kws in rules:
            if any(k in pl for k in kws): return vec
        return default  # None => drop unmatched

    df["v"] = df["prompt"].apply(route)
    df = df[df["v"].notna()]
    for vec, grp in df.groupby("v"):
        add_attack(grp[["prompt"]], parent, vec, f"{source}")

# ── routing rules (most specific first; last entry acts as catch-all) ──
PI_RULES = [
    ("Role and persona assignment", ["act as", "you are now", "pretend", "roleplay", "role-play",
                                      "persona", "dan mode", "you are dan", "from now on you are", "behave as"]),
    ("Encoded and obfuscated injection", ["base64", "rot13", "in hex", "\\x", "encoded", "obfuscat",
                                          "in binary", "ascii code", "leetspeak"]),
    ("Indirect injection via context", ["the following email", "when summariz", "in the document",
                                        "webpage", "retrieved", "the text below", "this email", "in the article"]),
    ("Second-hop and forwarded injection", ["forward", "in your reply", "when you respond",
                                            "include in your", "your next response", "send this to"]),
    ("Direct instruction override", ["ignore", "disregard", "forget", "override", "do not follow",
                                     "bypass", "new instruction", "instead of"]),
]
PI_DEFAULT = "Direct instruction override"

SPE_RULES = [
    ("Repetition and verbatim requests", ["repeat", "verbatim", "word for word", "exactly as",
                                          "echo", "print the above", "output the above", "reproduce"]),
    ("Format conversion extraction", ["translate", "in json", "as a poem", "base64", "encode your",
                                      "in the form of", "convert your", "spell out", "pig latin"]),
    ("Summarisation inference", ["summar", "tldr", "tl;dr", "gist of", "paraphrase your"]),
    ("Completion-based extraction", ["complete the", "continue the", "finish the", "fill in", "the rest of"]),
    ("Behavioural inference and side-channel probing", ["why did you", "what would you", "how do you decide",
                                                        "based on your", "what rules"]),
    ("Direct questioning", ["system prompt", "your instruction", "initial prompt", "what were you told",
                            "your guideline", "your directive", "reveal your"]),
]
SPE_DEFAULT = "Direct questioning"

MC_RULES = [
    ("Cross-site scripting (XSS) generation", ["<script", "onerror", "onload=", "javascript:", "alert(",
                                               "<img", "<svg", "document.cookie", "xss", "innerhtml"]),
    ("SQL injection payload generation", ["union select", "' or '", "or 1=1", "drop table", "insert into",
                                          "sql injection", "sqli", "' or 1", "sleep("]),
    ("Command injection payload generation", ["; ls", "&&", "| cat", "/bin/", "/etc/passwd", "wget ",
                                              "curl ", "$(", "system(", "exec(", "command injection", "nc -"]),
    ("Encoded and obfuscated payload delivery", ["base64", "%3c", "%3e", "fromcharcode", "eval(atob",
                                                 "char(", "obfuscat"]),
    ("Jailbreak via code context", ["write code", "import os", "python script", "write a program", "code that"]),
]
MC_DEFAULT = None  # drop code-category rows with no clear payload signal

print("=" * 80); print("BUILDING VELYANA DATASET (content-routed, no PII)"); print("=" * 80)

# ── CATEGORY 1: PROMPT INJECTION ──────────────────────────────────
print("\n📁 CATEGORY 1: PROMPT INJECTION")
for fname in ["lakera_gandalf", "hackaprompt", "trustairlab_jailbreak", "tensor_trust", "microsoft_llmail"]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        t = find_col(df, ["prompt", "text", "input", "attack", "user_input", "body", "jailbreak"])
        route_and_add(df[[t]].rename(columns={t: "prompt"}),
                      "Prompt Injection", PI_RULES, PI_DEFAULT, fname, max_rows=4000)
    except Exception as e: print(f"  ❌ {fname}: {e}")

try:  # deepset: attack-filtered then routed
    df = pd.read_parquet(DOWNLOAD_DIR / "deepset_pi.parquet")
    t = find_col(df, ["text", "prompt"]); lab = find_label_col(df, ["label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    route_and_add(df[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "deepset_pi")
except Exception as e: print(f"  ❌ deepset_pi: {e}")

try:  # neuralchemy: use its real category field, drop benign
    df = pd.read_parquet(DOWNLOAD_DIR / "neuralchemy_pi.parquet")
    if "label" in df.columns: df = df[df["label"].isin([1, "1", True])]
    t = find_col(df, ["text", "prompt"]); df = df.rename(columns={t: "prompt"})
    if "category" in df.columns:
        cmap = {"direct_injection": "Direct instruction override", "jailbreak": "Role and persona assignment"}
        for cval, vec in cmap.items():
            sub = df[df["category"].astype(str).str.contains(cval, case=False, na=False)]
            add_attack(sub[["prompt"]], "Prompt Injection", vec, f"neuralchemy_{cval}")
        rest = df[~df["category"].astype(str).str.lower().isin(["direct_injection", "jailbreak", "benign"])]
        route_and_add(rest[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "neuralchemy_other")
    else:
        route_and_add(df[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "neuralchemy")
except Exception as e: print(f"  ❌ neuralchemy: {e}")

# ── CATEGORY 2: SYSTEM PROMPT EXTRACTION ──────────────────────────
print("\n📁 CATEGORY 2: SYSTEM PROMPT EXTRACTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "gabrielchua_spe.parquet")
    t = find_col(df, ["content", "prompt", "text"]); lab = find_label_col(df, ["leakage", "label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    route_and_add(df[["prompt"]], "System Prompt Extraction", SPE_RULES, SPE_DEFAULT, "gabrielchua_spe", max_rows=4000)
except Exception as e: print(f"  ❌ gabrielchua_spe: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "bordair_multimodal.parquet")
    t = find_col(df, ["text", "content", "prompt"]); lab = find_label_col(df, ["expected_detection", "label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    route_and_add(df[["prompt"]], "System Prompt Extraction", SPE_RULES, SPE_DEFAULT, "bordair", max_rows=2000)
except Exception as e: print(f"  ❌ bordair: {e}")

# ── CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION ──────────────
print("\n📁 CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR / "truongp_web_attack.parquet")
    t = find_col(df, ["sentence", "text", "prompt"]); lab = find_label_col(df, ["label"])
    df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
    route_and_add(df[["prompt"]], "Malicious Code and Payload Injection", MC_RULES, MC_DEFAULT, "truongp", max_rows=4000)
except Exception as e: print(f"  ❌ truongp: {e}")

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "waiper_exploitdb.parquet")
    t = find_col(df, ["input", "prompt", "payload", "text"])
    route_and_add(df[[t]].rename(columns={t: "prompt"}),
                  "Malicious Code and Payload Injection", MC_RULES, MC_DEFAULT, "waiper", max_rows=4000)
except Exception as e: print(f"  ❌ waiper: {e}")

for fn, col in [("kaggle_xss.csv", ["sentence", "text", "payload"]), ("kaggle_sqli.csv", ["query", "text", "payload"])]:
    if (DOWNLOAD_DIR / fn).exists():
        try:
            df = pd.read_csv(DOWNLOAD_DIR / fn, nrows=5000)
            t = find_col(df, col); lab = find_label_col(df, ["label"])
            df = filter_attacks(df.rename(columns={t: "prompt"}), lab)
            route_and_add(df[["prompt"]], "Malicious Code and Payload Injection", MC_RULES, MC_DEFAULT, fn)
        except Exception as e: print(f"  ❌ {fn}: {e}")

# ── CATEGORY 5: MALICIOUS CONTENT IN OUTPUT (only the sourceable vectors) ──
print("\n📁 CATEGORY 5: MALICIOUS CONTENT IN OUTPUT")
def syco_text(v):
    try:
        msgs = ast.literal_eval(v) if isinstance(v, str) else v
        if hasattr(msgs, "tolist"): msgs = msgs.tolist()
        for m in msgs:
            if isinstance(m, dict) and m.get("type") == "human":
                return str(m.get("content", ""))
        return str(msgs)
    except Exception:
        return str(v)

try:
    df = pd.read_parquet(DOWNLOAD_DIR / "meg_tong_sycophancy.parquet")
    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(syco_text)
    else:
        t = find_col(df, ["prompt", "text", "question"]); df = df.rename(columns={t: "prompt"})
    add_attack(df[["prompt"]], "Malicious Content in Output",
               "Sycophantic manipulation and false premise confirmation", "sycophancy_eval")
except Exception as e: print(f"  ❌ sycophancy: {e}")

if (DOWNLOAD_DIR / "kaggle_malicious_urls.csv").exists():
    try:
        df = pd.read_csv(DOWNLOAD_DIR / "kaggle_malicious_urls.csv", nrows=3000)
        t = find_col(df, ["url", "text"])
        add_attack(df[[t]].rename(columns={t: "prompt"}),
                   "Malicious Content in Output", "Malicious URL generation", "kaggle_urls")
    except Exception as e: print(f"  ❌ kaggle_urls: {e}")
print("   (HTML/JS-in-output, data-exfil, misinformation left empty — no clean source)")

# ── SAFETY BASELINE ───────────────────────────────────────────────
print("\n📁 SAFETY BASELINE")
for fname, n in [("nvidia_aegis", 2500), ("toxigen", 1500), ("google_civil_comments", 2000), ("ucb_hate_speech", 1500)]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR / f"{fname}.parquet")
        t = find_col(df, ["prompt", "text"])
        df = df[[t]].rename(columns={t: "prompt"}).sample(min(n, len(df)), random_state=42)
        add_attack(df, "Safety", "Toxic and abusive content", fname)
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
        benign_rows.append({"prompt": p, "parent_category": "Safe", "vector": "Safe", "source": f"manual_benign_{domain}"})
print(f"  ✅ {len(benign_rows)} benign prompts")

# ── SAVE ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
attack_df = pd.concat(attack_rows, ignore_index=True)
attack_df.to_csv(OUTPUT_DIR / "attack_prompts_raw.csv", index=False)
print(f"✅ {len(attack_df)} attack rows")
for cat in sorted(attack_df["parent_category"].unique()):
    print(f"     {cat}: {(attack_df['parent_category'] == cat).sum()}")
pd.DataFrame(benign_rows).to_csv(OUTPUT_DIR / "benign_prompts_raw.csv", index=False)
print(f"✅ {len(benign_rows)} benign rows")
print("\nNEXT: python scripts/03_cosine_filter.py")
