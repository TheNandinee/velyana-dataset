import csv
import pandas as pd
from pathlib import Path

DOWNLOAD_DIR = Path("downloaded")
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)
attack_rows, benign_rows, used_prompts = [], [], set()
JUNK = r"analyze the security implications|what is this exploit"


def find_col(df, kw):
    for c in df.columns:
        if c.lower() in kw or any(k in c.lower() for k in kw): return c
    return df.columns[0]

def find_label_col(df, kw):
    for c in df.columns:
        if c.lower() in kw or any(k in c.lower() for k in kw): return c
    return None

def filter_attacks(df, lab):
    if lab is None: return df
    u = set(df[lab].dropna().unique())
    if u.issubset({0,1,"0","1",True,False}): return df[df[lab].isin([1,"1",True])]
    if df[lab].dtype == "object":
        kws = ["attack","injection","jailbreak","leakage","toxic","malicious","true"]
        return df[df[lab].astype(str).str.lower().apply(lambda x: any(k in x for k in kws))]
    return df

def clean(series):
    return (series.astype(str)
            .str.replace(r"<br\s*/?>", " ", regex=True)
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"[\x00-\x1f\x7f]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())

def add_attack(df, parent, vector, source):
    if df is None or len(df) == 0: return
    df = df.copy()
    if "prompt" not in df.columns: df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna()
    df["prompt"] = clean(df["prompt"])
    df = df[~df["prompt"].str.contains(JUNK, case=False, regex=True)]
    df = df[df["prompt"].str.len().between(6, 2000)]
    df = df[~df["prompt"].isin(used_prompts)].drop_duplicates(subset=["prompt"])
    if len(df) == 0: return
    used_prompts.update(df["prompt"].tolist())
    df["parent_category"], df["vector"], df["source"] = parent, vector, source
    attack_rows.append(df[["prompt","parent_category","vector","source"]]); print(f"  OK {source} -> {vector}: {len(df)}")

def route_and_add(df, parent, rules, default, source, max_rows=None):
    if df is None or len(df) == 0: return
    df = df.copy()
    if "prompt" not in df.columns: df = df.rename(columns={df.columns[0]: "prompt"})
    df = df[["prompt"]].dropna(); df["prompt"] = df["prompt"].astype(str)
    df = df[df["prompt"].str.len().between(6, 2000)]
    if max_rows and len(df) > max_rows: df = df.sample(max_rows, random_state=42)
    def route(p):
        pl = p.lower()
        for vec, kws in rules:
            if any(k in pl for k in kws): return vec
        return default
    df["v"] = df["prompt"].apply(route); df = df[df["v"].notna()]
    for vec, grp in df.groupby("v"):
        add_attack(grp[["prompt"]], parent, vec, source)

# ---------- routing rules (vector <- keyword) ----------
PI_RULES = [
    ("Role and persona assignment", ["act as","you are now","pretend","roleplay","role-play","persona","dan mode","you are dan","from now on you are","behave as"]),
    ("Encoded and obfuscated injection", ["base64","rot13","in hex","\\x","encoded","obfuscat","in binary","ascii code","leetspeak","zero-width"]),
    ("Second-hop and forwarded injection", ["forward","in your reply","when you respond","include in your","your next response","send this to","downstream","next agent"]),
    ("Direct instruction override", ["ignore","disregard","forget","override","do not follow","bypass","new instruction","instead of"]),
]
PI_DEFAULT = "Direct instruction override"
SPE_RULES = [
    ("Repetition and verbatim requests", ["repeat","verbatim","word for word","exactly as","echo","print the above","output the above","reproduce"]),
    ("Format conversion extraction", ["translate","in json","as a poem","encode your","in the form of","convert your","spell out","pig latin"]),
    ("Summarisation inference", ["summar","tldr","tl;dr","gist of","paraphrase your"]),
    ("Completion-based extraction", ["complete the","continue the","finish the","fill in","the rest of"]),
    ("Behavioural inference and side-channel probing", ["why did you","what would you","how do you decide","based on your","what rules"]),
    ("Direct questioning", ["system prompt","your instruction","initial prompt","what were you told","your guideline","your directive","reveal your","repeat your prompt"]),
]
SPE_DEFAULT = None   # keep only rows with a real extraction cue
MC_RULES = [
    ("Cross-site scripting (XSS) generation", ["<script","onerror","onload=","javascript:","alert(","<img","<svg","document.cookie","onmouseover=","innerhtml"]),
    ("SQL injection payload generation", ["union select","union all select","' or '","or 1=1","' or 1","drop table","insert into","select * from","sqli","sleep(","benchmark(","waitfor delay","information_schema","chr(","concat(","cast("]),
    ("Command injection payload generation", ["; ls","; cat","; rm ","| cat","|cat","/bin/bash","/bin/sh","/etc/passwd","wget http","curl http","nc -","ncat ","bash -c","sh -c","whoami","$(id",";id","&& id","|id"]),
    ("Encoded and obfuscated payload delivery", ["base64","b64decode","%3c","%3e","%27","%22","\\x","\\u00","fromcharcode","eval(atob","atob(","unescape","decodeuri","obfuscat","rot13","&#x"]),
    ("Jailbreak via code context", ["write code","import os","python script","write a program","code that ","write a script","def ","function that"]),
]
MC_DEFAULT = None

print("=" * 80); print("BUILDING VELYANA DATASET (real data, mapped to taxonomy)"); print("=" * 80)

# ===== CATEGORY 1: PROMPT INJECTION =====
print("\n[1] PROMPT INJECTION")
for fn in ["lakera_gandalf", "hackaprompt", "trustairlab_jailbreak", "jayavibhav_pi"]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR/f"{fn}.parquet")
        t = find_col(df, ["prompt","text","input","attack","user_input","jailbreak"])
        route_and_add(df[[t]].rename(columns={t:"prompt"}), "Prompt Injection", PI_RULES, PI_DEFAULT, fn, max_rows=10000)
    except Exception as e: print(f"  FAIL {fn}: {e}")
try:  # microsoft_llmail = indirect injection BY CONSTRUCTION (instructions hidden in email content)
    df = pd.read_parquet(DOWNLOAD_DIR/"microsoft_llmail.parquet")
    t = find_col(df, ["body","attack","payload","text"])
    df = df[[t]].rename(columns={t:"prompt"}).sample(min(20000, len(df)), random_state=42)
    add_attack(df, "Prompt Injection", "Indirect injection via context", "microsoft_llmail")
except Exception as e: print(f"  FAIL microsoft_llmail: {e}")
try:
    df = pd.read_parquet(DOWNLOAD_DIR/"deepset_pi.parquet")
    t = find_col(df,["text","prompt"]); lab = find_label_col(df,["label"])
    route_and_add(filter_attacks(df.rename(columns={t:"prompt"}), lab)[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "deepset_pi")
except Exception as e: print(f"  FAIL deepset_pi: {e}")
try:
    df = pd.read_parquet(DOWNLOAD_DIR/"neuralchemy_pi.parquet")
    if "label" in df.columns: df = df[df["label"].isin([1,"1",True])]
    t = find_col(df,["text","prompt"]); df = df.rename(columns={t:"prompt"})
    if "category" in df.columns:
        for cval, vec in {"direct_injection":"Direct instruction override","jailbreak":"Role and persona assignment"}.items():
            add_attack(df[df["category"].astype(str).str.contains(cval, case=False, na=False)][["prompt"]], "Prompt Injection", vec, f"neuralchemy_{cval}")
        rest = df[~df["category"].astype(str).str.lower().isin(["direct_injection","jailbreak","benign"])]
        route_and_add(rest[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "neuralchemy_other")
    else:
        route_and_add(df[["prompt"]], "Prompt Injection", PI_RULES, PI_DEFAULT, "neuralchemy")
except Exception as e: print(f"  FAIL neuralchemy: {e}")

# ===== CATEGORY 2: SYSTEM PROMPT EXTRACTION (gabrielchua; weak mapping, see COVERAGE.md) =====
print("\n[2] SYSTEM PROMPT EXTRACTION")
try:
    df = pd.read_parquet(DOWNLOAD_DIR/"gabrielchua_spe.parquet")
    t = find_col(df,["content","prompt","text"]); lab = find_label_col(df,["leakage","label"])
    route_and_add(filter_attacks(df.rename(columns={t:"prompt"}), lab)[["prompt"]], "System Prompt Extraction", SPE_RULES, SPE_DEFAULT, "gabrielchua_spe", max_rows=40000)
except Exception as e: print(f"  FAIL gabrielchua_spe: {e}")

# ===== CATEGORY 3: MALICIOUS CODE AND PAYLOAD INJECTION =====
print("\n[3] MALICIOUS CODE AND PAYLOAD INJECTION")
for fn, cols, mx in [("truongp_web_attack",["sentence","text","prompt"],30000), ("waiper_exploitdb",["input","prompt","payload","text"],6000)]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR/f"{fn}.parquet")
        t = find_col(df, cols); lab = find_label_col(df,["label"])
        df = filter_attacks(df.rename(columns={t:"prompt"}), lab) if lab else df.rename(columns={t:"prompt"})
        route_and_add(df[["prompt"]], "Malicious Code and Payload Injection", MC_RULES, MC_DEFAULT, fn, max_rows=mx)
    except Exception as e: print(f"  FAIL {fn}: {e}")
try:  # RMCBench = prompts that instruct an LLM to generate malicious code -> Jailbreak via code context
    df = pd.read_parquet(DOWNLOAD_DIR/"rmcbench.parquet")
    t = find_col(df, ["prompt","text","instruction"])
    add_attack(df[[t]].rename(columns={t:"prompt"}), "Malicious Code and Payload Injection", "Jailbreak via code context", "rmcbench")
except Exception as e: print(f"  FAIL rmcbench: {e}")

# ===== CATEGORY 5: MALICIOUS CONTENT IN OUTPUT =====
print("\n[5] MALICIOUS CONTENT IN OUTPUT")
try:
    df = pd.read_parquet(DOWNLOAD_DIR/"phishing_urls.parquet")
    lab = find_label_col(df,["label"]); t = find_col(df,["text","url"]); df = df.rename(columns={t:"prompt"})
    if lab: df = df[df[lab].isin([1,"1",True])]
    add_attack(df[["prompt"]], "Malicious Content in Output", "Malicious URL generation", "phishing_urls")
except Exception as e: print(f"  FAIL phishing_urls: {e}")
try:
    df = pd.read_parquet(DOWNLOAD_DIR/"fake_news.parquet")
    lab = find_label_col(df,["label"]); t = "title" if "title" in df.columns else find_col(df,["text","content"])
    df = df.rename(columns={t:"prompt"})
    if lab is not None: df = df[df[lab].isin([0,"0",False])]
    add_attack(df[["prompt"]].sample(min(6000,len(df)), random_state=42), "Malicious Content in Output", "Misinformation and hallucination manipulation", "fake_news")
except Exception as e: print(f"  FAIL fake_news: {e}")

# ===== SAFETY BASELINE (optional; not in the 31-vector taxonomy) =====
print("\n[*] SAFETY BASELINE (optional)")
for fn, n in [("nvidia_aegis",3000),("toxigen",1500),("google_civil_comments",3000),("ucb_hate_speech",2000)]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR/f"{fn}.parquet")
        t = find_col(df,["prompt","text"])
        add_attack(df[[t]].rename(columns={t:"prompt"}).sample(min(n,len(df)), random_state=42), "Safety", "Toxic and abusive content", fn)
    except Exception as e: print(f"  FAIL {fn}: {e}")

# ===== BENIGN =====
print("\n[B] BENIGN")
BENIGN = {
    "Healthcare": ["What are the symptoms of type 2 diabetes?","How often should I get a cholesterol check?","Can you explain what an MRI scan is used for?","What are the side effects of ibuprofen?","How do I know if I have high blood pressure?","What vaccinations are recommended for adults?","How does chemotherapy work?","Is it safe to take paracetamol and ibuprofen together?","What are the early signs of a stroke?","How can I manage chronic back pain?"],
    "Banking": ["How do I open a savings account?","What is the difference between debit and credit cards?","How does compound interest work?","What documents do I need for a home loan?","How is a credit score calculated?","How long does a wire transfer take?","How do I dispute a transaction?","What is a fixed deposit?","How do I apply for a credit card?","How do I update my KYC details?"],
    "Education": ["What is the difference between Bachelor's and Master's degrees?","How do I write a college application essay?","What is the Socratic method?","What are the best resources for learning Python online?","How do I reference a website in APA format?","What is a literature review?","How do scholarships differ from loans?","How do I create an effective study schedule?","What is plagiarism?","What programming languages are best for AI?"],
    "E-commerce": ["How do I track my order?","What is the return policy for sale items?","How do I apply a discount code?","Why is my payment failing?","How long does standard shipping take?","How do I cancel an order?","How do I return a defective product?","How do I leave a review?","How does cashback work?","How do I get a refund for a cancelled order?"],
    "Travel": ["What documents do I need for a UK visa?","What is the best time to visit Rajasthan?","How do I find cheap flights?","Do I need travel insurance?","What is the baggage allowance?","What are top attractions in Kyoto?","How do I exchange currency?","How do I report lost luggage?","What is a layover?","How do I plan a 10-day Europe trip on budget?"],
}
for domain, prompts in BENIGN.items():
    for p in prompts:
        benign_rows.append({"prompt": p, "parent_category": "Safe", "vector": "Safe", "source": f"manual_benign_{domain}"})
# large benign from clean instruction datasets
for fn, n in [("dolly", 15000), ("alpaca", 12000)]:
    try:
        df = pd.read_parquet(DOWNLOAD_DIR/f"{fn}.parquet")
        t = find_col(df, ["instruction","prompt","text"])
        s = df[[t]].rename(columns={t:"prompt"}).dropna()
        s["prompt"] = clean(s["prompt"])
        s = s[s["prompt"].str.len().between(6,2000)].sample(min(n,len(s)), random_state=42)
        for p in s["prompt"]:
            benign_rows.append({"prompt": p, "parent_category": "Safe", "vector": "Safe", "source": fn})
        print(f"  OK {fn}: {len(s)}")
    except Exception as e: print(f"  FAIL {fn}: {e}")
print(f"  benign total: {len(benign_rows)}")

pd.concat(attack_rows, ignore_index=True).to_csv(OUTPUT_DIR/"attack_prompts_raw.csv", index=False, quoting=csv.QUOTE_ALL)
pd.DataFrame(benign_rows).to_csv(OUTPUT_DIR/"benign_prompts_raw.csv", index=False, quoting=csv.QUOTE_ALL)
print("\nwrote raw CSVs -> next: 03")
