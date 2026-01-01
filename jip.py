import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import joblib
import os
from urllib.parse import urlparse
import tldextract
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="🔎 Malicious URL Detector", page_icon="🕵️‍♂️", layout="wide")


st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #1e293b;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #8b5cf6, #6366f1);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .safe {
        background-color: rgba(34,197,94,0.2);
        color: #22c55e;
        border: 2px solid #22c55e;
    }
    .malicious {
        background-color: rgba(239,68,68,0.2);
        color: #ef4444;
        border: 2px solid #ef4444;
    }
    h1 {
        color: #93c5fd;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    h3 {
        color: #c084fc;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


SUSPICIOUS_KEYWORDS = [
    "login", "signin", "bank", "update", "verify", "confirm", "secure",
    "account", "password", "free", "offer", "winner", "win", "click",
    "download", "ebayisapi", "paypal", "google", "bit.ly", "short", "verify-login"
]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [count/len(s) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def has_ip_in_domain(url: str) -> int:
    match = re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', url)
    return 1 if match else 0

def extract_features(url: str) -> dict:
    features = {}
    url = url.strip()
    features['url'] = url
    features['url_length'] = len(url)
    features['num_digits'] = len(re.findall(r'\d', url))
    features['digit_ratio'] = features['num_digits'] / features['url_length'] if features['url_length'] else 0
    features['num_special'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dashes'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_at'] = url.count('@')
    features['num_query_params'] = url.count('?')
    features['num_slashes'] = url.count('/')
    features['has_https'] = 1 if url.lower().startswith('https') else 0
    features['has_www'] = 1 if 'www.' in url.lower() else 0
    features['has_ip'] = has_ip_in_domain(url)
    features['contains_double_slash'] = 1 if '//' in urlparse(url).path else 0
    features['entropy'] = shannon_entropy(url)
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        hostname = parsed.hostname or ''
    except Exception:
        hostname = ''
    ext = tldextract.extract(hostname)
    features['tld'] = ext.suffix.lower() if ext.suffix else ''
    features['num_subdomains'] = 0 if not ext.subdomain else ext.subdomain.count('.') + 1
    low = url.lower()
    features['suspicious_keyword'] = int(any(k in low for k in SUSPICIOUS_KEYWORDS))
    features['special_ratio'] = features['num_special'] / features['url_length'] if features['url_length'] else 0
    return features

def features_to_vector(feats: dict, tld_list=None):
    vector = [
        feats['url_length'], feats['num_digits'], feats['digit_ratio'], feats['num_special'],
        feats['num_dashes'], feats['num_underscores'], feats['num_at'], feats['num_query_params'],
        feats['num_slashes'], feats['has_https'], feats['has_www'], feats['has_ip'],
        feats['contains_double_slash'], feats['entropy'], feats['num_subdomains'],
        feats['suspicious_keyword'], feats['special_ratio'],
    ]
    if tld_list is None:
        return np.array(vector)
    tld = feats.get('tld', '')
    for tt in tld_list:
        vector.append(1 if tld == tt else 0)
    vector.append(1 if (tld not in tld_list and tld != '') else 0)
    return np.array(vector)

@st.cache_data
def demo_dataframe():
    return pd.DataFrame({
        'URL': [
            'http://free-offers-win.com',
            'https://secure.google.com',
            'http://login-bankverify.net',
            'https://github.com',
            'http://get-rich-now.info',
            'http://192.168.0.1/admin',
            'http://paypal-secure-update.com/verify',
            'https://example.com/about'
        ],
        'Label': ['malicious', 'safe', 'malicious', 'safe', 'malicious', 'malicious', 'malicious', 'safe']
    })


st.markdown("<h1>🕵️‍♂️ Malicious URL Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a5b4fc;'>Detect phishing and malicious websites using machine learning 🚀</p>", unsafe_allow_html=True)

st.write("")

with st.container():
    uploaded = st.file_uploader("📂 Upload a CSV file (must contain 'URL' and 'Label' columns):", type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if 'URL' not in df.columns or 'Label' not in df.columns:
                st.error("❌ CSV must contain 'URL' and 'Label' columns.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        df = demo_dataframe()
        st.info("💡 Using demo dataset. Upload your own dataset to train a new model.")

    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

def build_training_data(df):
    feats = [extract_features(u) for u in df['URL'].astype(str)]
    feats_df = pd.DataFrame(feats)
    labels = df['Label'].astype(str).map(lambda x: 1 if x.lower() in ['malicious','bad','phishing','malware','1','true','yes'] else 0)
    top_tlds = feats_df['tld'].value_counts().nlargest(5).index.tolist()
    X = np.vstack([features_to_vector(row, tld_list=top_tlds) for _, row in feats_df.iterrows()])
    y = labels.values
    return X, y, top_tlds, feats_df

model_path = "model.pkl"
st.markdown("<h3>⚙️ Model Training</h3>", unsafe_allow_html=True)
if st.button("Train / Retrain Model"):
    with st.spinner("🔧 Training model..."):
        X, y, top_tlds, feats_df = build_training_data(df)
        if len(np.unique(y)) < 2:
            st.error("❌ Dataset must contain at least two classes: Safe and Malicious.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained successfully — Accuracy: {acc:.3f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            joblib.dump({'model': model, 'top_tlds': top_tlds}, model_path)
else:
    if os.path.exists(model_path):
        saved = joblib.load(model_path)
        model = saved['model']
        top_tlds = saved.get('top_tlds', [])
        st.success("📦 Loaded existing trained model.")
    else:
        X, y, top_tlds, feats_df = build_training_data(df)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        st.info("🚀 Trained small demo model automatically (not saved).")

st.write("---")

st.markdown("<h3>🔗 Analyze a URL</h3>", unsafe_allow_html=True)
url_input = st.text_input("Enter a URL to analyze (e.g., http://example.com/login):")

if st.button("🔍 Analyze"):
    if not url_input.strip():
        st.error("⚠️ Please enter a valid URL.")
    else:
        feats = extract_features(url_input)
        vec = features_to_vector(feats, tld_list=top_tlds)
        prob = model.predict_proba([vec])[0]
        pred = model.predict([vec])[0]
        result_class = "malicious" if pred == 1 else "safe"
        prob_mal = f"{prob[1]*100:.2f}%"
        prob_safe = f"{prob[0]*100:.2f}%"

        st.markdown(f"""
            <div class="result-box {'malicious' if result_class=='malicious' else 'safe'}">
                {'🚨 Malicious URL Detected!' if result_class=='malicious' else '✅ Safe URL Detected!'}
                <br><br>
                <b>Malicious Probability:</b> {prob_mal} &nbsp;&nbsp; | &nbsp;&nbsp; 
                <b>Safe Probability:</b> {prob_safe}
            </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬 Detailed Feature Breakdown"):
            display_df = pd.DataFrame([feats]).drop(columns=['url'])
            st.dataframe(display_df.T.rename(columns={0:'value'}), use_container_width=True)

