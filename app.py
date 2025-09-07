# app.py
import streamlit as st
from transformers import pipeline
import requests
import re
import time

# ================== Hugging Face ML Model ==================
# Make sure the model identifier is correct and public
fake_news_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

# ================== Preprocessing Function ==================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ================== ML Prediction Function ==================
def ml_predict(text):
    result = fake_news_model(text)[0]
    label = result['label']
    if "REAL" in label.upper():
        return "🟢 Real News (ML Prediction)"
    else:
        return "🔴 Fake News (ML Prediction)"

# ================== NewsAPI Function ==================
def verify_with_newsapi(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    return articles

# ================== Streamlit Page Config ==================
st.set_page_config(page_title="✨ The Truth Detector ✨", layout="wide")

# ================== CSS Styling ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #E6E6FA, #D8BFD8);
    color: #222222;
}
h1.title {
    text-align: center;
    font-family: 'Georgia', serif;
    font-size: 55px;
    color: #6a0572;
    text-shadow: 2px 2px #e0d4f7;
}
h3.subtitle {
    text-align: center;
    font-family: 'Arial', sans-serif;
    color: #4b0082;
    margin-bottom: 30px;
}
textarea {
    background-color: #ffffffcc;
    color: #222222;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
}
div.stButton > button {
    background: linear-gradient(90deg, #b19cd9, #d8bfd8);
    color: #fff;
    font-weight: bold;
    border-radius: 15px;
    padding: 15px 35px;
    transition: transform 0.2s;
    animation: pulse 1.5s infinite;
    font-size: 18px;
}
div.stButton > button:hover {
    transform: scale(1.1);
    cursor: pointer;
}
@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.05);}
    100% {transform: scale(1);}
}
.article-card {
    background-color: #ffffffcc;
    padding:20px;
    border-radius:15px;
    margin-bottom:20px;
    opacity: 0;
    transform: translateY(20px);
    animation: fadeIn 1s forwards;
}
.article-title {
    color:#6a0572;
    font-size:24px;
    font-weight:bold;
}
.article-desc {
    color:#222222;
    font-size:16px;
}
.article-link {
    color:#8a2be2;
    text-decoration:none;
    font-weight:bold;
}
@keyframes fadeIn {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1 class='title'>📰 The Truth Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Uncover the real from the fake ✨</h3>", unsafe_allow_html=True)
user_input = st.text_area("Paste the news article here:", height=180)

# ================== Your NewsAPI Key ==================
api_key = "e4d442912a3c4aa1a6cfa3d2e2e3e80b"

# ================== Detect Button ==================
if st.button("🔍 Reveal the Truth"):
    placeholder = st.empty()
    
    # Typing effect simulation
    text = "Analyzing the news magic..."
    for i in range(len(text)+1):
        placeholder.text(text[:i])
        time.sleep(0.05)
    
    # 1️⃣ Check NewsAPI first
    articles = verify_with_newsapi(user_input, api_key)
    
    if articles:
        placeholder.empty()
        st.success("🟢 Verified: Real News from trusted sources")
        for idx, art in enumerate(articles[:5]):  # Show top 5 results
            st.markdown(f"""
            <div class="article-card" style="animation-delay:{0.2*idx}s">
                <h3 class="article-title">{art['title']}</h3>
                <p class="article-desc">{art['description'] or art['content']}</p>
                <a class="article-link" href="{art['url']}" target="_blank">Read full article</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        placeholder.empty()
        # 2️⃣ ML fallback
        result = ml_predict(user_input)
        if "Real" in result:
            st.success(result)
        else:
            st.error(result)
        st.markdown("Full Article Entered:")
        st.write(user_input)
