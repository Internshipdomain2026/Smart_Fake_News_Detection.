from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
METRICS_PATH = ROOT / "outputs" / "evaluation_results" / "metrics.json"

st.set_page_config(page_title="Smart Fake News Detection", page_icon="📰", layout="wide")
st.title("📰 Smart Fake News Detection System")
st.caption("TF-IDF + Logistic Regression / Multinomial Naive Bayes")
st.warning("This is a text-classification research prototype, not a fact-checker. Predictions can be wrong and do not independently verify claims.")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

model, vectorizer = load_artifacts()
if model is None:
    st.error("Trained artifacts are missing. Follow README.md and run `python src/train_model.py` first.")
    st.stop()

st.subheader("Check a news headline or article")
news_text = st.text_area("Paste news text here", height=220,
    placeholder="Enter a headline or article to classify...")
if st.button("Analyze news", type="primary"):
    if not news_text.strip():
        st.error("Please enter a headline or article before submitting.")
    elif len(news_text.strip()) < 10:
        st.warning("Please provide a little more text for a meaningful prediction.")
    else:
        features = vectorizer.transform([clean_text(news_text)])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
        if prediction == "real":
            st.success("Model prediction: LIKELY REAL")
        else:
            st.error("Model prediction: LIKELY FAKE")
        if probabilities is not None:
            prob_df = pd.DataFrame({"Class": model.classes_, "Model score": probabilities})
            st.bar_chart(prob_df.set_index("Class"))
            st.caption("Scores are model estimates, not verified probabilities that the claim is true.")

st.divider()
st.subheader("Model evaluation dashboard")
if METRICS_PATH.exists():
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.write(f"Selected model: **{data['selected_model']}**")
    st.write(f"Test records: **{data['test_records']:,}** | Training records: **{data['train_records']:,}**")
    rows = []
    for name, m in data["models"].items():
        rows.append({"Model": name, "Accuracy": m["accuracy"],
                     "Precision (macro)": m["precision_macro"],
                     "Recall (macro)": m["recall_macro"], "F1 (macro)": m["f1_macro"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
    for filename, title in [
        ("label_distribution.png", "Dataset label distribution"),
        ("article_length_distribution.png", "Article length distribution"),
        ("model_comparison.png", "Model comparison"),
        ("logistic_regression_confusion_matrix.png", "Logistic Regression confusion matrix"),
        ("multinomial_naive_bayes_confusion_matrix.png", "Multinomial Naive Bayes confusion matrix")
    ]:
        path = ROOT / "outputs" / "visualizations" / filename
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
else:
    st.info("Evaluation metrics will appear here after model training.")
