"""Train, compare and persist Logistic Regression and Multinomial Naive Bayes."""
from pathlib import Path
import json, sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from data_utils import load_dataset
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "WELFake_Dataset.csv"
REDUCED_DATA = ROOT / "data" / "fake_news_dataset_200.csv"
DATA = REDUCED_DATA if REDUCED_DATA.exists() else SOURCE_DATA
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
VIZ = OUT / "visualizations"
EVAL = OUT / "evaluation_results"

def main():
    for p in [MODELS, VIZ, EVAL]:
        p.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA)
    df["clean_text"] = df["combined_text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # Use a reproducible, balanced subset to keep training manageable.
    # 100 fake + 100 real = 200 total records.
    # If the downloaded dataset contains fewer examples in either class,
    # use the largest balanced subset available rather than failing.
    per_class = min(100, int(df["label"].value_counts().min()))
    if per_class < 2:
        raise ValueError("Not enough records in each class to create a train/test split.")
    df = (df.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(n=per_class, random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True))
    # Save a compact CSV so future runs can use the smaller file directly.
    REDUCED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df[["combined_text", "label"]].rename(columns={"combined_text": "text"}).to_csv(
        REDUCED_DATA, index=False
    )
    print(f"Using balanced subset: {per_class:,} fake + {per_class:,} real = {len(df):,} records")
    print(f"Saved compact dataset to: {REDUCED_DATA}")
    print("Dataset shape:", df.shape)
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("\nMissing values:\n", df.isna().sum())

    # EDA outputs
    counts = df["label"].value_counts().reindex(["fake", "real"], fill_value=0)
    ax = counts.plot(kind="bar", rot=0, title="Real vs Fake News Distribution",
                     xlabel="Label", ylabel="Number of articles")
    ax.figure.tight_layout()
    ax.figure.savefig(VIZ / "label_distribution.png", dpi=160)
    plt.close(ax.figure)
    lengths = df["clean_text"].str.split().str.len()
    ax = lengths.hist(bins=40)
    ax.set(title="Article Length Distribution", xlabel="Words per article", ylabel="Frequency")
    ax.figure.tight_layout()
    ax.figure.savefig(VIZ / "article_length_distribution.png", dpi=160)
    plt.close(ax.figure)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    # Fit vocabulary/IDF on training data only: no test-set leakage.
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                 min_df=2, max_df=0.98, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB()
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
        }
        results[name] = metrics
        fitted[name] = model
        cm = confusion_matrix(y_test, pred, labels=["fake", "real"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"])
        disp.plot(cmap="Blues", values_format="d")
        disp.ax_.set_title(f"{name} - Confusion Matrix")
        disp.figure_.tight_layout()
        safe = name.lower().replace(" ", "_")
        disp.figure_.savefig(VIZ / f"{safe}_confusion_matrix.png", dpi=160)
        plt.close(disp.figure_)
        print(f"\n{name}\n", classification_report(y_test, pred, zero_division=0))

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    joblib.dump(fitted[best_name], MODELS / "model.joblib")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")
    (EVAL / "metrics.json").write_text(json.dumps({
        "selected_model": best_name, "test_size": 0.20, "random_state": 42,
        "n_records_after_cleaning": int(len(df)),
        "train_records": int(len(X_train)), "test_records": int(len(X_test)),
        "label_mapping": {"fake": "fake", "real": "real"},
        "models": results
    }, indent=2), encoding="utf-8")
    metric_df = pd.DataFrame({n: {k: results[n][k] for k in
                         ["accuracy", "precision_macro", "recall_macro", "f1_macro"]}
                         for n in results}).T
    ax = metric_df.plot(kind="bar", ylim=(0, 1), rot=0, title="Classifier Comparison")
    ax.set_ylabel("Score")
    ax.figure.tight_layout()
    ax.figure.savefig(VIZ / "model_comparison.png", dpi=160)
    plt.close(ax.figure)
    print(f"\nSelected model: {best_name}")
    print("Saved model and vectorizer under models/")

if __name__ == "__main__":
    main()
