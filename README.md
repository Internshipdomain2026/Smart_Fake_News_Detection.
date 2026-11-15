# Smart Fake News Detection System

A Python NLP and Streamlit project that classifies a supplied news headline/article as **likely fake** or **likely real** using TF-IDF features and a trained machine-learning classifier.

> **Important limitation:** This is a text-classification research prototype, not a fact-checking service. It learns patterns in the supplied labelled dataset; it does not verify claims against authoritative sources. A prediction can be wrong and must not be treated as proof that a story is true or false.

## 1. Project overview and objectives

The project demonstrates an end-to-end machine-learning workflow:

1. Load and inspect a labelled news dataset.
2. Explore class distribution and article lengths.
3. Clean and normalize news text.
4. Split the data into training and held-out test sets.
5. Fit TF-IDF features using training text only.
6. Train and compare Logistic Regression and Multinomial Naive Bayes.
7. Evaluate both classifiers and select the model with the highest macro F1.
8. Save the selected classifier and fitted vectorizer with Joblib.
9. Load those saved artifacts in a Streamlit interface for interactive predictions.

## 2. Dataset: description, source and download

### Included data

`data/fake_news_dataset_200.csv` is bundled so the project can run without a separate download. It contains **200 labelled article records**: 100 labelled `fake` and 100 labelled `real`. The CSV has:
- `text`: article text used by the training pipeline
- `label`: class label (`fake` or `real`)

The included compact dataset is the dataset used for the bundled model, metrics and visualizations. The dataset is balanced by label. The project files attribute the source collection to the WELFake dataset; consult the original dataset record and its reuse terms before redistributing the records.

### Original source and obtaining the larger dataset

The project documentation identifies WELFake (Verma et al., 2021) as the source dataset. The original collection is available from:
- Dataset record: https://zenodo.org/records/4561253
- Associated paper DOI: https://doi.org/10.1109/TCSS.2021.3068519

To use a downloaded source file, obtain `WELFake_Dataset.csv` from the dataset record and place it at `data/WELFake_Dataset.csv`. The training script currently prefers the bundled 200-row CSV when it exists. To train from the downloaded source instead, temporarily move the compact CSV out of `data/` (or change the `DATA` selection in `src/train_model.py`). The loader supports common title/headline, text/content/article and label/classification column names. Numeric labels are interpreted using the WELFake convention `0 = fake`, `1 = real`; verify the source's label mapping before using a different dataset.

## 3. Project structure

```text
Smart_Fake_News_Detection/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── fake_news_dataset_200.csv
│   └── DATASET_INSTRUCTIONS.md
├── models/
│   ├── model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── README.md
├── notebooks/
│   └── fake_news_analysis.ipynb
├── outputs/
│   ├── evaluation_results/
│   │   └── metrics.json
│   └── visualizations/
│       ├── label_distribution.png
│       ├── article_length_distribution.png
│       ├── model_comparison.png
│       └── *_confusion_matrix.png
├── screenshots/
├── src/
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── MODEL_PERSISTENCE.md
├── Project_Report.pdf
├── requirements.txt
└── README.md
```

## 4. Technology stack and dependencies

- Python 3.10 or newer (recommended)
- Pandas and NumPy — tabular data handling
- Scikit-learn — TF-IDF, train/test split, classifiers and evaluation metrics
- Matplotlib — charts and confusion matrices
- Joblib — model and vectorizer persistence
- Streamlit — interactive web application
- Jupyter — notebook environment
- ReportLab — report-generation dependency

All Python dependencies are listed in `requirements.txt`.

## 5. Installation

Run these commands from the `Smart_Fake_News_Detection` project root.

### Windows (PowerShell / Command Prompt)

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Preprocessing, training and evaluation

### Preprocessing

The shared function in `src/preprocessing.py`:
- decodes HTML entities and normalizes Unicode;
- removes HTML tags, URLs and email addresses;
- converts text to lowercase;
- removes punctuation while preserving apostrophes and digits;
- normalizes repeated whitespace.

The data loader combines title and article text when both are available, removes missing/empty labels and exact duplicate article text, and normalizes supported labels to `fake` and `real`.

### Train and generate outputs

```bash
python src/train_model.py
```

This command:
- loads the selected CSV and prepares cleaned text;
- creates a reproducible balanced sample of up to 100 examples per class;
- saves the compact CSV;
- creates dataset and article-length plots;
- uses an 80/20 stratified train/test split with random seed 42;
- fits TF-IDF on training data only and transforms the held-out test data;
- trains Logistic Regression and Multinomial Naive Bayes;
- calculates accuracy, macro precision, macro recall, macro F1 and classification reports;
- saves both confusion matrices and a model-comparison chart;
- selects the classifier with the highest macro F1;
- saves the selected model and fitted vectorizer under `models/`;
- writes the actual evaluation results to `outputs/evaluation_results/metrics.json`.

### Print the saved evaluation summary

```bash
python src/evaluate_model.py
```

Do not manually edit or invent metric values. The JSON file is the source of the reported evaluation results and is regenerated when training is rerun.

## 7. Saved model and vectorizer

The project includes the following generated artifacts:

- `models/model.joblib` — selected fitted classifier.
- `models/tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer.

The training script writes both files using Joblib. The Streamlit application loads them using `joblib.load()` through a cached resource loader. It does **not** retrain the classifiers during application startup or each prediction. If the dataset, preprocessing logic or training configuration changes, rerun `python src/train_model.py` to regenerate the artifacts.

See `MODEL_PERSISTENCE.md` for further details.

## 8. Launch the Streamlit application

From the project root, after installing dependencies and ensuring the saved artifacts exist:

```bash
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) in the terminal. Open that address in a browser. Paste a headline or article and select **Analyze news**.

Input validation:
- Empty input displays an error message.
- Input shorter than 10 characters displays a warning.
- Longer input is cleaned, transformed using the saved vectorizer and classified by the saved model.

The app also displays model evaluation information and available dataset/model visualizations.

## 9. Current model evaluation results

The included metrics were regenerated from the bundled 200-row dataset using an 80/20 stratified split (160 training records and 40 test records). Results below are from the current `outputs/evaluation_results/metrics.json`.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7750 | 0.7813 | 0.7750 | 0.7737 |
| Multinomial Naive Bayes | 0.8000 | 0.8030 | 0.8000 | 0.7995 |

**Selected model:** Multinomial Naive Bayes (highest macro F1 in this run).

These values describe performance on this particular held-out split only. The dataset is small; results should not be generalized as real-world fake-news detection accuracy.

## 10. Screenshots and output evidence

The `screenshots/` folder contains evidence images covering the requested documentation points:

| Screenshot | Evidence |
|---|---|
| `04_dataset_overview_label_distribution.png` | Dataset overview and balanced label distribution |
| `05_preprocessing_eda_evidence.png` | Text-cleaning steps and an exploratory text example |
| `06_model_evaluation_metrics.png` | Accuracy, macro precision, macro recall and macro F1 |
| `07_confusion_matrices_model_comparison.png` | Both confusion matrices and model-comparison chart |
| `08_streamlit_app_interface_preview.png` | Streamlit application input/dashboard layout |
| `09_prediction_result_preview.png` | Example prediction-result interface state |
| `10_empty_input_validation_preview.png` | Empty-input validation message |
| `11_short_input_validation_preview.png` | Short-input validation warning |

The existing `01_` to `03_` preview images are retained. **Important evidence note:** the numbered UI images are static documentation previews, not screenshots captured from a running browser session. For a final live demonstration, launch the app and replace or supplement these previews with genuine screenshots from your own running Streamlit session. The dataset charts and metric/confusion-matrix evidence are generated from the included dataset and current evaluation outputs.

## 11. Testing checklist

- [ ] Install dependencies from `requirements.txt`.
- [ ] Run `python src/train_model.py` and confirm the model, vectorizer, metrics and visualizations are regenerated.
- [ ] Run `python src/evaluate_model.py` and compare its output with `metrics.json`.
- [ ] Launch Streamlit using the command above.
- [ ] Test a normal headline/article input.
- [ ] Test empty input and confirm the validation message.
- [ ] Test fewer than 10 characters and confirm the short-input warning.
- [ ] Restart the app and confirm the saved artifacts load without retraining.
- [ ] Capture genuine browser screenshots of the live app for final submission.
- [ ] Add student/institution details to the report if required by the submission format.

## 12. Deployment

The app is currently configured for local execution; no public deployment is included in this ZIP. For a hosted deployment (for example, Streamlit Community Cloud), push the project to a repository, ensure `requirements.txt`, `app/`, `src/`, `models/` and the required evaluation assets are available, then configure the app entry point as `app/streamlit_app.py`. Confirm dataset and model redistribution permissions before making a repository public. Deployment availability and platform settings may change; follow the hosting provider's current instructions.

## 13. Limitations and future improvements

- Only 200 labelled articles are included in the compact training dataset.
- Labels and source coverage may contain bias or errors.
- Random train/test splitting may allow related or same-source articles to appear across both sets.
- Language, topic, satire, manipulated context and breaking-news patterns may differ from training data.
- Model scores are not independent verification of a claim.
- Future work could use a larger licensed dataset, source-aware/time-based splitting, external validation, probability calibration, explainability and periodic retraining.

Always communicate predictions as model estimates, not factual verdicts.
