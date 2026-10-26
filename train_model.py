"""
CardioScore Pro - dataset handling, preprocessing, training and evaluation.

CORE REQUIREMENTS COVERED HERE
  1. Dataset handling   : load the UCI Cleveland heart-disease data, check missing
                          values, clean and preprocess it.
  3. ML prediction      : train models on the labelled outcomes.
  4. Model evaluation   : accuracy, precision, recall, F1, confusion matrix, ROC-AUC.

EXTRA FEATURES COVERED HERE
  B. Five models (Logistic Regression, Random Forest, Decision Tree,
     Gradient Boosting, Extra Trees). The best one is chosen on VALIDATION
     (5-fold cross-validation) results, never on training accuracy.
  E. Explainability data (permutation importance, what-if baselines).

Run:
    python train_model.py                     # uses data/ or downloads UCI data
    python train_model.py --data path/to/file # explicit dataset file

Evaluation protocol (no data leakage):
    full data -> stratified 80/20 split
      * 80% "train" : 5-fold CV for model selection (imputer/scaler/encoder are
                      inside the Pipeline, so they are re-fitted inside every fold)
                      then each model is fitted on the whole train part
      * 20% "test"  : touched ONLY once, for the final reported metrics
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METADATA_FILE = "metadata.json"

UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/processed.cleveland.data",
]

# --------------------------------------------------------------------------
# Feature specification: the SINGLE source of truth used by training, the
# Streamlit form, input validation, PDF reports and the tests.
# Coding follows the ORIGINAL UCI Cleveland file (processed.cleveland.data).
# --------------------------------------------------------------------------
FEATURES = [
    {"name": "age", "label": "Age", "kind": "num", "dtype": "int", "unit": "years",
     "min": 20, "max": 100, "default": 54, "step": 1,
     "help": "Age of the patient in years."},
    {"name": "sex", "label": "Sex", "kind": "cat",
     "options": {0: "Female", 1: "Male"}, "default": 1,
     "help": "Biological sex as recorded in the dataset."},
    {"name": "cp", "label": "Chest pain type", "kind": "cat",
     "options": {1: "Typical angina", 2: "Atypical angina",
                 3: "Non-anginal pain", 4: "Asymptomatic"}, "default": 4,
     "help": "Type of chest pain experienced."},
    {"name": "trestbps", "label": "Resting blood pressure", "kind": "num",
     "dtype": "int", "unit": "mm Hg", "min": 80, "max": 220, "default": 130,
     "step": 1, "help": "Resting blood pressure on admission."},
    {"name": "chol", "label": "Serum cholesterol", "kind": "num", "dtype": "int",
     "unit": "mg/dl", "min": 100, "max": 600, "default": 240, "step": 1,
     "help": "Serum cholesterol in mg/dl."},
    {"name": "fbs", "label": "Fasting blood sugar", "kind": "cat",
     "options": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}, "default": 0,
     "help": "Whether fasting blood sugar is above 120 mg/dl."},
    {"name": "restecg", "label": "Resting ECG result", "kind": "cat",
     "options": {0: "Normal", 1: "ST-T wave abnormality",
                 2: "Left ventricular hypertrophy"}, "default": 0,
     "help": "Resting electrocardiographic result."},
    {"name": "thalach", "label": "Maximum heart rate achieved", "kind": "num",
     "dtype": "int", "unit": "bpm", "min": 60, "max": 220, "default": 150,
     "step": 1, "help": "Maximum heart rate reached during the exercise test."},
    {"name": "exang", "label": "Exercise-induced angina", "kind": "cat",
     "options": {0: "No", 1: "Yes"}, "default": 0,
     "help": "Whether exercise triggered chest pain."},
    {"name": "oldpeak", "label": "ST depression (exercise vs rest)", "kind": "num",
     "dtype": "float", "unit": "mm", "min": 0.0, "max": 7.0, "default": 1.0,
     "step": 0.1, "help": "ST depression induced by exercise relative to rest."},
    {"name": "slope", "label": "Slope of peak exercise ST segment", "kind": "cat",
     "options": {1: "Upsloping", 2: "Flat", 3: "Downsloping"}, "default": 2,
     "help": "Slope of the ST segment at peak exercise."},
    {"name": "ca", "label": "Major vessels coloured by fluoroscopy", "kind": "cat",
     "options": {0: "0", 1: "1", 2: "2", 3: "3"}, "default": 0,
     "help": "Number of major vessels (0-3) coloured by fluoroscopy."},
    {"name": "thal", "label": "Thalassemia (thallium stress test)", "kind": "cat",
     "options": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
     "default": 3, "help": "Thallium stress-test result."},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}
NUMERIC_FEATURES = [f["name"] for f in FEATURES if f["kind"] == "num"]
CATEGORICAL_FEATURES = [f["name"] for f in FEATURES if f["kind"] == "cat"]
RAW_COLUMNS = FEATURE_NAMES + ["target"]
TARGET_ALIASES = ("num", "target", "goal", "condition")

DISCLAIMER = (
    "CardioScore Pro is an educational machine-learning project. Its output is a "
    "statistical risk estimate, NOT a medical diagnosis, and must not replace "
    "advice from a qualified doctor. If you have symptoms or health concerns, "
    "consult a medical professional."
)
LIMITATIONS = [
    "Trained on one small, old dataset (UCI Cleveland, 303 patients, donated 1988) "
    "from a single hospital, with a majority of male patients. It may not "
    "generalise to other populations.",
    "The label is 'any heart-disease finding vs none' (original scores 1-4 are "
    "merged), so severity is not predicted.",
    "About 46% of patients in this dataset have the disease, which is far higher "
    "than in the general population. The probability is therefore relative to this "
    "dataset and is not a real-world risk figure.",
    "The LOW / MODERATE / HIGH bands are simple display thresholds chosen for "
    "readability, not clinical cut-offs.",
    "The test set is small (about 60 patients), so metrics have wide uncertainty; "
    "see the +/- values from cross-validation.",
    "Factor explanations are model-based what-if estimates. They show what the "
    "model relies on, not medical causes.",
    "Important risk factors (smoking, family history, diabetes status, BMI, "
    "medication, ethnicity) are not in the dataset.",
]


class DatasetError(Exception):
    """The dataset is missing or does not look like the UCI Cleveland data."""


class InputValidationError(ValueError):
    """Patient input is incomplete or out of range. `.errors` lists the problems."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class ModelNotTrainedError(FileNotFoundError):
    """Trained model files are missing."""


# --------------------------------------------------------------------------
# 1. Dataset handling
# --------------------------------------------------------------------------
def _find_local_dataset(data_dir: Path):
    for name in ("processed.cleveland.data", "heart.csv", "heart_disease.csv",
                 "cleveland.csv"):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return None


def _download_uci(dest: Path):
    last_error = None
    for url in UCI_URLS:
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                content = response.read()
            if len(content) < 5000 or b"," not in content[:200]:
                raise ValueError("unexpected content")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return dest
        except Exception as exc:  # network errors, 404, bad content
            last_error = exc
    raise DatasetError(
        "Could not download the UCI dataset automatically "
        f"({last_error}).\nPlease download 'processed.cleveland.data' manually from "
        "https://archive.ics.uci.edu/dataset/45/heart+disease and save it inside "
        "the data/ folder, then run `python train_model.py` again."
    )


def _read_table(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    first_token = first_line.split(",")[0].strip()
    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True
    if has_header:
        df = pd.read_csv(path, na_values=["?", "", " "])
        df.columns = [str(c).strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None, na_values=["?"])
        if df.shape[1] != len(RAW_COLUMNS):
            raise DatasetError(
                f"{path.name} has {df.shape[1]} columns; expected {len(RAW_COLUMNS)} "
                "(13 features + target) as in processed.cleveland.data.")
        df.columns = RAW_COLUMNS
    return df


def load_dataset(path=None, data_dir: Path = DATA_DIR, allow_download: bool = True):
    """Return (raw DataFrame, description of where it came from)."""
    if path:
        p = Path(path)
        if not p.exists():
            raise DatasetError(f"Dataset file not found: {p}")
        return _read_table(p), str(p)
    local = _find_local_dataset(Path(data_dir))
    if local is None:
        if not allow_download:
            raise DatasetError(
                "No dataset found in data/. Put 'processed.cleveland.data' (UCI "
                "Heart Disease, Cleveland) inside the data/ folder. See data/README.md.")
        print("No dataset in data/, trying to download the UCI Cleveland file ...")
        local = _download_uci(Path(data_dir) / "processed.cleveland.data")
        print(f"Saved to {local}")
    return _read_table(local), str(local)


def validate_and_clean(df_raw: pd.DataFrame):
    """Check and clean the raw data. Returns (clean DataFrame, report dict)."""
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for alias in TARGET_ALIASES:
        if alias in df.columns:
            if alias != "target":
                df = df.rename(columns={alias: "target"})
            break
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            "Dataset is missing required columns: " + ", ".join(missing_cols) +
            ". Expected UCI Cleveland columns: " + ", ".join(RAW_COLUMNS))
    df = df[RAW_COLUMNS]
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report = {"rows_raw": int(len(df))}

    # Rows without a label cannot be used for supervised learning.
    no_target = int(df["target"].isna().sum())
    df = df[df["target"].notna()]
    report["rows_without_label_removed"] = no_target

    # Category codes must follow the UCI coding, otherwise the app labels would lie.
    for name in CATEGORICAL_FEATURES:
        allowed = {float(k) for k in FEATURE_BY_NAME[name]["options"]}
        observed = set(df[name].dropna().unique().tolist())
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise DatasetError(
                f"Column '{name}' contains codes {unexpected} that do not match the "
                f"original UCI coding {sorted(allowed)}. This usually means a "
                "re-coded version (for example a Kaggle copy with cp 0-3 or a flipped "
                "target). Use the original 'processed.cleveland.data' from UCI, or "
                "update FEATURES in train_model.py to match your college dataset.")

    # Physiologically impossible numeric values become missing (imputed later).
    invalid = {}
    for name in NUMERIC_FEATURES:
        spec = FEATURE_BY_NAME[name]
        bad = df[name].notna() & ~df[name].between(spec["min"], spec["max"])
        if bad.any():
            invalid[name] = int(bad.sum())
            df.loc[bad, name] = np.nan
    report["invalid_values_set_missing"] = invalid

    df["target"] = (df["target"] > 0).astype(int)  # 0 = no disease, 1 = disease

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))
    report["rows_clean"] = int(len(df))
    report["missing_values"] = {c: int(v) for c, v in df[FEATURE_NAMES].isna().sum().items()
                                if v > 0}
    report["total_missing_values"] = int(sum(report["missing_values"].values()))
    counts = df["target"].value_counts()
    report["class_counts"] = {"no_disease": int(counts.get(0, 0)),
                              "disease": int(counts.get(1, 0))}
    report["disease_rate"] = float(df["target"].mean()) if len(df) else 0.0

    if len(df) < 60 or min(report["class_counts"].values()) < 20:
        raise DatasetError(
            f"Too little data after cleaning ({len(df)} rows, class counts "
            f"{report['class_counts']}). At least 60 rows and 20 per class are needed.")
    df[FEATURE_NAMES] = df[FEATURE_NAMES].astype(float)
    return df, report


# --------------------------------------------------------------------------
# 2. Preprocessing + models
# --------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def get_models(seed: int = RANDOM_STATE, fast: bool = False) -> dict:
    trees = 60 if fast else 300
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, max_depth=6, min_samples_leaf=2, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50 if fast else 150, learning_rate=0.05, max_depth=3,
            random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=trees, max_depth=6, min_samples_leaf=2, random_state=seed),
    }


def make_pipeline(model) -> Pipeline:
    # Preprocessing is INSIDE the pipeline: it is fitted on training data only
    # (and re-fitted inside each CV fold), so nothing leaks from validation/test.
    return Pipeline([("prep", build_preprocessor()), ("clf", model)])


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def split_data(df: pd.DataFrame, seed: int = RANDOM_STATE, test_size: float = TEST_SIZE):
    return train_test_split(df[FEATURE_NAMES], df["target"], test_size=test_size,
                            stratify=df["target"], random_state=seed)


# --------------------------------------------------------------------------
# 3 + 4. Training and evaluation
# --------------------------------------------------------------------------
def _f(x) -> float:
    return float(round(float(x), 6))


def _metric_dict(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": _f(accuracy_score(y_true, y_pred)),
        "precision": _f(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _f(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _f(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _f(roc_auc_score(y_true, y_prob)),
    }


def train_all(df: pd.DataFrame, report: dict | None = None, source: str = "",
              models_dir: Path = MODELS_DIR, reports_dir: Path | None = REPORTS_DIR,
              seed: int = RANDOM_STATE, test_size: float = TEST_SIZE,
              fast: bool = False, verbose: bool = True) -> dict:
    """Train, validate, evaluate and save every model. Returns the metadata dict."""
    warnings.filterwarnings("ignore")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(df, seed, test_size)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scoring = {m: m for m in METRICS}

    results, fitted = {}, {}
    for name, model in get_models(seed, fast).items():
        pipe = make_pipeline(model)

        # Validation: cross-validation on the TRAIN part only.
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_summary = {m: {"mean": _f(cv_res[f"test_{m}"].mean()),
                          "std": _f(cv_res[f"test_{m}"].std())} for m in METRICS}

        # Final fit on the whole train part, evaluation on the untouched test part.
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        perm = permutation_importance(
            pipe, X_test, y_test, scoring="roc_auc", n_repeats=5 if fast else 20,
            random_state=seed)
        importance = {FEATURE_NAMES[i]: {"mean": _f(perm.importances_mean[i]),
                                         "std": _f(perm.importances_std[i])}
                      for i in np.argsort(-perm.importances_mean)}

        results[name] = {
            "file": f"{slugify(name)}.joblib",
            "cv": cv_summary,
            "train_accuracy": _f(accuracy_score(y_train, pipe.predict(X_train))),
            "test": _metric_dict(y_test, pred, proba),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            "roc": {"fpr": [_f(v) for v in fpr], "tpr": [_f(v) for v in tpr]},
            "importance": importance,
        }
        fitted[name] = pipe
        if verbose:
            print(f"  {name:<20} CV ROC-AUC {cv_summary['roc_auc']['mean']:.3f} "
                  f"(+/- {cv_summary['roc_auc']['std']:.3f}) | "
                  f"test accuracy {results[name]['test']['accuracy']:.3f}")

    # Model selection uses VALIDATION (cross-validation) results only.
    ranking = sorted(results, key=lambda n: (results[n]["cv"]["roc_auc"]["mean"],
                                             results[n]["cv"]["f1"]["mean"]),
                     reverse=True)
    best = ranking[0]
    for rank, name in enumerate(ranking, start=1):
        results[name]["cv_rank"] = rank

    baseline = {}
    for name in FEATURE_NAMES:
        col = X_train[name].dropna()
        baseline[name] = float(col.median() if name in NUMERIC_FEATURES
                               else col.mode().iloc[0])

    meta = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "config": {"seed": seed, "test_size": test_size, "cv_folds": CV_FOLDS,
                   "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                   "fast_mode": bool(fast)},
        "dataset": {"source": source, **(report or {})},
        "features": FEATURE_NAMES,
        "baseline": baseline,
        "models": results,
        "best_model": best,
        "selection": {
            "criterion": f"Highest mean {CV_FOLDS}-fold cross-validation ROC-AUC on "
                         "the training split (mean F1 breaks ties). Training accuracy "
                         "and test-set results are not used for selection.",
            "ranking": ranking,
        },
    }

    for name, pipe in fitted.items():
        joblib.dump(pipe, models_dir / results[name]["file"])
    (models_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if reports_dir is not None:
        write_reports(meta, Path(reports_dir))
    return meta


def write_reports(meta: dict, reports_dir: Path) -> None:
    """Write a CSV table and a Markdown summary that can be pasted into the college report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in meta["selection"]["ranking"]:
        m = meta["models"][name]
        row = {"model": name, "cv_rank": m["cv_rank"], "selected": name == meta["best_model"]}
        for metric in METRICS:
            row[f"cv_{metric}_mean"] = m["cv"][metric]["mean"]
            row[f"cv_{metric}_std"] = m["cv"][metric]["std"]
        row["train_accuracy"] = m["train_accuracy"]
        for metric in METRICS:
            row[f"test_{metric}"] = m["test"][metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(reports_dir / "model_comparison.csv", index=False)

    ds, cfg, best = meta["dataset"], meta["config"], meta["best_model"]
    cm = meta["models"][best]["confusion_matrix"]
    lines = [
        "# CardioScore Pro - Training Report", "",
        f"Generated: {meta['trained_at']}  ",
        f"scikit-learn {meta['versions']['scikit_learn']}, pandas {meta['versions']['pandas']}", "",
        "## Dataset",
        f"- Source: `{ds.get('source', '')}`",
        f"- Rows: {ds.get('rows_raw', '?')} raw, {ds.get('rows_clean', '?')} after cleaning "
        f"({ds.get('duplicates_removed', 0)} duplicates removed)",
        f"- Missing values (before imputation): {ds.get('missing_values', {})}",
        f"- Invalid values set to missing: {ds.get('invalid_values_set_missing', {})}",
        f"- Class counts: {ds.get('class_counts', {})}", "",
        "## Method",
        f"- Stratified split: {cfg['n_train']} train / {cfg['n_test']} test (seed {cfg['seed']})",
        f"- {cfg['cv_folds']}-fold stratified cross-validation on the train part for model selection",
        "- Imputation, scaling and one-hot encoding live inside the Pipeline (no leakage)",
        f"- Selection rule: {meta['selection']['criterion']}", "",
        "## Results", "",
        "| Model | CV ROC-AUC | CV F1 | Train acc. | Test acc. | Test precision | Test recall | Test F1 | Test ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in meta["selection"]["ranking"]:
        m = meta["models"][name]
        star = " (selected)" if name == best else ""
        lines.append(
            f"| {name}{star} | {m['cv']['roc_auc']['mean']:.3f} +/- {m['cv']['roc_auc']['std']:.3f} "
            f"| {m['cv']['f1']['mean']:.3f} | {m['train_accuracy']:.3f} "
            f"| {m['test']['accuracy']:.3f} | {m['test']['precision']:.3f} "
            f"| {m['test']['recall']:.3f} | {m['test']['f1']:.3f} | {m['test']['roc_auc']:.3f} |")
    lines += [
        "", f"## Selected model: {best}", "",
        "Confusion matrix on the test set (rows = actual, columns = predicted):", "",
        "| | Predicted no disease | Predicted disease |", "|---|---|---|",
        f"| Actual no disease | {cm[0][0]} (TN) | {cm[0][1]} (FP) |",
        f"| Actual disease | {cm[1][0]} (FN) | {cm[1][1]} (TP) |", "",
        "Top factors by permutation importance (ROC-AUC drop on the test set):", "",
    ]
    for feat, vals in list(meta["models"][best]["importance"].items())[:5]:
        lines.append(f"- {FEATURE_BY_NAME[feat]['label']}: {vals['mean']:.3f} +/- {vals['std']:.3f}")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    lines += ["", "## Disclaimer", "", DISCLAIMER, ""]
    (reports_dir / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Loading saved artifacts
# --------------------------------------------------------------------------
def artifacts_exist(models_dir: Path = MODELS_DIR) -> bool:
    models_dir = Path(models_dir)
    meta_path = models_dir / METADATA_FILE
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return all((models_dir / m["file"]).exists() for m in meta["models"].values())
    except (ValueError, KeyError, OSError):
        return False


def load_artifacts(models_dir: Path = MODELS_DIR):
    """Return (metadata dict, {model name: fitted pipeline})."""
    models_dir = Path(models_dir)
    if not artifacts_exist(models_dir):
        raise ModelNotTrainedError(
            "Trained models not found. Train them first with:  python train_model.py")
    meta = json.loads((models_dir / METADATA_FILE).read_text(encoding="utf-8"))
    models = {name: joblib.load(models_dir / info["file"])
              for name, info in meta["models"].items()}
    return meta, models


# --------------------------------------------------------------------------
# 5. Prediction helpers (used by the Streamlit app)
# --------------------------------------------------------------------------
def validate_patient_input(inputs: dict) -> dict:
    """Check a patient's values against the feature specification.

    Returns a clean {feature: float} dict or raises InputValidationError."""
    if not isinstance(inputs, dict):
        raise InputValidationError(["Input must be a dictionary of feature values."])
    errors, clean = [], {}
    for spec in FEATURES:
        name, label = spec["name"], spec["label"]
        raw = inputs.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors.append(f"{label}: a value is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label}: '{raw}' is not a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{label}: value must be a finite number.")
            continue
        if spec["kind"] == "num":
            if not spec["min"] <= value <= spec["max"]:
                errors.append(f"{label}: {value:g} is outside the allowed range "
                              f"{spec['min']} to {spec['max']} {spec['unit']}.")
                continue
        elif value not in [float(k) for k in spec["options"]]:
            errors.append(f"{label}: {value:g} is not a valid option "
                          f"({', '.join(str(k) for k in spec['options'])}).")
            continue
        clean[name] = value
    if errors:
        raise InputValidationError(errors)
    return clean


def to_frame(clean: dict) -> pd.DataFrame:
    return pd.DataFrame([clean], columns=FEATURE_NAMES).astype(float)


def display_value(name: str, value) -> str:
    spec = FEATURE_BY_NAME[name]
    if spec["kind"] == "cat":
        return str(spec["options"].get(int(round(float(value))), value))
    unit = f" {spec['unit']}" if spec.get("unit") else ""
    return f"{float(value):g}{unit}"


def describe_inputs(inputs: dict) -> list:
    """[(label, readable value), ...] in form order."""
    return [(spec["label"], display_value(spec["name"], inputs[spec["name"]]))
            for spec in FEATURES if spec["name"] in inputs]


def risk_level(probability: float) -> str:
    if probability < 0.33:
        return "LOW"
    if probability < 0.66:
        return "MODERATE"
    return "HIGH"


def predict_risk(pipe, inputs: dict) -> dict:
    clean = validate_patient_input(inputs)
    p = float(pipe.predict_proba(to_frame(clean))[0, 1])
    return {"probability": p, "risk_level": risk_level(p), "prediction": int(p >= 0.5)}


def predict_all_models(models: dict, inputs: dict) -> dict:
    clean = validate_patient_input(inputs)
    frame = to_frame(clean)
    return {name: float(pipe.predict_proba(frame)[0, 1]) for name, pipe in models.items()}


def explain_prediction(pipe, meta: dict, inputs: dict) -> pd.DataFrame:
    """What-if influence of each input.

    For each feature the value is replaced by the typical training value (median or
    most common category) and the change in predicted probability is measured.
    `influence` is in percentage points: positive = this value pushes the risk UP
    compared with a typical patient, negative = pulls it DOWN."""
    clean = validate_patient_input(inputs)
    baseline = meta["baseline"]
    original = float(pipe.predict_proba(to_frame(clean))[0, 1])
    variants = []
    for name in FEATURE_NAMES:
        variant = dict(clean)
        variant[name] = float(baseline[name])
        variants.append(variant)
    probs = pipe.predict_proba(pd.DataFrame(variants, columns=FEATURE_NAMES))[:, 1]
    rows = [{
        "feature": name,
        "label": FEATURE_BY_NAME[name]["label"],
        "value": display_value(name, clean[name]),
        "typical": display_value(name, baseline[name]),
        "influence": float(round((original - probs[i]) * 100, 2)),
    } for i, name in enumerate(FEATURE_NAMES)]
    out = pd.DataFrame(rows)
    return out.reindex(out["influence"].abs().sort_values(ascending=False).index) \
              .reset_index(drop=True)


# --------------------------------------------------------------------------
def _print_summary(meta: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'Model':<20}{'CV AUC':>9}{'CV F1':>8}{'Train acc':>11}{'Test acc':>10}"
          f"{'Prec':>7}{'Recall':>8}{'F1':>7}{'AUC':>7}")
    print("-" * 78)
    for name in meta["selection"]["ranking"]:
        m = meta["models"][name]
        mark = " *" if name == meta["best_model"] else ""
        print(f"{name:<20}{m['cv']['roc_auc']['mean']:>9.3f}{m['cv']['f1']['mean']:>8.3f}"
              f"{m['train_accuracy']:>11.3f}{m['test']['accuracy']:>10.3f}"
              f"{m['test']['precision']:>7.3f}{m['test']['recall']:>8.3f}"
              f"{m['test']['f1']:>7.3f}{m['test']['roc_auc']:>7.3f}{mark}")
    print("=" * 78)
    print(f"* Selected model (best validation ROC-AUC): {meta['best_model']}")
    print(f"Models saved in: {MODELS_DIR}\nReports saved in: {REPORTS_DIR}")
    print("Next step: streamlit run app.py")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train CardioScore Pro models.")
    parser.add_argument("--data", help="path to the dataset file (UCI Cleveland format)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--no-download", action="store_true",
                        help="never try to download the dataset")
    parser.add_argument("--fast", action="store_true", help="smaller models (quick check)")
    args = parser.parse_args(argv)

    try:
        print("[1/3] Loading dataset ...")
        raw, source = load_dataset(args.data, allow_download=not args.no_download)
        print(f"      source: {source}")
        print("[2/3] Validating and cleaning ...")
        df, report = validate_and_clean(raw)
        print(f"      rows: {report['rows_raw']} raw -> {report['rows_clean']} clean | "
              f"duplicates removed: {report['duplicates_removed']}")
        print(f"      missing values: {report['missing_values'] or 'none'}")
        print(f"      class counts: {report['class_counts']}")
    except DatasetError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("[3/3] Training and evaluating models ...")
    meta = train_all(df, report=report, source=source, seed=args.seed,
                     test_size=args.test_size, fast=args.fast)
    _print_summary(meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
