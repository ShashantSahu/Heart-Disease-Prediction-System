"""
CardioScore Pro - automated tests.

Run with either:
    python -m unittest discover -s tests -v
    pytest tests

The tests build a SMALL SYNTHETIC dataset in the original UCI coding inside a
temporary folder. It is only a test fixture; real results come from the real
UCI Cleveland dataset used by `python train_model.py`.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auth  # noqa: E402
import reporting  # noqa: E402
import train_model as tm  # noqa: E402


def make_fixture_dataset(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": rng.integers(29, 78, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.choice([0, 1, 2], n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": np.round(rng.gamma(1.2, 0.9, n), 1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float)})
    z = (0.05 * (d.age - 54) + 0.8 * d.sex + 0.6 * (d.cp == 4) + 0.8 * d.exang
         + 0.6 * d.oldpeak + 0.7 * d.ca + 0.8 * (d.thal == 7)
         - 0.03 * (d.thalach - 150) + rng.normal(0, 1, n))
    d["num"] = np.where(z > 1.6, rng.integers(1, 5, n), 0)
    d.loc[[3, 40], "ca"] = np.nan
    d.loc[[8], "thal"] = np.nan
    return d


def default_patient() -> dict:
    return {f["name"]: f["default"] for f in tm.FEATURES}


class TestInputValidation(unittest.TestCase):
    def test_default_form_values_are_valid(self):
        clean = tm.validate_patient_input(default_patient())
        self.assertEqual(set(clean), set(tm.FEATURE_NAMES))

    def test_missing_field_is_reported(self):
        patient = default_patient()
        del patient["chol"]
        with self.assertRaises(tm.InputValidationError) as ctx:
            tm.validate_patient_input(patient)
        self.assertTrue(any("Serum cholesterol" in e for e in ctx.exception.errors))

    def test_out_of_range_numeric_rejected(self):
        for name, bad in [("age", 5), ("age", 150), ("trestbps", 400), ("oldpeak", -1)]:
            patient = default_patient()
            patient[name] = bad
            with self.assertRaises(tm.InputValidationError, msg=f"{name}={bad}"):
                tm.validate_patient_input(patient)

    def test_invalid_category_rejected(self):
        for name, bad in [("cp", 0), ("cp", 9), ("thal", 1), ("sex", 2), ("slope", 0)]:
            patient = default_patient()
            patient[name] = bad
            with self.assertRaises(tm.InputValidationError, msg=f"{name}={bad}"):
                tm.validate_patient_input(patient)

    def test_non_numeric_and_nan_rejected(self):
        for bad in ("abc", float("nan"), float("inf"), None, ""):
            patient = default_patient()
            patient["age"] = bad
            with self.assertRaises(tm.InputValidationError, msg=repr(bad)):
                tm.validate_patient_input(patient)

    def test_all_errors_listed_together(self):
        patient = default_patient()
        patient.update(age=1, cp=0)
        with self.assertRaises(tm.InputValidationError) as ctx:
            tm.validate_patient_input(patient)
        self.assertEqual(len(ctx.exception.errors), 2)


class TestDatasetCleaning(unittest.TestCase):
    def test_cleaning_report(self):
        raw = make_fixture_dataset()
        raw = pd.concat([raw, raw.iloc[:5]], ignore_index=True)  # 5 duplicates
        raw.loc[10, "age"] = 500  # impossible value
        clean, report = tm.validate_and_clean(raw)
        self.assertEqual(report["duplicates_removed"], 5)
        self.assertEqual(report["invalid_values_set_missing"], {"age": 1})
        self.assertIn("ca", report["missing_values"])
        self.assertTrue(set(clean["target"].unique()) <= {0, 1})
        self.assertEqual(report["rows_clean"], len(clean))

    def test_recoded_dataset_is_rejected_with_clear_message(self):
        raw = make_fixture_dataset()
        raw["cp"] = raw["cp"] - 1  # Kaggle-style 0-3 coding
        with self.assertRaises(tm.DatasetError) as ctx:
            tm.validate_and_clean(raw)
        self.assertIn("cp", str(ctx.exception))

    def test_missing_columns_rejected(self):
        with self.assertRaises(tm.DatasetError):
            tm.validate_and_clean(make_fixture_dataset().drop(columns=["chol"]))

    def test_file_loading_headerless_with_question_marks(self):
        raw = make_fixture_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed.cleveland.data"
            raw.to_csv(path, header=False, index=False, na_rep="?")
            loaded, source = tm.load_dataset(path)
        self.assertEqual(list(loaded.columns), tm.RAW_COLUMNS)
        self.assertEqual(int(loaded["ca"].isna().sum()), 2)

    def test_missing_dataset_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(tm.DatasetError) as ctx:
                tm.load_dataset(None, data_dir=Path(tmp), allow_download=False)
        self.assertIn("data/", str(ctx.exception))


class TestTrainingAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.models_dir = Path(cls._tmp.name) / "models"
        cls.df, cls.report = tm.validate_and_clean(make_fixture_dataset())
        cls.meta = tm.train_all(cls.df, report=cls.report, source="fixture",
                                models_dir=cls.models_dir, reports_dir=None,
                                fast=True, verbose=False)
        cls.meta_loaded, cls.models = tm.load_artifacts(cls.models_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_five_models_and_files_saved(self):
        expected = {"Logistic Regression", "Random Forest", "Decision Tree",
                    "Gradient Boosting", "Extra Trees"}
        self.assertEqual(set(self.meta["models"]), expected)
        self.assertEqual(set(self.models), expected)
        self.assertTrue((self.models_dir / "metadata.json").exists())

    def test_metrics_are_real_and_in_range(self):
        for name, info in self.meta["models"].items():
            for metric in tm.METRICS:
                self.assertTrue(0.0 <= info["test"][metric] <= 1.0, (name, metric))
                self.assertTrue(0.0 <= info["cv"][metric]["mean"] <= 1.0, (name, metric))
            (tn, fp), (fn, tp) = info["confusion_matrix"]
            self.assertEqual(tn + fp + fn + tp, self.meta["config"]["n_test"])

    def test_test_metrics_match_a_fresh_evaluation(self):
        """Metrics in metadata must equal what the saved model really scores."""
        _, X_test, _, y_test = tm.split_data(self.df)
        name = self.meta["best_model"]
        proba = self.models[name].predict_proba(X_test)[:, 1]
        fresh = tm._metric_dict(y_test, (proba >= 0.5).astype(int), proba)
        for metric in tm.METRICS:
            self.assertAlmostEqual(fresh[metric], self.meta["models"][name]["test"][metric], 5)

    def test_best_model_chosen_by_validation_not_training_or_test(self):
        cv_auc = {n: i["cv"]["roc_auc"]["mean"] for n, i in self.meta["models"].items()}
        top_auc = max(cv_auc.values())
        self.assertAlmostEqual(cv_auc[self.meta["best_model"]], top_auc)
        self.assertIn("cross-validation", self.meta["selection"]["criterion"])
        self.assertEqual(self.meta["selection"]["ranking"][0], self.meta["best_model"])

    def test_no_leakage_split_and_preprocessing_fit_on_train_only(self):
        X_train, X_test, _, _ = tm.split_data(self.df)
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertTrue(set(X_train.index).isdisjoint(set(X_test.index)))
        self.assertEqual(self.meta["config"]["n_train"], len(X_train))
        pipe = self.models["Logistic Regression"]
        imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["imputer"]
        expected = X_train[tm.NUMERIC_FEATURES].median().to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected)

    def test_form_fields_match_model_features(self):
        form_fields = [f["name"] for f in tm.FEATURES]
        for name, pipe in self.models.items():
            self.assertEqual(list(pipe.named_steps["prep"].feature_names_in_), form_fields, name)
        self.assertEqual(self.meta["features"], form_fields)

    def test_prediction_returns_valid_probability(self):
        for name, pipe in self.models.items():
            result = tm.predict_risk(pipe, default_patient())
            self.assertTrue(0.0 <= result["probability"] <= 1.0, name)
            self.assertIn(result["risk_level"], {"LOW", "MODERATE", "HIGH"})

    def test_prediction_rejects_invalid_input(self):
        patient = default_patient()
        patient["age"] = 999
        with self.assertRaises(tm.InputValidationError):
            tm.predict_risk(self.models["Random Forest"], patient)

    def test_all_models_prediction(self):
        probs = tm.predict_all_models(self.models, default_patient())
        self.assertEqual(set(probs), set(self.models))

    def test_explanation_covers_every_feature(self):
        expl = tm.explain_prediction(self.models["Random Forest"], self.meta, default_patient())
        self.assertEqual(set(expl["feature"]), set(tm.FEATURE_NAMES))
        self.assertTrue({"label", "value", "typical", "influence"} <= set(expl.columns))
        self.assertTrue((expl["influence"].abs().diff().dropna() <= 1e-9).all())  # sorted

    def test_risk_bands(self):
        self.assertEqual(tm.risk_level(0.10), "LOW")
        self.assertEqual(tm.risk_level(0.50), "MODERATE")
        self.assertEqual(tm.risk_level(0.90), "HIGH")

    def test_missing_model_files_detected(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertFalse(tm.artifacts_exist(Path(empty)))
            with self.assertRaises(tm.ModelNotTrainedError) as ctx:
                tm.load_artifacts(Path(empty))
            self.assertIn("python train_model.py", str(ctx.exception))

    def test_training_report_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tm.write_reports(self.meta, Path(tmp))
            self.assertTrue((Path(tmp) / "model_comparison.csv").exists())
            text = (Path(tmp) / "training_report.md").read_text(encoding="utf-8")
            self.assertIn("Selected model", text)


class TestAuthAndPrivacy(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = (auth.DB_PATH, auth.PBKDF2_ITERATIONS)
        auth.DB_PATH = Path(self._tmp.name) / "test.db"
        auth.PBKDF2_ITERATIONS = 1000  # fast for tests only
        auth.init_db()

    def tearDown(self):
        auth.DB_PATH, auth.PBKDF2_ITERATIONS = self._old
        self._tmp.cleanup()

    def test_signup_and_login(self):
        user = auth.create_user("Alice_1", "Secret123", "Alice A")
        self.assertEqual(user["username"], "alice_1")
        self.assertEqual(auth.authenticate("ALICE_1", "Secret123")["id"], user["id"])
        self.assertIsNone(auth.authenticate("alice_1", "wrong-pass1"))
        self.assertIsNone(auth.authenticate("nobody", "Secret123"))

    def test_password_is_hashed_not_stored(self):
        auth.create_user("bob", "Secret123")
        import sqlite3
        stored = sqlite3.connect(str(auth.DB_PATH)).execute(
            "SELECT password_hash FROM users").fetchone()[0]
        self.assertNotIn("Secret123", stored)
        self.assertTrue(stored.startswith("pbkdf2_sha256$"))
        self.assertNotEqual(auth.hash_password("x1234567"), auth.hash_password("x1234567"))

    def test_duplicate_and_weak_accounts_rejected(self):
        auth.create_user("carol", "Secret123")
        with self.assertRaises(auth.AuthError):
            auth.create_user("CAROL", "Another123")
        for username, password in [("ab", "Secret123"), ("dave", "short1"),
                                   ("dave", "onlyletters"), ("dave", "12345678"),
                                   ("bad name!", "Secret123")]:
            with self.assertRaises(auth.AuthError, msg=(username, password)):
                auth.create_user(username, password)

    def test_users_cannot_access_each_others_history(self):
        a = auth.create_user("usera", "Secret123")
        b = auth.create_user("userb", "Secret123")
        aid = auth.save_assessment(a["id"], "Random Forest", 0.7, "HIGH",
                                   default_patient(), [{"label": "Age"}])
        self.assertEqual(len(auth.list_assessments(a["id"])), 1)
        self.assertEqual(auth.list_assessments(b["id"]), [])
        self.assertIsNone(auth.get_assessment(b["id"], aid))
        self.assertFalse(auth.delete_assessment(b["id"], aid))
        self.assertIsNotNone(auth.get_assessment(a["id"], aid))
        self.assertTrue(auth.delete_assessment(a["id"], aid))
        self.assertEqual(auth.list_assessments(a["id"]), [])

    def test_sql_injection_attempt_does_not_log_in(self):
        auth.create_user("erin", "Secret123")
        self.assertIsNone(auth.authenticate("erin' OR '1'='1", "x"))
        self.assertIsNone(auth.authenticate("erin", "' OR '1'='1"))


class TestReporting(unittest.TestCase):
    def _assessment(self):
        return {"id": 7, "created_at": "2026-10-13T10:30:00+05:30",
                "model_name": "Random Forest", "probability": 0.72, "risk_level": "HIGH",
                "inputs": default_patient(),
                "factors": [{"label": "Age", "value": "54 years", "typical": "55 years",
                             "influence": 3.2}]}

    def test_pdf_is_generated(self):
        user = {"id": 1, "username": "alice", "full_name": "Alice <b>A</b> & Co"}
        pdf = reporting.build_pdf_report(user, self._assessment())
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 2000)

    def test_pdf_with_model_details(self):
        meta = {"trained_at": "2026-10-13T09:00:00+05:30",
                "config": {"n_train": 242, "n_test": 61},
                "models": {"Random Forest": {
                    "cv": {m: {"mean": 0.8, "std": 0.03} for m in tm.METRICS},
                    "test": {m: 0.81 for m in tm.METRICS}}}}
        pdf = reporting.build_pdf_report({"username": "alice"}, self._assessment(), meta)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_csv_export(self):
        data = reporting.assessments_to_csv([self._assessment()]).decode("utf-8-sig")
        lines = data.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("assessment_id,assessment_date,model"))
        self.assertIn("Random Forest", lines[1])
        self.assertIn("72.0", lines[1])

    def test_csv_export_empty(self):
        data = reporting.assessments_to_csv([]).decode("utf-8-sig")
        self.assertEqual(len(data.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
