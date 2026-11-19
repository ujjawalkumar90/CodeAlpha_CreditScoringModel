import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from credit_scoring import MODEL_PATH, engineer_features, ensure_dataset, load_data  # noqa: E402


def test_dataset_loads_and_is_clean():
    df = load_data()
    assert df.shape[0] == 1000
    assert df.isna().sum().sum() == 0
    assert set(df["default"].unique()) == {0, 1}
    assert abs(df["default"].mean() - 0.3) < 0.01  # 30% bad credit


def test_ensure_dataset_returns_existing_file():
    assert ensure_dataset().exists()


def test_feature_engineering_adds_columns():
    df = engineer_features(load_data())
    for col in ["credit_per_month", "instalment_burden", "log_amount", "amount_per_age", "long_term_loan", "age_group"]:
        assert col in df.columns
    assert (df["credit_per_month"] > 0).all()
    assert set(df["long_term_loan"].unique()) <= {0, 1}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run credit_scoring.py first")
def test_saved_model_predicts_valid_probabilities():
    bundle = joblib.load(MODEL_PATH)
    X = engineer_features(load_data()).drop(columns="default")
    proba = bundle["model"].predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert bundle["metrics"]["ROC-AUC"] > 0.7


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run credit_scoring.py first")
def test_risky_applicant_scores_higher_than_safe_one():
    bundle = joblib.load(MODEL_PATH)
    df = engineer_features(load_data())
    X = df.drop(columns="default")
    proba = pd.Series(bundle["model"].predict_proba(X)[:, 1], index=X.index)
    assert proba[df["default"] == 1].mean() > proba[df["default"] == 0].mean()
