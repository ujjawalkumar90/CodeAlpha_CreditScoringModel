"""
CodeAlpha Machine Learning Internship - Task 1: Credit Scoring Model

Objective : Predict an individual's creditworthiness from past financial data.
Approach  : Classification with Logistic Regression, Decision Tree, Random Forest.
Features  : Feature engineering from financial history; evaluation with
            Precision, Recall, F1-Score and ROC-AUC.
Dataset   : UCI Statlog (German Credit Data) - 1000 applicants, 20 attributes
            (income-related, debts, payment history, savings, employment, ...).
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german_credit.csv"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "credit_model.joblib"
RANDOM_STATE = 42

# Labelled copy of the UCI Statlog German Credit data (used if the local CSV is missing).
# The original UCI file (german.data) uses coded values like A11/A12; this copy has readable labels.
DATASET_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"


# --------------------------------------------------------------------------- #
# 1. Dataset download + load
# --------------------------------------------------------------------------- #
def ensure_dataset(path: Path = DATA_PATH, url: str = DATASET_URL) -> Path:
    """Download the German Credit dataset if it is not already in data/."""
    if path.exists():
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ensure_dataset())
    # In the source file credit_risk: 1 = good, 0 = bad.
    # For credit scoring we model the risky class: default = 1 means BAD credit.
    df["default"] = 1 - df["credit_risk"]
    df = df.drop(columns="credit_risk")
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering from financial history
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Monthly repayment burden of the requested credit
    df["credit_per_month"] = df["amount"] / df["duration"]
    # Burden weighted by the share of disposable income the instalment takes
    df["instalment_burden"] = df["credit_per_month"] * df["installment_rate"]
    # Log transform to tame the right-skewed credit amount
    df["log_amount"] = np.log1p(df["amount"])
    # Loan size relative to age (young + large loan = riskier)
    df["amount_per_age"] = df["amount"] / df["age"]
    # Long-term loan flag (> 2 years)
    df["long_term_loan"] = (df["duration"] > 24).astype(int)
    # Age buckets
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 120], labels=["<=25", "26-35", "36-50", "50+"]
    ).astype(str)
    return df


# --------------------------------------------------------------------------- #
# 3. Exploratory plots
# --------------------------------------------------------------------------- #
def save_eda_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.countplot(x="default", data=df, ax=axes[0], hue="default", legend=False)
    axes[0].set_title("Class distribution (1 = bad credit)")
    sns.boxplot(x="default", y="amount", data=df, ax=axes[1], hue="default", legend=False)
    axes[1].set_title("Credit amount vs. default")
    sns.boxplot(x="default", y="duration", data=df, ax=axes[2], hue="default", legend=False)
    axes[2].set_title("Duration (months) vs. default")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #
def build_models(preprocessor_factory):
    def pipe(clf):
        return Pipeline([("prep", preprocessor_factory()), ("clf", clf)])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "Random Forest": (
            pipe(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = engineer_features(load_data())
    print(f"Dataset shape: {df.shape}")
    print("Class balance (default = bad credit):")
    print(df["default"].value_counts(normalize=True).round(3).to_string(), "\n")
    save_eda_plots(df)

    X = df.drop(columns="default")
    y = df["default"]
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    def make_preprocessor():
        return ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted, roc_data = [], {}, {}
    for name, (pipeline, grid) in build_models(make_preprocessor).items():
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best

        proba = best.predict_proba(X_test)[:, 1]
        pred = best.predict(X_test)
        results.append(
            {
                "Model": name,
                "CV ROC-AUC": round(search.best_score_, 4),
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "Precision": round(precision_score(y_test, pred), 4),
                "Recall": round(recall_score(y_test, pred), 4),
                "F1-Score": round(f1_score(y_test, pred), 4),
                "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
            }
        )
        roc_data[name] = roc_curve(y_test, proba)
        print(f"=== {name} === best params: {search.best_params_}")
        print(classification_report(y_test, pred, target_names=["Good credit", "Bad credit"]))

        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Good", "Bad"], cmap="Blues"
        )
        plt.title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=130)
        plt.close()

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n===== Model comparison (test set) =====")
    print(results_df.to_string())
    results_df.to_csv(OUT_DIR / "model_comparison.csv")

    # Save the best model (highest test ROC-AUC) for the Streamlit app
    best_name = results_df["ROC-AUC"].idxmax()
    joblib.dump(
        {"model": fitted[best_name], "model_name": best_name, "metrics": results_df.loc[best_name].to_dict()},
        MODEL_PATH,
    )
    print(f"\nBest model: {best_name} -> saved to {MODEL_PATH}")

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results_df.loc[name, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - credit scoring")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=130)
    plt.close()

    # Feature importance of the Random Forest
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    top = importances.sort_values(ascending=False).head(15)[::-1]
    plt.figure(figsize=(8, 6))
    top.plot.barh(color="steelblue")
    plt.title("Top 15 features - Random Forest")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=130)
    plt.close()
    print("\nTop 10 features (Random Forest):")
    print(top[::-1].head(10).round(4).to_string())
    print(f"\nPlots and tables saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
