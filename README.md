# CodeAlpha_CreditScoringModel

Machine Learning Internship @ **CodeAlpha** - **Task 1: Credit Scoring Model**

## Objective
Predict an individual's creditworthiness (good vs. bad credit) using past financial data.

## Dataset
UCI Statlog **German Credit Data** - 1000 applicants, 20 attributes (checking/savings status,
credit history, loan purpose, amount, duration, employment, instalment rate, age, housing, etc.).
A labelled CSV copy is stored in `data/german_credit.csv`.
Target: `default = 1` means **bad credit** (30% of applicants), `0` means good credit (70%).

## Approach
Classification with three algorithms: **Logistic Regression, Decision Tree, Random Forest**.
Each model is tuned with 5-fold stratified `GridSearchCV` (scoring = ROC-AUC) and uses
`class_weight="balanced"` because the data is imbalanced.

## Feature Engineering (from financial history)
| Feature | Meaning |
|---|---|
| `credit_per_month` | amount / duration - monthly repayment burden |
| `instalment_burden` | credit_per_month x instalment rate |
| `log_amount` | log of credit amount (reduces skew) |
| `amount_per_age` | loan size relative to applicant age |
| `long_term_loan` | 1 if duration > 24 months |
| `age_group` | age buckets (<=25, 26-35, 36-50, 50+) |

Numeric columns are standardised, categorical columns one-hot encoded (inside a `Pipeline`, so there is no data leakage).

## Evaluation
Precision, Recall, F1-Score and ROC-AUC on a stratified 80/20 hold-out test set
(metrics are for the **bad credit** class, the one a lender wants to catch).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.720 | 0.521 | 0.817 | 0.636 | 0.811 |
| Decision Tree | 0.635 | 0.427 | 0.633 | 0.510 | 0.676 |
| Random Forest | 0.800 | 0.667 | 0.667 | 0.667 | 0.807 |

Logistic Regression catches the most bad-credit applicants (highest recall), Random Forest gives
the best balance (highest accuracy and F1). Full results are in `outputs/model_comparison.csv`;
plots (EDA, ROC curves, confusion matrices, feature importance) are in `outputs/`.

## Trained model
The best model on the test set (highest ROC-AUC, **Logistic Regression**) is saved together with its
preprocessing pipeline to `models/credit_model.joblib`.
`ensure_dataset()` in `credit_scoring.py` downloads the dataset automatically if `data/german_credit.csv` is missing.

## How to run
```bash
pip install -r requirements.txt
python credit_scoring.py          # train, evaluate, save plots + model
streamlit run app.py              # web interface for predictions
python -m pytest tests            # run the tests
```

## Streamlit interface
`app.py` lets you enter an applicant's details (20 attributes), then shows the probability of
good / bad credit and a risk verdict. It loads `models/credit_model.joblib`, so train once first.

## Testing
`tests/` checks the dataset (size, no missing values, 30% bad credit), the engineered features,
the saved model (valid probabilities, ROC-AUC > 0.7, risky applicants score higher) and the Streamlit app.

## Project structure
```
CodeAlpha_CreditScoringModel/
|-- credit_scoring.py
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/german_credit.csv
|-- models/credit_model.joblib
|-- outputs/   (plots + model_comparison.csv)
`-- tests/
```
