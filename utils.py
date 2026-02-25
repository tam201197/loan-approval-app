import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)
import json
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os

CSV_FILE = "loan_data.csv"

COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file"
]

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=COLUMNS)

def save_data(df):
    df.to_csv(CSV_FILE, index=False)


def load_model(df):
    y = df["loan_status"]
    X = df.drop("loan_status", axis=1)
    

    # Define feature groups
    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score"
    ]

    categorical_features = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file"
    ]

    # Preprocessing
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    overfit_diff = train_acc - test_acc

    return model, train_acc, test_acc, overfit_diff


def validate_models(df, preprocessor, log_callback=None):

    df = df.dropna(subset=["loan_status"])

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"classifier__C": [0.1, 1, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10]
            }
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss"),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 6]
            }
        )
    }

    results = []
    best_auc = 0
    best_model_pipeline = None
    best_name = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, param_grid) in models.items():

        if log_callback:
            log_callback(f"🔍 Starting validation for {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1
        )

        if log_callback:
            log_callback("   ⏳ Running GridSearchCV...") 

        grid.fit(X, y)

        if log_callback:
            log_callback(f"   ✅ Best ROC-AUC: {grid.best_score_:.4f}")
            log_callback(f"   ⚙ Best Params: {grid.best_params_}")

        y_pred = grid.predict(X)
        y_prob = grid.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_prob)

        results.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "ROC-AUC": auc
        })

        if auc > best_auc:
            best_auc = auc
            best_model_pipeline = grid.best_estimator_
            best_name = name
            best_params = grid.best_params_

    if log_callback:
        log_callback("💾 Saving best model...")

    # 🔥 Auto-save best model
    joblib.dump(best_model_pipeline, "best_model.pkl")

    # Save metadata
    metadata = {
        "model_name": best_name,
        "roc_auc": float(best_auc),
        "best_params": best_params,
        "training_samples": len(df),
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("best_model_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)

    if log_callback:
        log_callback("💾 Best model and metadata saved.")

    return pd.DataFrame(results), best_model_pipeline, best_name

def build_preprocessor(df):
    """
    Build preprocessing pipeline for numeric and categorical features.
    Does NOT fit — just defines transformation logic.
    """

    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score"
    ]

    categorical_features = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file"
    ]

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor