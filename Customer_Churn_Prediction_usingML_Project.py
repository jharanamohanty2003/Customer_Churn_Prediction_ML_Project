
# Customer Churn Prediction Project
# Author: Jharana Mohanty

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

from imblearn.over_sampling import SMOTE


# -----------------------------
# Load Dataset
# -----------------------------
DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print(df.head())


# -----------------------------
# Data Preprocessing
# -----------------------------
def preprocess_data(dataframe):
    df = dataframe.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature Engineering
    df["LifetimeValue"] = df["MonthlyCharges"] * df["tenure"]

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df


df = preprocess_data(df)

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Feature Selection
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X, y)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_selector.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = feature_importance["Feature"].head(15).values

print("\nTop Features:")
print(top_features)

X_selected = df[top_features]


# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Save scaler
joblib.dump(scaler, "scaler.pkl")


# -----------------------------
# Handle Imbalanced Data
# -----------------------------
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)


# -----------------------------
# XGBoost Model
# -----------------------------
xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=5,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_

print("\nBest XGBoost Parameters:")
print(xgb_search.best_params_)


# -----------------------------
# CatBoost Model
# -----------------------------
cat_model = CatBoostClassifier(
    verbose=0,
    random_state=42
)

cat_model.fit(X_train, y_train)


# -----------------------------
# LightGBM Model
# -----------------------------
lgb_model = lgb.LGBMClassifier(random_state=42)

lgb_model.fit(X_train, y_train)


# -----------------------------
# Stacking Model
# -----------------------------
stack_model = StackingClassifier(
    estimators=[
        ("xgb", best_xgb),
        ("cat", cat_model),
        ("lgb", lgb_model)
    ],
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

stack_model.fit(X_train, y_train)


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return predictions


# -----------------------------
# Model Evaluation
# -----------------------------
y_pred_xgb = evaluate_model(
    best_xgb,
    X_test,
    y_test,
    "XGBoost"
)

y_pred_cat = evaluate_model(
    cat_model,
    X_test,
    y_test,
    "CatBoost"
)

y_pred_lgb = evaluate_model(
    lgb_model,
    X_test,
    y_test,
    "LightGBM"
)

y_pred_stack = evaluate_model(
    stack_model,
    X_test,
    y_test,
    "Stacking Model"
)


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_stack)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Stacking Model Confusion Matrix")
plt.show()


# -----------------------------
# Save Final Model
# -----------------------------
joblib.dump(best_xgb, "xgboost_churn_model.pkl")
joblib.dump(top_features, "top_features.pkl")

print("\nModel and files saved successfully.")


# -----------------------------
# Project Summary
# -----------------------------
print("\nPROJECT SUMMARY")
print("-" * 50)

print("""
1. Data preprocessing performed using encoding and scaling.
2. Missing values handled using median imputation.
3. Feature selection performed using Random Forest importance.
4. SMOTE used for class imbalance handling.
5. Multiple ML models implemented:
   - XGBoost
   - CatBoost
   - LightGBM
   - Stacking Classifier
6. Hyperparameter tuning performed using RandomizedSearchCV.
7. Model evaluation done using Accuracy, Precision, Recall, and F1-score.
8. Final trained model saved using Joblib.
""")
