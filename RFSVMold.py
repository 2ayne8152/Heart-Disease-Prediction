# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest–Support Vector Machine Model
#
# FINAL MODEL: both RF and SVM are set directly to their best
# hyperparameters (found previously via separate GridSearchCV runs).
# No grid search is performed in this script -- it trains and evaluates
# the final hybrid model only.
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("heart_disease_dataset.csv")
# ==========================
# Separate Features and Target
# ==========================

X = df.drop("Heart Disease", axis=1)
#X = df[["Age", "Cholesterol"]]
y = df["Heart Disease"]

# ==========================
# Detect Numerical and Categorical Columns
# ==========================

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["str"]).columns

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# ==========================
# Handle Missing Values
# ==========================

num_imputer = SimpleImputer(strategy="median")
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])

# ==========================
# Outlier Removal
# ==========================

mask = pd.Series(True, index=X.index)
for col in numerical_columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = X[(X[col] < lower) |
                (X[col] > upper)]

    print(col, len(outliers))

    mask &= (X[col] >= lower) & (X[col] <= upper)
X = X[mask]
y = y[mask]

# ==========================
# One-Hot Encoding
# ==========================

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Best Hyperparameters (from prior tuning)
# ==========================

rf_best_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
}

svm_best_params = {
    "C": 100,
    "gamma": 0.01,
    "kernel": "rbf"
}

# ==========================
# Train Random Forest (tuned)
# ==========================

rf = RandomForestClassifier(
    **rf_best_params,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)
rf_test_prob = rf.predict_proba(X_test)

# ==========================
# Train SVM (tuned, calibrated for probabilities)
# ==========================
# SVC(probability=True) is deprecated (removed once scikit-learn hits
# 1.11) since its internal Platt scaling reuses the same folds as the
# decision function. CalibratedClassifierCV(ensemble=False) is the
# forward-compatible replacement: it fits the SVM on part of the training
# data and calibrates probabilities on a held-out fold.

svm = CalibratedClassifierCV(
    estimator=SVC(**svm_best_params, random_state=42),
    method="sigmoid",
    cv=5,
    ensemble=False
)

svm.fit(X_train, y_train)

svm_train_prob = svm.predict_proba(X_train)
svm_test_prob = svm.predict_proba(X_test)

# ==========================
# Create Meta Features
# ==========================

meta_train = np.hstack((
    rf_train_prob,
    svm_train_prob
))

meta_test = np.hstack((
    rf_test_prob,
    svm_test_prob
))

# ==========================
# Train Logistic Regression (meta-model)
# ==========================

meta_model = LogisticRegression(
    random_state=42
)

meta_model.fit(
    meta_train,
    y_train
)

# ==========================
# Final Prediction
# ==========================

y_pred = meta_model.predict(meta_test)
y_prob = meta_model.predict_proba(meta_test)[:, 1]

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===============================")
print(" Hybrid RF (tuned) + SVM (tuned) + LR Results")
print("===============================\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix Graph (Final Hybrid Model)
# ==========================
 
cm = confusion_matrix(y_test, y_pred)
 
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=meta_model.classes_
)
 
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_aspect("equal")
 
# Attach the colorbar via make_axes_locatable instead of disp.plot's
# default colorbar=True -- this ties the colorbar's height to the axes'
# actual height, so it matches the square matrix instead of stretching
# to the full figure height.
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(disp.im_, cax=cax)
 
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# ROC Curve Graph (Final Hybrid Model)
# ==========================
# pos_label is set explicitly to meta_model.classes_[1] -- the class that
# predict_proba's second column (used for y_prob) actually corresponds
# to -- so the curve/AUC line up with the reported roc_auc_score above
# regardless of whether the target is encoded as 0/1, "No"/"Yes", etc.

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=meta_model.classes_[1])

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"Hybrid Model (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Guess (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("final_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================
# Hybrid Model Learning Curve (Final LR Result)
# ==========================
# CHANGED: "validation" here is no longer an internal CV split carved
# out of X_train -- it is always the SAME held-out X_test/y_test used
# for the final reported metrics above. Only the TRAINING side is varied:
# at each training_sizes fraction, a random subsample of X_train is drawn,
# RF (tuned) + SVM (tuned) are fit on it, meta-features are built for
# both the subsample and X_test, and LR is fit on the subsample's
# meta-features. Training accuracy is scored on the subsample; test
# accuracy is scored on the fixed, never-resampled X_test.
#
# NO INTERNAL FOLDING: unlike the final model's SVM (which uses
# CalibratedClassifierCV(cv=5) to calibrate probabilities on held-out
# folds), the SVM here is SVC(probability=True) fit directly on the
# WHOLE subsample -- no data is carved out for a separate calibration
# fold. probability=True uses SVC's own built-in 5-fold internal Platt
# scaling under the hood (this is the deprecated path mentioned earlier,
# still available pre-1.11), but critically it still trains the
# classifier itself on 100% of the subsample rather than reserving a
# fold purely for calibration the way CalibratedClassifierCV(cv=3) does.
# This is what "no fold" means here: no rows are held back from the
# decision-function fit. Comparing this curve against a folded version
# shows whether reserving data for calibration (fewer effective training
# rows per fit) noticeably hurts accuracy at small subsample sizes.

def hybrid_learning_curve(X_train, y_train, X_test, y_test, rf_params, svm_params,
                           train_sizes=np.linspace(0.1, 1.0, 10), n_repeats=5, random_state=42):
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
 
    sizes_out, train_acc_out, val_acc_out = [], [], []
 
    for frac in train_sizes:
        n_sub = max(int(len(X_train) * frac), 20)
 
        rep_train_acc, rep_val_acc = [], []
 
        for rep in range(n_repeats):
            rng = np.random.RandomState(random_state + rep)
            sample_idx = rng.choice(X_train.index, size=n_sub, replace=False)
 
            X_sub, y_sub = X_train.loc[sample_idx], y_train.loc[sample_idx]
 
            if y_sub.nunique() < 2:
                continue
 
            rf_m = RandomForestClassifier(**rf_params, random_state=random_state)
            # No CalibratedClassifierCV wrapper here -- probability=True
            # fits directly on the full X_sub with no rows held back for
            # a separate calibration fold.
            svm_m = SVC(**svm_params, probability=True, random_state=random_state)
            rf_m.fit(X_sub, y_sub)
            svm_m.fit(X_sub, y_sub)
 
            meta_sub = np.hstack((rf_m.predict_proba(X_sub), svm_m.predict_proba(X_sub)))
            meta_val = np.hstack((rf_m.predict_proba(X_test), svm_m.predict_proba(X_test)))
 
            lr_m = LogisticRegression(random_state=random_state, max_iter=1000)
            lr_m.fit(meta_sub, y_sub)
 
            rep_train_acc.append(accuracy_score(y_sub, lr_m.predict(meta_sub)))
            rep_val_acc.append(accuracy_score(y_test, lr_m.predict(meta_val)))
 
        sizes_out.append(n_sub)
        train_acc_out.append(np.mean(rep_train_acc))
        val_acc_out.append(np.mean(rep_val_acc))
 
    return np.array(sizes_out), np.array(train_acc_out), np.array(val_acc_out)
 
 
hybrid_sizes, hybrid_train_acc, hybrid_val_acc = hybrid_learning_curve(
    X_train,
    y_train,
    X_test,
    y_test,
    rf_best_params,
    svm_best_params
)
 
plt.figure(figsize=(8, 6))
plt.plot(hybrid_sizes, hybrid_train_acc, marker="o", label="Training Accuracy")
plt.plot(hybrid_sizes, hybrid_val_acc, marker="s", label="Validation Accuracy")
plt.title("Hybrid Model Learning Curve (RF + SVM + LR, Final Result)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("hybrid_learning_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Save Hybrid Model
# ==========================
import joblib
import json
import os


# Save the three components of the hybrid model
joblib.dump(rf, f"outputs/rf_model.pkl")
joblib.dump(svm, f"outputs/svm_model.pkl")
joblib.dump(meta_model, f"outputs/meta_model.pkl")

print("\nModels Saved Successfully")
print(f"RF model      : outputs/rf_model.pkl")
print(f"SVM model     : outputs/svm_model.pkl")
print(f"Meta-model    : outputs/meta_model.pkl")


# ==========================
# Save Model Performance
# ==========================

metrics_path = f"outputs/metrics.json"

existing = {}

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        existing = json.load(f)

existing["RF + SVM + LR"] = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(auc)
}

with open(metrics_path, "w") as f:
    json.dump(existing, f, indent=2)

print(f"Metrics saved to: {metrics_path}")