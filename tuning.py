# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest–Support Vector Machine Model
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict,
    learning_curve
)
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
y = df["Heart Disease"]

# ==========================
# Detect Numerical and Categorical Columns
# ==========================

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

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
# Handle Missing Values
# ==========================

num_imputer = SimpleImputer(strategy="median")
X_train[numerical_columns] = num_imputer.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = num_imputer.transform(X_test[numerical_columns])

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = cat_imputer.transform(X_test[categorical_columns])

# ==========================
# One-Hot Encoding
# ==========================

X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Ensure both datasets have identical columns (any category unseen in
# training collapses to the reference/dropped category in X_test)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# ==========================
# Random Forest Hyperparameter Tuning
# ==========================

rf_parameters = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nBest Random Forest Parameters")
print(rf_grid.best_params_)

rf = rf_grid.best_estimator_

# ==========================
# SVM Hyperparameter Tuning
# ==========================
# NOTE: previously this used a manual double for-loop that wrapped SVC in
# CalibratedClassifierCV(cv=5) and ran that inside cross_val_predict(cv=5)
# -- a nested 5x5 CV repeated for every (C, gamma) pair (~400 SVM fits).
# A single GridSearchCV does the same job far more cheaply and gives us
# real cv_results_ to plot later.
#
# SVC(probability=True) is deprecated as of sklearn 1.9 (removed in 1.11).
# sklearn's own internal Platt scaling for probability=True is also known
# to be slightly biased since it's fit on the same folds used for the
# decision function. CalibratedClassifierCV(ensemble=False) is the
# recommended replacement: it fits the SVM on part of the data and
# calibrates probabilities on a held-out fold, which is both the
# forward-compatible fix and the more methodologically correct one.
# Param names get an "estimator__" prefix since SVC is now wrapped.

svm_parameters = {
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__gamma": ["scale", 0.1, 0.01, 0.001]
}

calibrated_svc = CalibratedClassifierCV(
    estimator=SVC(kernel="rbf", random_state=42),
    method="sigmoid",
    cv=3,          # keeps runtime reasonable; bump to 5 for a more thorough search
    ensemble=False
)

svm_grid = GridSearchCV(
    estimator=calibrated_svc,
    param_grid=svm_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\nBest SVM Parameters")
print(svm_grid.best_params_)

svm = svm_grid.best_estimator_

# ==========================
# Generate Out-of-Fold Predictions
# ==========================

rf_oof = cross_val_predict(rf, X_train, y_train, cv=5, method="predict_proba")
svm_oof = cross_val_predict(svm, X_train, y_train, cv=5, method="predict_proba")

# Fit models on full training data
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

rf_test_prob = rf.predict_proba(X_test)
svm_test_prob = svm.predict_proba(X_test)

# Meta-features
meta_train = np.hstack((rf_oof, svm_oof))
meta_test = np.hstack((rf_test_prob, svm_test_prob))

# ==========================
# Train Logistic Regression
# ==========================

meta_model = LogisticRegression(
    C=1,
    solver="lbfgs",
    random_state=42,
    max_iter=1000
)

meta_model.fit(meta_train, y_train)

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
print(" Hybrid RF + SVM + LR Results")
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
# Overfitting Check: Train vs Test Performance
# ==========================
# High test scores alone don't tell you whether a model is overfitting --
# overfitting is a GAP between train and test/validation performance.
# This block computes the same metrics on the training set and compares.
# A large positive gap (train >> test) suggests overfitting. Both scores
# being suspiciously high with a small gap (as here) instead points to an
# easy/separable dataset, a leaked feature, or a small test set -- worth
# checking for duplicate rows or a feature that encodes the target.

y_train_pred = meta_model.predict(meta_train)
y_train_prob = meta_model.predict_proba(meta_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_prob)

print("\n===============================")
print(" Overfitting Check (Train vs Test)")
print("===============================\n")
print(f"{'Metric':<12}{'Train':>10}{'Test':>10}{'Gap':>10}")
for name, tr, te in [
    ("Accuracy", train_accuracy, accuracy),
    ("Precision", train_precision, precision),
    ("Recall", train_recall, recall),
    ("F1", train_f1, f1),
    ("ROC-AUC", train_auc, auc),
]:
    print(f"{name:<12}{tr:>10.4f}{te:>10.4f}{tr - te:>10.4f}")

# Cross-validated training scores from the grids give another reference
# point -- these are already out-of-fold, so compare them to the test
# score too, not just to the in-sample train score above.
print(f"\nRF best cross-val (out-of-fold) accuracy : {rf_grid.best_score_:.4f}")
print(f"SVM best cross-val (out-of-fold) accuracy: {svm_grid.best_score_:.4f}")

overfit_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
train_scores = [train_accuracy, train_precision, train_recall, train_f1, train_auc]
test_scores = [accuracy, precision, recall, f1, auc]

x = np.arange(len(overfit_metrics))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width / 2, train_scores, width, label="Train (in-sample)")
plt.bar(x + width / 2, test_scores, width, label="Test (held-out)")
plt.xticks(x, overfit_metrics)
plt.ylim(0, 1.08)
plt.ylabel("Score")
plt.title("Hybrid Model: Train vs Test Performance (Overfitting Check)")
plt.legend()
plt.grid(True, axis="y")

for i, (tr, te) in enumerate(zip(train_scores, test_scores)):
    plt.text(i - width / 2, tr + 0.015, f"{tr:.3f}", ha="center", fontsize=8)
    plt.text(i + width / 2, te + 0.015, f"{te:.3f}", ha="center", fontsize=8)

plt.savefig("overfit_check.png", dpi=300, bbox_inches="tight")
plt.show()


def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, validation_mean, marker="s", label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_learning_curve(rf, X_train, y_train, "Random Forest Learning Curve")
plot_learning_curve(svm, X_train, y_train, "Support Vector Machine Learning Curve")

# ==========================
# Parameter Visualization Dashboard
# ==========================

rf_results = pd.DataFrame(rf_grid.cv_results_)
svm_results = pd.DataFrame(svm_grid.cv_results_)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RF n_estimators
n_est = rf_results.groupby("param_n_estimators")["mean_test_score"].max()
axes[0, 0].plot(n_est.index, n_est.values, marker="o")
axes[0, 0].set_title("RF: Number of Trees")
axes[0, 0].set_xlabel("n_estimators")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].grid(True)

# RF max_depth
depth = rf_results.dropna(subset=["param_max_depth"])
depth = depth.groupby("param_max_depth")["mean_test_score"].max()
axes[0, 1].plot(depth.index.astype(int), depth.values, marker="o")
axes[0, 1].set_title("RF: Max Depth")
axes[0, 1].set_xlabel("max_depth")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].grid(True)

# SVM C -- now uses REAL scores from svm_grid.cv_results_
svm_c = svm_results.groupby("param_estimator__C")["mean_test_score"].max()
axes[1, 0].semilogx(svm_c.index.astype(float), svm_c.values, marker="o")
axes[1, 0].set_title("SVM: C Parameter")
axes[1, 0].set_xlabel("C")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].grid(True)

# SVM gamma -- gamma mixes a string ("scale") with floats, so use a
# categorical bar chart instead of a log-scale line plot
svm_gamma = svm_results.groupby("param_estimator__gamma")["mean_test_score"].max()
axes[1, 1].bar(svm_gamma.index.astype(str), svm_gamma.values)
axes[1, 1].set_title("SVM: Gamma Parameter")
axes[1, 1].set_xlabel("gamma")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].grid(True, axis="y")

plt.tight_layout()
plt.savefig("parameter_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Model Comparison Table
# ==========================

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

svm_pred = svm.predict(X_test)
svm_prob = svm.predict_proba(X_test)[:, 1]

comparison = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Hybrid RF+SVM+LR"],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred),
        accuracy
    ],
    "Precision": [
        precision_score(y_test, rf_pred),
        precision_score(y_test, svm_pred),
        precision
    ],
    "Recall": [
        recall_score(y_test, rf_pred),
        recall_score(y_test, svm_pred),
        recall
    ],
    "F1": [
        f1_score(y_test, rf_pred),
        f1_score(y_test, svm_pred),
        f1
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, svm_prob),
        auc
    ]
})

print("\nModel Comparison")
print(comparison.round(4))

comparison.to_csv("model_comparison.csv", index=False)

# ==========================
# ROC Curve Comparison
# ==========================

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
hy_fpr, hy_tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"RF (AUC={roc_auc_score(y_test, rf_prob):.3f})")
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC={roc_auc_score(y_test, svm_prob):.3f})")
plt.plot(hy_fpr, hy_tpr, label=f"Hybrid (AUC={auc:.3f})", linewidth=3)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("roc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Confusion Matrix Heatmap
# ==========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid Model Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Feature Importance
# ==========================

importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="Importance", y="Feature")
plt.title("Top 10 Important Features")

plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()