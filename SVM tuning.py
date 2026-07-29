# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest-Support Vector Machine Model
#
# STAGE: SVM is hyperparameter-tuned (GridSearchCV, linear vs rbf kernel
# comparison). RF is left at default settings (no tuning).
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve,
    StratifiedKFold
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

df = pd.read_csv("cleaned_merged_heart_dataset.csv")

print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())

# ==========================
# Separate Features and Target
# ==========================

X = df.drop("target", axis=1)
y = df["target"]

# ==========================
# Detect Numerical and Categorical Columns
# ==========================

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalachh",
    "oldpeak"
]

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

# Numerical Features

num_imputer = SimpleImputer(strategy="median")

X_train[numerical_columns] = num_imputer.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = num_imputer.transform(
    X_test[numerical_columns]
)

# Categorical Features

cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[categorical_columns] = cat_imputer.fit_transform(
    X_train[categorical_columns]
)

X_test[categorical_columns] = cat_imputer.transform(
    X_test[categorical_columns]
)

# ==========================
# One-Hot Encoding
# ==========================

X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    drop_first=True
)

# Ensure both datasets have identical columns

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

# ==========================
# Feature Scaling
# ==========================

continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalachh",
    "oldpeak"
]

scaler = StandardScaler()

X_train[continuous_features] = scaler.fit_transform(
    X_train[continuous_features]
)

X_test[continuous_features] = scaler.transform(
    X_test[continuous_features]
)

# ==========================
# Random Forest (default, no tuning)
# ==========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)

rf_test_prob = rf.predict_proba(X_test)

# ==========================
# SVM Hyperparameter Tuning: Linear vs RBF Kernel
# ==========================
# CalibratedClassifierCV(ensemble=False) wraps SVC to get well-calibrated
# probabilities without relying on the deprecated probability=True param.
#
# Using a LIST of param dicts (not one combined dict) matters here: a
# linear kernel doesn't use gamma at all, so combining kernel/C/gamma into
# one grid would waste fits training identical linear models 4x over
# (once per gamma value) with no effect on the result. Two dicts means
# GridSearchCV only explores gamma where it's actually meaningful (rbf).

svm_parameters = [
    {
        "estimator__kernel": ["linear"],
        "estimator__C": [0.1, 1, 10, 100]
    },
    {
        "estimator__kernel": ["rbf"],
        "estimator__C": [0.1, 1, 10, 100],
        "estimator__gamma": ["scale", 0.1, 0.01, 0.001]
    }
]

calibrated_svc = CalibratedClassifierCV(
    estimator=SVC(random_state=42),
    method="sigmoid",
    cv=5,
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
print(f"SVM best cross-val (out-of-fold) accuracy: {svm_grid.best_score_:.4f}")

svm = svm_grid.best_estimator_

svm_train_prob = svm.predict_proba(X_train)

svm_test_prob = svm.predict_proba(X_test)

# ==========================
# SVM Tuning: Accuracy Across Iterations (Linear vs RBF)
# ==========================

svm_cv_results = pd.DataFrame(svm_grid.cv_results_)
best_idx = svm_grid.best_index_
kernels = svm_cv_results["param_estimator__kernel"].astype(str)

svm_table = svm_cv_results[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values(by="rank_test_score")

print(svm_table.to_string(index=False))

kernel_colors = {"linear": "tab:blue", "rbf": "tab:orange"}

# --- Chart 1: every candidate, in run order, colored by kernel ---
plt.figure(figsize=(12, 6))
for kernel, color in kernel_colors.items():
    mask = kernels == kernel
    plt.scatter(
        np.where(mask)[0],
        svm_cv_results.loc[mask, "mean_test_score"],
        color=color,
        alpha=0.7,
        s=40,
        label=kernel
    )
plt.scatter(
    best_idx,
    svm_cv_results.loc[best_idx, "mean_test_score"],
    color="red",
    zorder=5,
    s=100,
    label=f"Best: {svm_cv_results.loc[best_idx, 'mean_test_score']:.4f} "
          f"({svm_cv_results.loc[best_idx, 'param_estimator__kernel']})"
)
plt.xlabel("Iteration (candidate index, run order)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("SVM Tuning: Accuracy Across All Iterations (Linear vs RBF)")
plt.legend()
plt.grid(True)
plt.savefig("svm_tuning_iterations.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Chart 2: same data, ranked best-to-worst, colored by kernel ---
svm_cv_sorted = svm_cv_results.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
kernels_sorted = svm_cv_sorted["param_estimator__kernel"].astype(str)

plt.figure(figsize=(12, 6))
plt.plot(
    range(len(svm_cv_sorted)),
    svm_cv_sorted["mean_test_score"],
    color="gray",
    alpha=0.3,
    linewidth=1
)
for kernel, color in kernel_colors.items():
    mask = kernels_sorted == kernel
    plt.scatter(
        np.where(mask)[0],
        svm_cv_sorted.loc[mask, "mean_test_score"],
        color=color,
        alpha=0.8,
        s=30,
        label=kernel
    )
plt.xlabel("Rank (best to worst)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("SVM Tuning: Iterations Ranked by Accuracy (Linear vs RBF)")
plt.legend()
plt.grid(True)
plt.savefig("svm_tuning_ranked.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Chart 3: direct kernel comparison (the headline chart) ---
kernel_summary = svm_cv_results.groupby(kernels)["mean_test_score"].agg(["max", "mean"])

x = np.arange(len(kernel_summary))
width = 0.35

plt.figure(figsize=(7, 6))
plt.bar(x - width / 2, kernel_summary["max"], width, label="Best CV accuracy",
        color=[kernel_colors[k] for k in kernel_summary.index])
plt.bar(x + width / 2, kernel_summary["mean"], width, label="Mean CV accuracy (all candidates)",
        color=[kernel_colors[k] for k in kernel_summary.index], alpha=0.5)
plt.xticks(x, kernel_summary.index)
plt.ylabel("Cross-Validated Accuracy")
plt.title("SVM Kernel Comparison: Linear vs RBF")
plt.legend()
plt.grid(True, axis="y")
for i, (mx, mn) in enumerate(zip(kernel_summary["max"], kernel_summary["mean"])):
    plt.text(i - width / 2, mx + 0.003, f"{mx:.4f}", ha="center", fontsize=8)
    plt.text(i + width / 2, mn + 0.003, f"{mn:.4f}", ha="center", fontsize=8)
plt.savefig("svm_kernel_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Chart 4: per-parameter breakdown, split by kernel ---
linear_results = svm_cv_results[kernels == "linear"]
rbf_results = svm_cv_results[kernels == "rbf"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

c_linear = linear_results.groupby("param_estimator__C")["mean_test_score"].max()
axes[0].plot(c_linear.index.astype(float), c_linear.values, marker="o", color=kernel_colors["linear"])
axes[0].set_xscale("log")
axes[0].set_title("Linear Kernel: C")
axes[0].set_xlabel("C")
axes[0].set_ylabel("Accuracy")
axes[0].grid(True)

c_rbf = rbf_results.groupby("param_estimator__C")["mean_test_score"].max()
axes[1].plot(c_rbf.index.astype(float), c_rbf.values, marker="o", color=kernel_colors["rbf"])
axes[1].set_xscale("log")
axes[1].set_title("RBF Kernel: C")
axes[1].set_xlabel("C")
axes[1].grid(True)

gamma_rbf = rbf_results.groupby("param_estimator__gamma")["mean_test_score"].max()
axes[2].bar(gamma_rbf.index.astype(str), gamma_rbf.values, color=kernel_colors["rbf"])
axes[2].set_title("RBF Kernel: gamma")
axes[2].set_xlabel("gamma")
axes[2].grid(True, axis="y")

plt.tight_layout()
plt.savefig("svm_tuning_per_parameter.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Head-to-Head: Best Linear vs Best RBF on the Held-Out Test Set
# ==========================
# The charts above compare kernels on cross-validated TRAINING accuracy.
# This refits the single best linear-kernel model and the single best
# rbf-kernel model on the full training set and evaluates both on the
# untouched test set -- the actual evidence for which kernel generalizes
# better on this dataset, not just which fit the training folds better.

best_linear_row = linear_results.sort_values("mean_test_score", ascending=False).iloc[0]
best_rbf_row = rbf_results.sort_values("mean_test_score", ascending=False).iloc[0]

linear_final = CalibratedClassifierCV(
    estimator=SVC(kernel="linear", C=best_linear_row["param_estimator__C"], random_state=42),
    method="sigmoid",
    cv=5,
    ensemble=False
)
linear_final.fit(X_train, y_train)

rbf_final = CalibratedClassifierCV(
    estimator=SVC(
        kernel="rbf",
        C=best_rbf_row["param_estimator__C"],
        gamma=best_rbf_row["param_estimator__gamma"],
        random_state=42
    ),
    method="sigmoid",
    cv=5,
    ensemble=False
)
rbf_final.fit(X_train, y_train)

linear_test_pred = linear_final.predict(X_test)
linear_test_prob = linear_final.predict_proba(X_test)[:, 1]

rbf_test_pred = rbf_final.predict(X_test)
rbf_test_prob = rbf_final.predict_proba(X_test)[:, 1]

kernel_test_comparison = pd.DataFrame({
    "Kernel": ["Linear", "RBF"],
    "Test Accuracy": [
        accuracy_score(y_test, linear_test_pred),
        accuracy_score(y_test, rbf_test_pred)
    ],
    "Test ROC-AUC": [
        roc_auc_score(y_test, linear_test_prob),
        roc_auc_score(y_test, rbf_test_prob)
    ]
})

print("\nHead-to-Head on Held-Out Test Set")
print(kernel_test_comparison.round(4).to_string(index=False))

linear_fpr, linear_tpr, _ = roc_curve(y_test, linear_test_prob)
rbf_fpr, rbf_tpr, _ = roc_curve(y_test, rbf_test_prob)

plt.figure(figsize=(8, 6))
plt.plot(linear_fpr, linear_tpr, color=kernel_colors["linear"],
         label=f"Linear (AUC={roc_auc_score(y_test, linear_test_prob):.3f})")
plt.plot(rbf_fpr, rbf_tpr, color=kernel_colors["rbf"], linewidth=2,
         label=f"RBF (AUC={roc_auc_score(y_test, rbf_test_prob):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Held-Out Test ROC: Best Linear vs Best RBF SVM")
plt.legend()
plt.grid(True)
plt.savefig("svm_kernel_roc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

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
# Train Logistic Regression
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
print(" Hybrid RF (default) + SVM (tuned) + LR Results")
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


plot_learning_curve(
    rf,
    X_train,
    y_train,
    "Random Forest Learning Curve (Default, base learner only)"
)

best_kernel = svm_grid.best_params_["estimator__kernel"]
plot_learning_curve(
    svm,
    X_train,
    y_train,
    f"Support Vector Machine Learning Curve (Tuned - {best_kernel} kernel, base learner only)"
)

# ==========================
# Hybrid Model Learning Curve (Final LR Result)
# ==========================
# Rebuilds the whole pipeline at each training size: fit RF (default) +
# SVM (tuned kernel/C/gamma) on the subsample, build meta-features, fit
# LR, then score the LR's own predictions on both the training subsample
# and a held-out validation fold -- the final hybrid result, not a base
# learner in isolation.

svm_best_svc_params = {
    k.replace("estimator__", ""): v
    for k, v in svm_grid.best_params_.items()
}


def hybrid_learning_curve(X, y, svm_params, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    sizes_out, train_acc_out, val_acc_out = [], [], []

    for frac in train_sizes:
        fold_train_acc, fold_val_acc = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr_full, y_tr_full = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            n_sub = max(int(len(X_tr_full) * frac), 20)
            X_sub, y_sub = X_tr_full.iloc[:n_sub], y_tr_full.iloc[:n_sub]

            if y_sub.nunique() < 2:
                continue

            rf_m = RandomForestClassifier(random_state=random_state)
            # cv=3 here (vs. 5 elsewhere) since the smallest training_sizes
            # subsample can be as few as 20 rows -- keeps each calibration
            # fold large enough to be meaningful
            svm_m = CalibratedClassifierCV(
                estimator=SVC(**svm_params, random_state=random_state),
                method="sigmoid",
                cv=3,
                ensemble=False
            )
            rf_m.fit(X_sub, y_sub)
            svm_m.fit(X_sub, y_sub)

            meta_sub = np.hstack((rf_m.predict_proba(X_sub), svm_m.predict_proba(X_sub)))
            meta_val = np.hstack((rf_m.predict_proba(X_val), svm_m.predict_proba(X_val)))

            lr_m = LogisticRegression(random_state=random_state, max_iter=1000)
            lr_m.fit(meta_sub, y_sub)

            fold_train_acc.append(accuracy_score(y_sub, lr_m.predict(meta_sub)))
            fold_val_acc.append(accuracy_score(y_val, lr_m.predict(meta_val)))

        sizes_out.append(n_sub)
        train_acc_out.append(np.mean(fold_train_acc))
        val_acc_out.append(np.mean(fold_val_acc))

    return np.array(sizes_out), np.array(train_acc_out), np.array(val_acc_out)


hybrid_sizes, hybrid_train_acc, hybrid_val_acc = hybrid_learning_curve(
    X_train,
    y_train,
    svm_best_svc_params
)

plt.figure(figsize=(8, 6))
plt.plot(hybrid_sizes, hybrid_train_acc, marker="o", label="Training Accuracy (Final Hybrid)")
plt.plot(hybrid_sizes, hybrid_val_acc, marker="s", label="Validation Accuracy (Final Hybrid)")
plt.title("Hybrid Model Learning Curve (RF + SVM + LR, Final Result)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("hybrid_learning_curve.png", dpi=300, bbox_inches="tight")
plt.show()