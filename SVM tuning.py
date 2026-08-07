# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest–Support Vector Machine Model
#
# STAGE: RF is left at default parameters, SVM is hyperparameter-tuned
# (GridSearchCV).
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
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
# Handle Missing Values
# ==========================

num_imputer = SimpleImputer(strategy="median")
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])

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
# Train Random Forest (vanilla, no tuning)
# ==========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)

rf_test_prob = rf.predict_proba(X_test)

# ==========================
# SVM Hyperparameter Tuning
# ==========================
# GridSearchCV tunes the raw SVC (C, gamma, kernel) on accuracy. The
# tuned hyperparameters are then carried into a CalibratedClassifierCV
# wrapper below to get well-calibrated probabilities for stacking --
# SVC(probability=True) is deprecated (removed once scikit-learn hits
# 1.11) since its internal Platt scaling reuses the same folds as the
# decision function. CalibratedClassifierCV(ensemble=False) is the
# forward-compatible replacement: it fits the SVM on part of the training
# data and calibrates probabilities on a held-out fold.

svm_parameters = [
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001]
    },
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100]
        # linear kernel has no gamma parameter, so it's left out here --
        # including it would just re-run the same linear fit once per
        # gamma value for no benefit
    }
]

svm_grid = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=svm_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\nBest SVM Parameters")
print(svm_grid.best_params_)
print(f"SVM best cross-val (out-of-fold) accuracy: {svm_grid.best_score_:.4f}")

# ==========================
# SVM Tuning: Accuracy Across Iterations
# ==========================
# Every row in cv_results_ is one candidate (one combination of
# C/gamma/kernel) that GridSearchCV evaluated with 5-fold CV. This plots
# the accuracy of every single iteration, in the order they were run, so
# you can see how much the choice of hyperparameters actually moved the
# needle.

svm_cv_results = pd.DataFrame(svm_grid.cv_results_)
best_idx = svm_grid.best_index_

plt.figure(figsize=(12, 6))
plt.errorbar(
    range(len(svm_cv_results)),
    svm_cv_results["mean_test_score"],
    yerr=svm_cv_results["std_test_score"],
    fmt="o",
    markersize=3,
    ecolor="lightgray",
    elinewidth=1,
    capsize=0,
    alpha=0.7,
    label="Each candidate (5-fold CV mean \u00b1 std)"
)
plt.scatter(
    best_idx,
    svm_cv_results.loc[best_idx, "mean_test_score"],
    color="red",
    zorder=5,
    s=80,
    label=f"Best: {svm_cv_results.loc[best_idx, 'mean_test_score']:.4f}"
)
plt.xlabel("Iteration (candidate index, run order)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("SVM Tuning: Accuracy Across All Iterations")
plt.legend()
plt.grid(True)
plt.savefig("svm_tuning_iterations.png", dpi=300, bbox_inches="tight")
plt.show()

# Same data sorted best-to-worst -- makes it easy to see how many
# candidates were close contenders vs. how quickly accuracy drops off
svm_cv_sorted = svm_cv_results.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(
    range(len(svm_cv_sorted)),
    svm_cv_sorted["mean_test_score"],
    marker="o",
    markersize=3,
    linewidth=1
)
plt.scatter(0, svm_cv_sorted["mean_test_score"].iloc[0], color="red", zorder=5,
            label=f"Best: {svm_cv_sorted['mean_test_score'].iloc[0]:.4f}")
plt.scatter(len(svm_cv_sorted) - 1, svm_cv_sorted["mean_test_score"].iloc[-1], color="gray", zorder=5,
            label=f"Worst: {svm_cv_sorted['mean_test_score'].iloc[-1]:.4f}")
plt.xlabel("Rank (best to worst)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("SVM Tuning: Iterations Ranked by Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("svm_tuning_ranked.png", dpi=300, bbox_inches="tight")
plt.show()

# Per-hyperparameter view: best accuracy achieved at each individual
# parameter value (other params marginalized out via max)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

panels = [
    ("param_C", "C", False),
    ("param_gamma", "gamma", True),
    ("param_kernel", "kernel", True),
]

for ax, (col, label, categorical) in zip(axes.flat, panels):
    grouped = svm_cv_results.dropna(subset=[col]).groupby(col)["mean_test_score"].max()
    if categorical:
        ax.bar(grouped.index.astype(str), grouped.values)
    else:
        try:
            x_vals = grouped.index.astype(float)
            ax.plot(x_vals, grouped.values, marker="o")
        except (TypeError, ValueError):
            ax.bar(grouped.index.astype(str), grouped.values)
    ax.set_title(f"SVM: {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Best Accuracy")
    ax.grid(True)

plt.tight_layout()
plt.savefig("svm_tuning_per_parameter.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# Train SVM (tuned hyperparameters, calibrated for probabilities)
# ==========================

svm = CalibratedClassifierCV(
    estimator=SVC(random_state=42, **svm_grid.best_params_),
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

y_prob = meta_model.predict_proba(meta_test)[:,1]

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("\n===============================")
print(" Hybrid RF (vanilla) + SVM (tuned) + LR Results")
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
        train_sizes=np.linspace(0.1,1.0,10),
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="s",
        label="Validation Accuracy"
    )

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

plot_learning_curve(
    svm,
    X_train,
    y_train,
    "Support Vector Machine Learning Curve (Tuned, base learner only)"
)

# ==========================
# Hybrid Model Learning Curve (Final LR Result)
# ==========================
# The two curves above only show each BASE learner in isolation --
# neither one goes through the stacking process, so neither reflects the
# final LR output. sklearn's learning_curve() can't do that automatically
# here since the stacking is hand-rolled rather than a Pipeline/
# StackingClassifier, so this rebuilds the whole pipeline at each
# training size: fit RF (default hyperparameters) + SVM (tuned
# hyperparameters) on the subsample, build meta-features, fit LR, then
# score the LR's own predictions -- i.e. the actual final hybrid result
# -- on both the training subsample and a held-out validation fold.

from sklearn.model_selection import StratifiedKFold

def hybrid_learning_curve(X, y, svm_best_params, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, random_state=42):
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
                estimator=SVC(random_state=random_state, **svm_best_params),
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
    svm_grid.best_params_
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