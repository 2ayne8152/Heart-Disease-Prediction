# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest–Support Vector Machine Model
#
# STAGE: RF is hyperparameter-tuned (GridSearchCV), SVM is left vanilla.
# Next stage (later): fix RF at its tuned settings, tune SVM instead.
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

df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# ==========================
# Separate Features and Target
# ==========================

X = df.drop("target", axis=1)
y = df["target"]

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

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

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
print(f"RF best cross-val (out-of-fold) accuracy: {rf_grid.best_score_:.4f}")

rf = rf_grid.best_estimator_

# ==========================
# RF Tuning: Accuracy Across Iterations
# ==========================
# Every row in cv_results_ is one candidate (one combination of
# n_estimators/max_depth/min_samples_split/min_samples_leaf/max_features)
# that GridSearchCV evaluated with 5-fold CV. This plots the accuracy of
# every single iteration, in the order they were run, so you can see how
# much the choice of hyperparameters actually moved the needle.

rf_cv_results = pd.DataFrame(rf_grid.cv_results_)
best_idx = rf_grid.best_index_

plt.figure(figsize=(12, 6))
plt.errorbar(
    range(len(rf_cv_results)),
    rf_cv_results["mean_test_score"],
    yerr=rf_cv_results["std_test_score"],
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
    rf_cv_results.loc[best_idx, "mean_test_score"],
    color="red",
    zorder=5,
    s=80,
    label=f"Best: {rf_cv_results.loc[best_idx, 'mean_test_score']:.4f}"
)
plt.xlabel("Iteration (candidate index, run order)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Random Forest Tuning: Accuracy Across All Iterations")
plt.legend()
plt.grid(True)
plt.savefig("rf_tuning_iterations.png", dpi=300, bbox_inches="tight")
plt.show()

# Same data sorted best-to-worst -- makes it easy to see how many
# candidates were close contenders vs. how quickly accuracy drops off
rf_cv_sorted = rf_cv_results.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(
    range(len(rf_cv_sorted)),
    rf_cv_sorted["mean_test_score"],
    marker="o",
    markersize=3,
    linewidth=1
)
plt.scatter(0, rf_cv_sorted["mean_test_score"].iloc[0], color="red", zorder=5,
            label=f"Best: {rf_cv_sorted['mean_test_score'].iloc[0]:.4f}")
plt.scatter(len(rf_cv_sorted) - 1, rf_cv_sorted["mean_test_score"].iloc[-1], color="gray", zorder=5,
            label=f"Worst: {rf_cv_sorted['mean_test_score'].iloc[-1]:.4f}")
plt.xlabel("Rank (best to worst)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Random Forest Tuning: Iterations Ranked by Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("rf_tuning_ranked.png", dpi=300, bbox_inches="tight")
plt.show()

# Per-hyperparameter view: best accuracy achieved at each individual
# parameter value (other params marginalized out via max)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

panels = [
    ("param_n_estimators", "n_estimators", False),
    ("param_max_depth", "max_depth", False),
    ("param_min_samples_split", "min_samples_split", False),
    ("param_min_samples_leaf", "min_samples_leaf", False),
    ("param_max_features", "max_features", True),
]

for ax, (col, label, categorical) in zip(axes.flat, panels):
    grouped = rf_cv_results.dropna(subset=[col]).groupby(col)["mean_test_score"].max()
    if categorical:
        ax.bar(grouped.index.astype(str), grouped.values)
    else:
        # max_depth includes None -- plot it as a categorical label too
        try:
            x_vals = grouped.index.astype(float)
            ax.plot(x_vals, grouped.values, marker="o")
        except (TypeError, ValueError):
            ax.bar(grouped.index.astype(str), grouped.values)
    ax.set_title(f"RF: {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Best Accuracy")
    ax.grid(True)

# Last subplot unused (5 params, 6 slots) -- hide it
axes.flat[-1].axis("off")

plt.tight_layout()
plt.savefig("rf_tuning_per_parameter.png", dpi=300, bbox_inches="tight")
plt.show()

rf_train_prob = rf.predict_proba(X_train)

rf_test_prob = rf.predict_proba(X_test)

# ==========================
# Train SVM (vanilla, no tuning)
# ==========================
# SVC(probability=True) is deprecated (removed once scikit-learn hits
# 1.11) since its internal Platt scaling reuses the same folds as the
# decision function. CalibratedClassifierCV(ensemble=False) is the
# forward-compatible replacement: it fits the SVM on part of the training
# data and calibrates probabilities on a held-out fold. Still "vanilla"
# in the sense that C/gamma are left at their defaults -- no tuning.

svm = CalibratedClassifierCV(
    estimator=SVC(kernel="rbf", random_state=42),
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
print(" Hybrid RF (tuned) + SVM (vanilla) + LR Results")
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
    "Random Forest Learning Curve (Tuned, base learner only)"
)

plot_learning_curve(
    svm,
    X_train,
    y_train,
    "Support Vector Machine Learning Curve (Default, base learner only)"
)

# ==========================
# Hybrid Model Learning Curve (Final LR Result)
# ==========================
# The two curves above only show each BASE learner in isolation --
# neither one goes through the stacking process, so neither reflects the
# final LR output. sklearn's learning_curve() can't do that automatically
# here since the stacking is hand-rolled rather than a Pipeline/
# StackingClassifier, so this rebuilds the whole pipeline at each
# training size: fit RF (tuned hyperparameters) + SVM (vanilla) on the
# subsample, build meta-features, fit LR, then score the LR's own
# predictions -- i.e. the actual final hybrid result -- on both the
# training subsample and a held-out validation fold.

from sklearn.model_selection import StratifiedKFold

def hybrid_learning_curve(X, y, rf_best_params, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, random_state=42):
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

            rf_m = RandomForestClassifier(**rf_best_params, random_state=random_state)
            # cv=3 here (vs. 5 elsewhere) since the smallest training_sizes
            # subsample can be as few as 20 rows -- keeps each calibration
            # fold large enough to be meaningful
            svm_m = CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", random_state=random_state),
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
    rf_grid.best_params_
)

plt.figure(figsize=(8, 6))
plt.plot(hybrid_sizes, hybrid_train_acc, marker="o", label="Training Accuracy (Final Hybrid)")
plt.plot(hybrid_sizes, hybrid_val_acc, marker="s", label="Validation Accuracy (Final Hybrid)")
plt.title("Hybrid Model Learning Curve (RF + SVM + LR, Final Result)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("hybrid_learning_curve.png", dpi=300, bbox_inches="twght")
plt.show()