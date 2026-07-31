# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest-Support Vector Machine Model
#
# Dataset: Heart Disease Dataset (Comprehensive) --
# Kaggle: sid321axn/heart-statlog-cleveland-hungary-final
# Merges Statlog + Cleveland + Hungary (+ Switzerland/VA via Cleveland/
# Hungary sources), 1190 raw rows, 11 features + binary target.
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

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_predict

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

df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

print(df.shape)

# Known data quality issue: this dataset merges Statlog + Cleveland +
# Hungary, and the Statlog portion re-includes many patients already
# present in the other two sources. ~272 exact duplicate rows are known
# to exist -- left in, the same patient can land in both train and test
# (leakage) and gets double-weighted in training either way.
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {rows_before - len(df)} duplicate rows ({len(df)} remaining)")

print("\nMissing Values (raw):")
print(df.isnull().sum())

# Second data quality issue, more subtle: missing readings in the source
# data were filled with 0 instead of left blank, so isnull().sum() above
# looks perfectly clean even though it isn't. A resting BP or cholesterol
# reading of 0 is not physiologically possible for a living patient, so
# treat 0 as missing for these two columns and let the median imputer
# handle it properly, rather than the model learning "0 mg/dl cholesterol"
# as a real value.
print("\nRows with cholesterol == 0:", (df["cholesterol"] == 0).sum())
print("Rows with resting bp s == 0:", (df["resting bp s"] == 0).sum())

df["cholesterol"] = df["cholesterol"].replace(0, np.nan)
df["resting bp s"] = df["resting bp s"].replace(0, np.nan)

print("\nMissing Values (after flagging 0s as NaN):")
print(df.isnull().sum())

# ==========================
# Separate Features and Target
# ==========================

# "target" here is already binary (0 = no disease, 1 = disease present) --
# no binarizing needed, unlike the UCI "num" version of this dataset.
X = df.drop("target", axis=1)
y = df["target"]

print("\nTarget Distribution:")
print(y.value_counts())

# ==========================
# Detect Numerical and Categorical Columns
# ==========================
# Unlike the previous dataset, everything here is already integer/float
# encoded rather than text labels -- but chest pain type/resting ecg/
# ST slope are still NOMINAL categories under the hood (their integer
# codes don't represent a meaningful order), so they still need one-hot
# encoding to avoid the model treating e.g. chest pain type 4 as "more"
# of something than type 1.
#
# sex/fasting blood sugar/exercise angina are already clean 0/1 binary
# flags -- one-hot encoding a column that's already two values would just
# reproduce it, so those are left out of both lists and passed through
# untouched.
#
# NOTE: this CSV uses spaces in its header row rather than underscores
# (e.g. "chest pain type", not "chest_pain_type") -- column names below
# match the file exactly, spaces included.

categorical_columns = [
    "chest pain type",  # 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic
    "resting ecg",       # 0=normal, 1=ST-T abnormality, 2=LV hypertrophy
    "ST slope"           # 0=unknown, 1=upsloping, 2=flat, 3=downsloping
]

numerical_columns = [
    "age",
    "resting bp s",     # resting blood pressure
    "cholesterol",
    "max heart rate",
    "oldpeak"            # ST depression induced by exercise
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
# Only cholesterol/resting bp s have missing values here (the 0s flagged
# above); chest pain type/resting ecg/ST slope have none in this dataset,
# so cat_imputer below is a no-op on this data but kept for robustness
# in case that ever changes.

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
# chest pain type/resting ecg/ST slope are int-typed but nominal, not
# ordinal -- pd.get_dummies works the same on int columns as on strings
# when explicitly named in "columns", so this collapses each into
# drop_first=True indicator columns as normal.

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
# Reusing numerical_columns here (rather than a separately hardcoded list)
# so the scaled columns can't silently drift out of sync with what was
# imputed above.

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
    "n_estimators": [100, 200, 300, 400, 500],
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

rf_table = rf_cv_results[
    [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].sort_values(
    by="rank_test_score"
)

print(rf_table.to_string(index=False))

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

svm_test_prob = svm.predict_proba(X_test)

# ==========================
# Create Meta Features (out-of-fold, not in-sample)
# ==========================
# Using rf.predict_proba(X_train) / svm.predict_proba(X_train) directly
# would feed the meta-model probabilities from models scored on the same
# rows they were just trained on -- RF in particular (min_samples_leaf=1
# candidates included) can get close to memorizing its own training rows,
# so those in-sample probabilities are overconfident in a way that
# doesn't reflect real generalization. The LR meta-model would then learn
# to trust that false confidence, inflating the hybrid model's apparent
# training accuracy beyond what either base learner actually earns.
#
# cross_val_predict fixes this the standard way for stacking: each row's
# meta-feature comes from a model that never saw that row during fitting
# (it was in the held-out fold when that prediction was made), so the
# meta-model trains on the same kind of out-of-sample signal it'll see
# at test time. rf/svm are then refit on the FULL X_train afterward
# (already true for rf via GridSearchCV's refit=True; explicit for svm)
# so the test-time predictions still use every available training row.

rf_oof = cross_val_predict(
    rf,
    X_train,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1
)

svm_oof = cross_val_predict(
    svm,
    X_train,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1
)

meta_train = np.hstack((
    rf_oof,
    svm_oof
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
plt.savefig("hybrid_learning_curve.png", dpi=300, bbox_inches="tight")
plt.show()