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
# Train Random Forest (Default Parameters)
# ==========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

# ==========================
# SVM Hyperparameter Tuning
# ==========================

svm_parameters = {

    "estimator__C": [0.1, 1, 10, 100],

    "estimator__gamma": [
        "scale",
        0.1,
        0.01,
        0.001
    ]

}

svm_grid = GridSearchCV(

    estimator=CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            random_state=42
        ),
        method="sigmoid",
        cv=3,
        ensemble=False
    ),

    param_grid=svm_parameters,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

svm_grid.fit(X_train, y_train)

print("\nBest SVM Parameters")
print(svm_grid.best_params_)
print(f"SVM Best CV Accuracy: {svm_grid.best_score_:.4f}")

svm = svm_grid.best_estimator_

svm_cv_results = pd.DataFrame(svm_grid.cv_results_)

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
plt.savefig("hybrid_learning_curve.png", dpi=300, bbox_inches="tight")
plt.show()