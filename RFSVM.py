import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

RANDOM_STATE = 42

# %% Config -- edit these for your file, then just run the script
DATA_PATH = "heart_disease_dataset.csv"
TARGET_COLUMN = "Heart Disease"     # e.g. "target" or "Heart Disease"
OUTPUT_DIR = "./outputs"

# ==========================
# Step 1: Load Dataset
# ==========================
df = pd.read_csv(DATA_PATH)

# ==========================
# Remove Duplicate Records
# ==========================
print(f"Dataset shape before removing duplicates: {df.shape}")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after removing duplicates: {df.shape}\n")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# # If the target is text (e.g. "Yes"/"No"), map it to 0/1
if not pd.api.types.is_numeric_dtype(y):
    y = y.astype(str).str.strip().str.lower().map({
        "yes": 1, "no": 0, "presence": 1, "absence": 0,
        "true": 1, "false": 0, "disease": 1, "healthy": 0
    })
# # If the target is multi-class severity (0 = none, 1-4 = disease),
# # as in the original Statlog/Cleveland encoding, binarize it
elif y.nunique() > 2:
    y = (y > 0).astype(int)

# print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
# print(f"Class balance:\n{y.value_counts()}\n")

# ==========================
# Step 2: Pre-processing
# ==========================

# Detect numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:", list(numerical_columns))
print("Categorical Columns:", list(categorical_columns))

# Handle missing values
if len(numerical_columns) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

if len(categorical_columns) > 0:
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

# One-hot encoding for categorical attributes (e.g. chest pain type)
if len(categorical_columns) > 0:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Re-detect numeric columns after one-hot encoding, for scaling
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns

# Train-test split (80:20), done BEFORE scaling to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# # Feature scaling -- fit on train, apply to both
# scaler = StandardScaler()
# X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
# X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# ==========================
# Step 3: Train Random Forest
# ==========================
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)
rf_test_prob = rf.predict_proba(X_test)

# ==========================
# Step 4: Train SVM (RBF kernel)
# ==========================
svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

svm_train_prob = svm.predict_proba(X_train)
svm_test_prob = svm.predict_proba(X_test)

# ==========================
# Step 5: Meta-Feature Matrix  Z = [P_RF(X), P_SVM(X)]
# ==========================
meta_train = np.hstack((rf_train_prob, svm_train_prob))
meta_test = np.hstack((rf_test_prob, svm_test_prob))

# ==========================
# Step 6: Train Logistic Regression (meta-model)
# ==========================
meta_model = LogisticRegression(random_state=RANDOM_STATE)
meta_model.fit(meta_train, y_train)

# ==========================
# Step 7: Final Prediction on X_test
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

# Also score the two individual base learners, for the paper's
# side-by-side comparison (Fig. 6 / 7)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
rf_proba_pos = rf_test_prob[:, 1]
svm_proba_pos = svm_test_prob[:, 1]

comparison = pd.DataFrame({
    "accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred),
        accuracy,
    ],
    "f1": [
        f1_score(y_test, rf_pred),
        f1_score(y_test, svm_pred),
        f1,
    ],
    "auc": [
        roc_auc_score(y_test, rf_proba_pos),
        roc_auc_score(y_test, svm_proba_pos),
        auc,
    ],
}, index=["RF", "SVM", "Hybrid RF+SVM"]).round(4)

print("\n=== Performance Comparison (paper's Fig. 6) ===")
print(comparison)

# # ==========================
# # Graphs
# # ==========================
# import os
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ----- Confusion Matrix (Final Hybrid Model) -----
# # improved: normalized annotations alongside raw counts
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=meta_model.classes_)

# fig, ax = plt.subplots(figsize=(6, 6))
# disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
# ax.set_title("Confusion Matrix - Hybrid RF+SVM+LR")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/final_confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.show()

# # ----- ROC Curve (Final Hybrid Model) -----
# # improved: RF and SVM curves are plotted alongside the hybrid model so
# # you can see the gain from stacking, not just the final AUC in isolation
# fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, y_prob, pos_label=meta_model.classes_[1])

# plt.figure(figsize=(7, 6))
# plt.plot(fpr_hybrid, tpr_hybrid, color="darkorange", linewidth=2.5,
#          label=f"Hybrid RF+SVM+LR (AUC = {auc:.4f})")
# plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Guess (AUC = 0.50)")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/final_roc_curve.png", dpi=300, bbox_inches="tight")
# plt.show()

# # ----- Learning Curves (Base Learners) -----
# # improved: shaded +/- 1 std band around each mean line, so you can see
# # how noisy each fold's score is, not just the average

# def plot_learning_curve(estimator, X, y, title, filename):
#     train_sizes, train_scores, validation_scores = learning_curve(
#         estimator=estimator,
#         X=X,
#         y=y,
#         cv=5,
#         scoring="accuracy",
#         train_sizes=np.linspace(0.1, 1.0, 10),
#         shuffle=True,
#         random_state=RANDOM_STATE
#     )

#     train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
#     val_mean, val_std = validation_scores.mean(axis=1), validation_scores.std(axis=1)

#     plt.figure(figsize=(8, 6))
#     plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
#     plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
#     plt.plot(train_sizes, val_mean, marker="s", label="Validation Accuracy")
#     plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
#     plt.title(title)
#     plt.xlabel("Training Set Size")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
#     plt.show()


# plot_learning_curve(
#     rf, X_train, y_train,
#     "Random Forest Learning Curve (base learner only)",
#     "rf_learning_curve.png"
# )

# plot_learning_curve(
#     svm, X_train, y_train,
#     "Support Vector Machine Learning Curve (base learner only)",
#     "svm_learning_curve.png"
# )

# # ----- Hybrid Model Learning Curve (Final LR Result) -----
# # "Validation" here is always the SAME held-out X_test/y_test used for
# # the final reported metrics above -- only the TRAINING side is varied.
# # At each training_sizes fraction, a random subsample of X_train is
# # drawn, RF + SVM are fit on it, meta-features are built for both the
# # subsample and X_test, and LR is fit on the subsample's meta-features.
# # Training accuracy is scored on the subsample; test accuracy is scored
# # on the fixed, never-resampled X_test -- so the curve's test-accuracy
# # line should converge toward the same accuracy reported above.
# #
# # Each training size is repeated n_repeats times with a different random
# # subsample and averaged (with a shaded +/- 1 std band, improved from
# # the mean-only version) to smooth out noise at the smallest sizes.

# def hybrid_learning_curve(X_train, y_train, X_test, y_test,
#                            train_sizes=np.linspace(0.1, 1.0, 10), n_repeats=5,
#                            random_state=RANDOM_STATE):
#     X_train = X_train.reset_index(drop=True)
#     y_train = y_train.reset_index(drop=True)

#     sizes_out = []
#     train_acc_mean, train_acc_std = [], []
#     val_acc_mean, val_acc_std = [], []

#     for frac in train_sizes:
#         n_sub = max(int(len(X_train) * frac), 20)

#         rep_train_acc, rep_val_acc = [], []

#         for rep in range(n_repeats):
#             rng = np.random.RandomState(random_state + rep)
#             sample_idx = rng.choice(X_train.index, size=n_sub, replace=False)

#             X_sub, y_sub = X_train.loc[sample_idx], y_train.loc[sample_idx]
#             if y_sub.nunique() < 2:
#                 continue

#             rf_m = RandomForestClassifier(random_state=random_state)
#             svm_m = SVC(kernel="rbf", probability=True, random_state=random_state)
#             rf_m.fit(X_sub, y_sub)
#             svm_m.fit(X_sub, y_sub)

#             meta_sub = np.hstack((rf_m.predict_proba(X_sub), svm_m.predict_proba(X_sub)))
#             meta_val = np.hstack((rf_m.predict_proba(X_test), svm_m.predict_proba(X_test)))

#             lr_m = LogisticRegression(random_state=random_state, max_iter=1000)
#             lr_m.fit(meta_sub, y_sub)

#             rep_train_acc.append(accuracy_score(y_sub, lr_m.predict(meta_sub)))
#             rep_val_acc.append(accuracy_score(y_test, lr_m.predict(meta_val)))

#         sizes_out.append(n_sub)
#         train_acc_mean.append(np.mean(rep_train_acc))
#         train_acc_std.append(np.std(rep_train_acc))
#         val_acc_mean.append(np.mean(rep_val_acc))
#         val_acc_std.append(np.std(rep_val_acc))

#     return (np.array(sizes_out), np.array(train_acc_mean), np.array(train_acc_std),
#             np.array(val_acc_mean), np.array(val_acc_std))


# (hybrid_sizes, hybrid_train_mean, hybrid_train_std,
#  hybrid_val_mean, hybrid_val_std) = hybrid_learning_curve(X_train, y_train, X_test, y_test)

# plt.figure(figsize=(8, 6))
# plt.plot(hybrid_sizes, hybrid_train_mean, marker="o", label="Training Accuracy (Final Hybrid)")
# plt.fill_between(hybrid_sizes, hybrid_train_mean - hybrid_train_std,
#                   hybrid_train_mean + hybrid_train_std, alpha=0.15)
# plt.plot(hybrid_sizes, hybrid_val_mean, marker="s",
#           label="Test Accuracy (Final Hybrid, fixed held-out set)")
# plt.fill_between(hybrid_sizes, hybrid_val_mean - hybrid_val_std,
#                   hybrid_val_mean + hybrid_val_std, alpha=0.15)
# plt.title("Hybrid Model Learning Curve (RF + SVM + LR)")
# plt.xlabel("Training Set Size")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/hybrid_learning_curve.png", dpi=300, bbox_inches="tight")
# plt.show()

# comparison.to_csv(f"{OUTPUT_DIR}/performance_comparison.csv")
# print(f"\nSaved figures and results table to: {OUTPUT_DIR}")

# ==========================
# Save Hybrid Model
# ==========================
import joblib
import json
import os

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the three components of the hybrid model
joblib.dump(rf, f"{OUTPUT_DIR}/rf_model.pkl")
joblib.dump(svm, f"{OUTPUT_DIR}/svm_model.pkl")
joblib.dump(meta_model, f"{OUTPUT_DIR}/meta_model.pkl")

print("\nModels Saved Successfully")
print(f"RF model      : {OUTPUT_DIR}/rf_model.pkl")
print(f"SVM model     : {OUTPUT_DIR}/svm_model.pkl")
print(f"Meta-model    : {OUTPUT_DIR}/meta_model.pkl")


# ==========================
# Save Model Performance
# ==========================

metrics_path = f"{OUTPUT_DIR}/metrics.json"

existing = {}

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        existing = json.load(f)

existing["RF + SVM + LR"] = {
    "r2": None,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(auc)
}

with open(metrics_path, "w") as f:
    json.dump(existing, f, indent=2)

print(f"Metrics saved to: {metrics_path}")