# ==========================
# Load Dataset
# ==========================

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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