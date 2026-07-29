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

df = pd.read_csv("heart_disease_dataset.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.columns)