"""
CSDS 340 Case Study 1 — Apple Quality Classification
Group 18 | Spring 2026

Trains the selected SVM classifier (RBF kernel, C=70, gamma=0.16) on
train.csv and evaluates accuracy on test.csv.  Hyperparameters were
selected via GridSearchCV with 5-fold stratified cross-validation
(see experiment report for details).
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load data ────────────────────────────────────────────────────────────────
train_data = pd.read_csv('./Data/train.csv').to_numpy()
X_train = train_data[:, 0:-1]  # columns 1-7: Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity
y_train = train_data[:, -1]    # column 8: Quality (0 = bad, 1 = good)

# ── Model ────────────────────────────────────────────────────────────────────
# Best configuration found via GridSearchCV over 9 tuning runs on train.csv:
#   kernel=rbf, C=70, gamma=0.16 — 5-fold stratified CV accuracy: 0.9109
# Hyperparameters are fixed here; no tuning is performed at inference time.
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # zero-mean, unit-variance; fit on train only
    ('clf', SVC(kernel='rbf', C=70, gamma=0.16, random_state=42)),
])

# ── Train and evaluate ───────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

if os.path.exists('./Data/test.csv'):
    test_data = pd.read_csv('./Data/test.csv').to_numpy()
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
else:
    # Fallback for local development when test.csv is not available
    accuracy = pipeline.score(X_train, y_train)
    print(f"[test.csv not found] Train Accuracy: {accuracy * 100:.2f}%")
