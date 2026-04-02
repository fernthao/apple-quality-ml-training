import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load data ────────────────────────────────────────────────────────────────
train_data = pd.read_csv('./Data/train.csv').to_numpy()
X_train = train_data[:, 0:-1]
y_train = train_data[:, -1]

# ── Model ────────────────────────────────────────────────────────────────────
# Best configuration found via GridSearchCV over 9 tuning runs on train.csv:
#   kernel=rbf, C=70, gamma=0.16 — 5-fold stratified CV accuracy: 0.9109
# Hyperparameters are fixed here; no tuning is performed at inference time.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=70, gamma=0.16, random_state=42)),
])

# ── Train and evaluate ───────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

import os
if os.path.exists('./Data/test.csv'):
    test_data = pd.read_csv('./Data/test.csv').to_numpy()
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
else:
    accuracy = pipeline.score(X_train, y_train)
    print(f"[test.csv not found] Train Accuracy: {accuracy * 100:.2f}%")
