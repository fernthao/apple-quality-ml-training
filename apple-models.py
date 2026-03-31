import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

path = './Data/train.csv'
data = pd.read_csv(path).to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Logistic Regression ──────────────────────────────────────────────────────
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
lr_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__solver': ['lbfgs', 'liblinear'],
}
lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=cv, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X, y)
print(f"Logistic Regression  — best params: {lr_grid.best_params_}")
print(f"                       CV accuracy : {lr_grid.best_score_:.4f}\n")

# ── SVM ──────────────────────────────────────────────────────────────────────
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(random_state=42)),
])
svm_params = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__kernel': ['rbf', 'linear'],
    'clf__gamma': ['scale', 'auto'],
}
svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X, y)
print(f"SVM                  — best params: {svm_grid.best_params_}")
print(f"                       CV accuracy : {svm_grid.best_score_:.4f}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
results = {
    'Logistic Regression': (lr_grid.best_score_, lr_grid.best_estimator_),
    'SVM':                 (svm_grid.best_score_, svm_grid.best_estimator_),
}

best_name = max(results, key=lambda k: results[k][0])
best_score, best_model = results[best_name]
print(f"Best model: {best_name}  (CV accuracy = {best_score:.4f})")