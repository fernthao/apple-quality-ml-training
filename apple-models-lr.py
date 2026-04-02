import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

path = './Data/train.csv'
data = pd.read_csv(path).to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# # See if classes are balanced
# print(np.unique(y, return_counts=True))
# # Got (array([0., 1.]), array([1584, 1616])) so pretty balanced -> using accuracy as a metrics is fine.

# n_samples, n_features = X.shape
# n_classes = len(np.unique(y))

# # Looking at different solver
# print(n_classes) # 2 -> only two class so bilinear (which can only handle binary classification) will work
# print(n_samples, n_features * n_classes) # 3200, 14 so newton-cholesky might be a good choice, since it works well for dataset where n_samples >> n_features * n_classes


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Logistic Regression ──────────────────────────────────────────────────────
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
# GridSearchCV accepts a list of dicts, each is searched independently, so we can
# assign solver-specific hyperparameters without triggering invalid combinations.
lr_params = [
    # lbfgs: L2 only
    {
        'clf__solver': ['lbfgs'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    # liblinear: coordinate descent; L1 and L2; suited for small/sparse data.
    # Binary-only -> matches our 2-class problem.
    {
        'clf__solver': ['liblinear'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    # newton-cg: full Newton with conjugate gradient; L2 only; more precise than
    # lbfgs but slower per iteration on larger feature sets.
    {
        'clf__solver': ['newton-cg'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    # newton-cholesky: Cholesky-factored Newton; L2 only; efficient when
    # n_samples >> n_features * n_classes (3200 >> 14 here) because the Hessian
    # is small enough to invert exactly. Binary-only — matches our problem.
    {
        'clf__solver': ['newton-cholesky'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    # sag: stochastic average gradient; L2 only; scales to large n_samples.
    # Requires feature scaling — already handled by the pipeline's StandardScaler.
    {
        'clf__solver': ['sag'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    # saga: extension of sag; the most flexible solver — supports L1, L2, ElasticNet,
    # and no regularization. Split into separate dicts per penalty to avoid invalid
    # combinations (l1_ratio only applies to elasticnet; C is ignored when penalty=None).
    {
        'clf__solver': ['saga'],
        'clf__penalty': ['elasticnet'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__l1_ratio': [0, 0.5, 1],  # 0 → pure L2, 0.5 → equal blend, 1 → pure L1
    },
    {
        'clf__solver': ['saga'],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
    },
    {
        'clf__solver': ['saga'],
        'clf__penalty': [None],  # no regularization; C and l1_ratio are irrelevant
    },
]
lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=cv, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X, y)
print(f"Logistic Regression  — best params: {lr_grid.best_params_}")
print(f"                       CV accuracy : {lr_grid.best_score_:.4f}\n")


# ── First run results (from apple-models.py notes) ────────────────────────────
#
# Data Preprocessing
#   Scaling:    StandardScaler applied to all 7 features (Size, Weight,
#               Sweetness, Crunchiness, Juiciness, Ripeness, Acidity). Required for both LR and SVM to
#               prevent features with larger ranges from dominating.
#   Validation: 5-fold stratified cross-validation (StratifiedKFold,
#               random_state=42) to preserve class balance across folds.
#   Dimensionality reduction: None. Since we only have 7 features and 2 output classes,
#                             adding PCA/LDA is likely to hurt performance by discarding variance that's discriminative
#
# Logistic Regression
#   Hyperparameter      Values searched          Best
#   C (regularization)  0.01, 0.1, 1, 10, 100    1
#   solver              lbfgs, liblinear         lbfgs
#   CV Accuracy: 0.7412
#
# ── Second run results ─────────────────────────────────────────────────────────
# Increase number of folds k = 10, add l1 and l2 and use other solvers for LR
# Logistic Regression  — best params: {'clf__C': 0.1, 'clf__l1_ratio': 1, 'clf__penalty': 'elasticnet', 'clf__solver': 'saga'}
#                        CV accuracy : 0.7425
