import pandas as pd
import numpy as np
from sklearn.svm import SVC, NuSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

path = './Data/train.csv'
data = pd.read_csv(path).to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# # See if classes are balanced
# print(np.unique(y, return_counts=True))
# # Got (array([0., 1.]), array([1584, 1616])) so pretty balanced -> using accuracy as a metrics is fine.

# # Looking at different solver
# print(n_classes) # 2 -> only two class so bilinear (which can only handle binary classification) will work
# print(n_samples, n_features * n_classes) # 3200, 14 so newton-cholesky might be a good choice, since it works well for dataset where n_samples >> n_features * n_classes


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── First run results ─────────────────────────────────────────────────────────
# RBF and linear kernels, coarse C grid
# SVM                  — best params: {'clf__C': 100, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9081
#
# svm_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', SVC(random_state=42)),
# ])
# svm_params = [
#     {'clf__kernel': ['rbf'], 'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 'auto']},
#     {'clf__kernel': ['linear'], 'clf__C': [0.1, 1, 10, 100]},
# ]
# svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
# svm_grid.fit(X, y)
# print(f"SVM — best params: {svm_grid.best_params_}")
# print(f"      CV accuracy : {svm_grid.best_score_:.4f}\n")


# ── Second run results ────────────────────────────────────────────────────────
# 5-fold, RBF C in [10,20,...,100]
# SVM                  — best params: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9081


# ── Third run results ─────────────────────────────────────────────────────────
# Refine C: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# SVM                  — best params: {'clf__C': 70, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9094


# ── Fourth run results ────────────────────────────────────────────────────────
# Tighter C: [60, 65, 70, 75] — 70 still best, 0.9094


# ── Fifth run results ─────────────────────────────────────────────────────────
# Polynomial kernel: C in [10,50,100], degree in [2,3], gamma in ['scale','auto']
# SVM                  — best params: {'clf__C': 100, 'clf__degree': 3, 'clf__gamma': 'scale', 'clf__kernel': 'poly'}
#                        CV accuracy : 0.8244
# Lower than RBF


# ── Sixth run results ─────────────────────────────────────────────────────────
# Hand-tuned gamma: C in [50,60,70,80,90,100], gamma in [0.001,0.01,0.1,0.5,1,'scale']
# scale still performs best, 0.9094


# ── Seventh run results ───────────────────────────────────────────────────────
# PolynomialFeatures(degree=2) before SVM, RBF, C in [0.1,1,10,70,100,1000], gamma in ['scale','auto']
# SVM                  — best params: {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9062
# Polynomial features hurt slightly — best stays: C=70, gamma='scale', kernel='rbf' at 0.9094
#
# svm_pipeline = Pipeline([
#     # ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('scaler', StandardScaler()),
#     ('clf', SVC(random_state=42)),
# ])
# svm_params = [
#     {
#         'clf__kernel': ['rbf'],
#         'clf__C': [70],
#         'clf__gamma': ['scale'],
#     },
# ]
# svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
# svm_grid.fit(X, y)
# print(f"SVM                  — best params: {svm_grid.best_params_}")
# print(f"                       CV accuracy : {svm_grid.best_score_:.4f}\n")


# ── Eighth run: fine gamma around 'scale' value + NuSVC ──────────────────────
# gamma='scale' = 1/7 ≈ 0.143 after StandardScaler. Searched neighbourhood around it.
# SVM (fine gamma)     — best params: {'clf__C': 70, 'clf__gamma': 0.16, 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9109
# NuSVC                — best params: {'clf__nu': 0.2}
#                        CV accuracy : 0.9091  (no gain over SVC)
#
# svm_pipeline8 = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', SVC(random_state=42)),
# ])
# svm_params8 = {
#     'clf__kernel': ['rbf'],
#     'clf__C': [70],
#     'clf__gamma': [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
# }
# svm_grid8 = GridSearchCV(svm_pipeline8, svm_params8, cv=cv, scoring='accuracy', n_jobs=-1)
# svm_grid8.fit(X, y)
# print(f"SVM (fine gamma)     — best params: {svm_grid8.best_params_}")
# print(f"                       CV accuracy : {svm_grid8.best_score_:.4f}\n")
#
# nu_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', NuSVC(kernel='rbf', gamma='scale', random_state=42)),
# ])
# nu_params = {'clf__nu': [0.1, 0.2, 0.3, 0.4, 0.5]}
# nu_grid = GridSearchCV(nu_pipeline, nu_params, cv=cv, scoring='accuracy', n_jobs=-1)
# nu_grid.fit(X, y)
# print(f"NuSVC                — best params: {nu_grid.best_params_}")
# print(f"                       CV accuracy : {nu_grid.best_score_:.4f}\n")


# ── Ninth run: tighter gamma around 0.16 ─────────────────────────────────────
# Confirmed 0.16 is the true peak — 0.15 and 0.17 were both available and lost.
# SVM (tighter gamma)  — best params: {'clf__C': 70, 'clf__gamma': 0.16, 'clf__kernel': 'rbf'}
#                        CV accuracy : 0.9109
#
# svm_pipeline9 = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', SVC(random_state=42)),
# ])
# svm_params9 = {
#     'clf__kernel': ['rbf'],
#     'clf__C': [70],
#     'clf__gamma': [0.14, 0.15, 0.16, 0.17, 0.18],
# }
# svm_grid9 = GridSearchCV(svm_pipeline9, svm_params9, cv=cv, scoring='accuracy', n_jobs=-1)
# svm_grid9.fit(X, y)
# print(f"SVM (tighter gamma)  — best params: {svm_grid9.best_params_}")
# print(f"                       CV accuracy : {svm_grid9.best_score_:.4f}\n")

# ── Conclusion ────────────────────────────────────────────────────────────────
# Final best: SVC(kernel='rbf', C=70, gamma=0.16) — CV accuracy 0.9109
