import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

path = './Data/train.csv'
data = pd.read_csv(path).to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# # See if classes are balanced
# print(np.unique(y, return_counts=True))
# # Got (array([0., 1.]), array([1584, 1616])) so pretty balanced -> using accuracy as a metrics is fine.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Decision Tree ───────────────────────────────────────────────────────────
# StandardScaler matches LR/SVM pipelines and assignment wording (same preprocessing
# on train/test later). Splits are order-equivalent to raw features per dimension.

# ── First run results ─────────────────────────────────────────────────────────
# Grid:
#   max_depth:          [None, 4, 6, 8, 10, 12, 15, 20]
#   min_samples_leaf:   [1, 2, 4, 8, 16]
#   min_samples_split:  [2, 5, 10, 20]
#   max_features:       [None, 'sqrt', 'log2']
#   criterion:          ['gini', 'entropy']
#
# Decision Tree       — best params: {'clf__criterion': 'gini', 'clf__max_depth': 12,
#                                     'clf__max_features': None, 'clf__min_samples_leaf': 2,
#                                     'clf__min_samples_split': 10}
#                        CV accuracy : 0.8134
#
# dt_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', DecisionTreeClassifier(random_state=42)),
# ])
# dt_params = {
#     'clf__max_depth': [None, 4, 6, 8, 10, 12, 15, 20],
#     'clf__min_samples_leaf': [1, 2, 4, 8, 16],
#     'clf__min_samples_split': [2, 5, 10, 20],
#     'clf__max_features': [None, 'sqrt', 'log2'],
#     'clf__criterion': ['gini', 'entropy'],
# }
# dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=cv, scoring='accuracy', n_jobs=-1)
# dt_grid.fit(X, y)
# print(f"Decision Tree       — best params: {dt_grid.best_params_}")
# print(f"                       CV accuracy : {dt_grid.best_score_:.4f}\n")


# ── Second run: cost-complexity pruning (ccp_alpha) ───────────────────────────
# Decision Tree (ccp) — best params: {'clf__ccp_alpha': 0.0010078812426093384}
#                        CV accuracy : 0.8184
#
# _base_tree = DecisionTreeClassifier(
#     criterion='gini', max_depth=12, max_features=None,
#     min_samples_leaf=2, min_samples_split=10, random_state=42,
# )
# from sklearn.preprocessing import StandardScaler as _SS
# _X_scaled = _SS().fit_transform(X)
# _path = _base_tree.cost_complexity_pruning_path(_X_scaled, y)
# _alphas = _path.ccp_alphas[:-1]
# _alpha_candidates = list(np.unique(np.concatenate([
#     [0.0],
#     np.geomspace(_alphas[_alphas > 0].min(), _alphas.max(), num=20),
# ])))
# dt_pipeline2 = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', DecisionTreeClassifier(
#         criterion='gini', max_depth=12, max_features=None,
#         min_samples_leaf=2, min_samples_split=10, random_state=42,
#     )),
# ])
# dt_params2 = {'clf__ccp_alpha': _alpha_candidates}
# dt_grid2 = GridSearchCV(dt_pipeline2, dt_params2, cv=cv, scoring='accuracy', n_jobs=-1)
# dt_grid2.fit(X, y)
# print(f"Decision Tree (ccp) — best params: {dt_grid2.best_params_}")
# print(f"                       CV accuracy : {dt_grid2.best_score_:.4f}\n")


# ── Third run: PCA before decision tree ───────────────────────────────────────
# Fix the best params from runs 1 & 2, insert PCA between scaler and clf, and
# search over n_components (1..7, where 7 = all features = no dimensionality
# reduction). PCA is fit only on the training fold inside each CV split.
dt_pipeline3 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('clf', DecisionTreeClassifier(
        criterion='gini',
        max_depth=12,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=10,
        ccp_alpha=0.0010078812426093384,
        random_state=42,
    )),
])
dt_params3 = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7],
}
dt_grid3 = GridSearchCV(dt_pipeline3, dt_params3, cv=cv, scoring='accuracy', n_jobs=-1)
dt_grid3.fit(X, y)
print(f"Decision Tree (PCA) — best params: {dt_grid3.best_params_}")
print(f"                       CV accuracy : {dt_grid3.best_score_:.4f}\n")
