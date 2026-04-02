import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, StratifiedKFold

# ── Data ─────────────────────────────────────────────────────────────────────
data = pd.read_csv('./Data/train.csv').to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# ── Model (best configuration) ───────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=70, gamma=0.16, random_state=42)),
])

# ── Learning curve ───────────────────────────────────────────────────────────
# train_sizes: fractions of the training set from 5% up to 100%
# cv: same 5-fold stratified CV used during tuning for consistency
cv = StratifiedKFold(n_splits=32, shuffle=True, random_state=42)
train_sizes, train_scores, cv_scores = learning_curve(
    pipeline, X, y,
    train_sizes=np.linspace(0.05, 1.0, 20),
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
cv_mean    = cv_scores.mean(axis=1)
cv_std     = cv_scores.std(axis=1)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(train_sizes, train_mean, label='Training accuracy', color='steelblue', marker='o', markersize=4)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='steelblue')

ax.plot(train_sizes, cv_mean, label='CV accuracy (5-fold)', color='tomato', marker='o', markersize=4)
ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color='tomato')

ax.axhline(y=cv_mean[-1], color='tomato', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(train_sizes[0], cv_mean[-1] + 0.002, f'{cv_mean[-1]:.4f}', color='tomato', fontsize=8)

ax.set_xlabel('Training set size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve — SVM (RBF, C=70, γ=0.16)')
ax.legend(loc='lower right')
ax.set_ylim(0.7, 1.02)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('./report/learning_curve.pdf', bbox_inches='tight')
plt.savefig('./report/learning_curve.png', dpi=150, bbox_inches='tight')
print("Saved: report/learning_curve.pdf and report/learning_curve.png")
