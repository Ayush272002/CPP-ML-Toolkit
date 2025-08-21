import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.datasets import load_breast_cancer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, '..', 'build')
sys.path.insert(0, BUILD_DIR)

import logistic_regression_cpp  

IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'images', 'logistic_regression')
os.makedirs(IMAGES_DIR, exist_ok=True)

# breast cancer dataset
data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(float)

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_mat = np.ascontiguousarray(X, dtype=np.float64)
y_vec = np.ascontiguousarray(y, dtype=np.float64)

reg_type = getattr(logistic_regression_cpp.RegularizationType, "None")
model = logistic_regression_cpp.LogisticRegression(
    fit_intercept=True,
    lr=0.01,
    epochs=100,
    reg_lambda=0.0,
    reg_type=reg_type
)
model.fit(X_mat, y_vec, verbose=True)
y_pred_cpp = model.predict(X_mat)
y_prob_cpp = model.predict_prob(X_mat)

# scikit-learn Logistic Regression
sk_model = SklearnLogisticRegression(fit_intercept=True, max_iter=1000)
sk_model.fit(X_mat, y_vec)
y_pred_sk = sk_model.predict(X_mat)
y_prob_sk = sk_model.predict_proba(X_mat)[:, 1]

acc_cpp = accuracy_score(y_vec, y_pred_cpp)
acc_sk = accuracy_score(y_vec, y_pred_sk)
auc_cpp = roc_auc_score(y_vec, y_prob_cpp)
auc_sk = roc_auc_score(y_vec, y_prob_sk)

print(f"C++ LogisticRegression Accuracy: {acc_cpp:.4f}, AUC: {auc_cpp:.4f}")
print(f"scikit-learn LogisticRegression Accuracy: {acc_sk:.4f}, AUC: {auc_sk:.4f}")

# Loss curve plot
plt.figure()
plt.plot(model.get_loss_history())
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training Loss Curve")
plt.savefig(os.path.join(IMAGES_DIR, "loss_curve.png"))

plt.figure()
plt.scatter(y_vec, y_prob_cpp, label="C++ Predicted", alpha=0.6)
plt.scatter(y_vec, y_prob_sk, label="sklearn Predicted", alpha=0.6)
plt.xlabel("Actual Class")
plt.ylabel("Predicted Probability")
plt.title("Predictions vs Actual (C++ vs scikit-learn)")
plt.legend()
plt.savefig(os.path.join(IMAGES_DIR, "pred_vs_actual_comparison.png"))