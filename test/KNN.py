import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, '..', 'build')
sys.path.insert(0, BUILD_DIR)

import knn_cpp

DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'winequality-red.csv')

IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'images', 'knn')
os.makedirs(IMAGES_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip() 
print(df.columns.tolist())
X = df.drop('quality', axis=1).values.astype(float)
y = df['quality'].values.astype(int)

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_mat = np.ascontiguousarray(X_train, dtype=np.float64)
y_train_vec = np.ascontiguousarray(y_train, dtype=np.float64)
X_test_mat = np.ascontiguousarray(X_test, dtype=np.float64)
y_test_vec = np.ascontiguousarray(y_test, dtype=np.float64)

# C++ KNN
model_cpp = knn_cpp.KNN(k=5, knn_type=knn_cpp.KNNType.Classification, metric=knn_cpp.DistanceMetric.EUCLIDEAN)
model_cpp.fit(X_train_mat, y_train_vec)
y_pred_cpp = model_cpp.predict(X_test_mat)

# sklearn KNN
model_sk = KNeighborsClassifier(n_neighbors=5)
model_sk.fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)

acc_cpp = accuracy_score(y_test, y_pred_cpp)
acc_sk = accuracy_score(y_test, y_pred_sk)
print(f"C++ KNN Accuracy: {acc_cpp:.4f}")
print(f"scikit-learn KNN Accuracy: {acc_sk:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_cpp, label="C++ Predicted", alpha=0.6, marker='o', color='tab:blue', s=60)
plt.scatter([v+0.15 for v in y_test], y_pred_sk, label="sklearn Predicted", alpha=0.6, marker='s', color='tab:orange', s=60)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("KNN Predictions vs Actual (Wine Quality)")
plt.legend()
plt.grid(True, linestyle=':')
plt.savefig(os.path.join(IMAGES_DIR, "pred_vs_actual_comparison.png"))

# Accuracy vs k plot
k_values = list(range(1, 21))
accs_cpp = []
accs_sk = []
for k in k_values:
    model_cpp = knn_cpp.KNN(k=k, knn_type=knn_cpp.KNNType.Classification, metric=knn_cpp.DistanceMetric.EUCLIDEAN)
    model_cpp.fit(X_train_mat, y_train_vec)
    y_pred_cpp = model_cpp.predict(X_test_mat)
    accs_cpp.append(accuracy_score(y_test, y_pred_cpp))

    model_sk = KNeighborsClassifier(n_neighbors=k)
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    accs_sk.append(accuracy_score(y_test, y_pred_sk))

plt.figure()
plt.plot(k_values, accs_cpp, label="C++ KNN", marker='o', linestyle='-', color='tab:blue', markersize=7, linewidth=2)
plt.plot([k+0.15 for k in k_values], accs_sk, label="sklearn KNN", marker='s', linestyle='--', color='tab:orange', markersize=7, linewidth=2)
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN Accuracy vs k")
plt.legend()
plt.grid(True, linestyle=':')
plt.savefig(os.path.join(IMAGES_DIR, "accuracy_vs_k.png"))
