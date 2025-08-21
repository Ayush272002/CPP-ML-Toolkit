import os
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, '..', 'build')
sys.path.insert(0, BUILD_DIR)

import linear_regression_cpp 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'housing.csv')
IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'images', 'linear_regression')
os.makedirs(IMAGES_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# Convert yes/no to 1/0
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
	df[col] = df[col].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

X = df.drop("price", axis=1).values.astype(float)
y = df["price"].values.astype(float)

X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

X_mat = np.ascontiguousarray(X, dtype=np.float64)
y_vec = np.ascontiguousarray(y, dtype=np.float64)

model = linear_regression_cpp.LinearRegression(True, 0.01, 1000)
model.fit(X_mat, y_vec, verbose=True)
y_pred_cpp = model.predict(X_mat)

sk_model = SklearnLinearRegression(fit_intercept=True)
sk_model.fit(X_mat, y_vec)
y_pred_sk = sk_model.predict(X_mat)

mse_cpp = mean_squared_error(y_vec, y_pred_cpp)
mse_sk = mean_squared_error(y_vec, y_pred_sk)
print(f"C++ LinearRegression MSE: {mse_cpp}")
print(f"scikit-learn LinearRegression MSE: {mse_sk}")

plt.figure()
plt.plot(model.get_loss_history())
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.savefig(os.path.join(IMAGES_DIR, "loss_curve.png"))

# Predictions vs Actual (C++ vs scikit-learn)
plt.figure()
plt.scatter(y, y_pred_cpp, label="C++ Predicted", alpha=0.6)
plt.scatter(y, y_pred_sk, label="sklearn Predicted", alpha=0.6)
plt.xlabel("Actual Price (normalized)")
plt.ylabel("Predicted Price (normalized)")
plt.title("Predictions vs Actual (C++ vs scikit-learn)")
plt.legend()
plt.savefig(os.path.join(IMAGES_DIR, "pred_vs_actual_comparison.png"))
