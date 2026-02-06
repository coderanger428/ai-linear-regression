import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/housing.csv")

# Inspect first rows
print(data.head())
print(data.info())

# 1. Select numeric features only for simplicity
X = data.select_dtypes(include=np.number).drop(columns=["price"], errors="ignore").fillna(0).values
y = data["price"].values

# 2. Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# 3. Slit into train/test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# 4. Initialize and train model
model = LinearRegression(learning_rate=0.01, n_epochs=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Test MSE: {mse:.4f}")

# 6. Visualize
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2)
plt.xlabel("True SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Predicted vs Actual SalePrice")
plt.show()
