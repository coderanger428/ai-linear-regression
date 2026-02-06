import numpy as np

class LinearRegression:
  def __init__(self, learning_rate=0.01, n_epochs=1000):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0.0

    for epoch in range(self.n_epochs):
      y_pred = X @ self.weights + self.bias
      loss = np.mean((y - y_pred) ** 2)

      grad_w = -(2 / n_samples) * (X.T @ (y - y_pred))
      grad_b = -(2 / n_samples) * np.sum(y - y_pred)

      self.weights -= self.learning_rate * grad_w
      self.bias -= self.learning_rate * grad_b

      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
  
  def predict(self, X):
    return X @ self.weights + self.bias
  