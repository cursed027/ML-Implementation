import numpy as np

class CustomLinearRegression:
    def fit(self, X, y, lr=0.01, epochs=1000):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        self.losses = []  # Track MSE loss
        for epoch in range(epochs):
            y_pred = X.dot(self.weights)
            error = y_pred - y
            mse = np.mean(error ** 2)
            self.losses.append(mse)

            grad = (1 / n_samples) * X.T.dot(error)
            self.weights -= lr * grad


    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X.dot(self.weights)
