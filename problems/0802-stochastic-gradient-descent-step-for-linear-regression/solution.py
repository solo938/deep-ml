import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    for i in range(n_iter):
        sample_index = i % len(X)
        x_i = X[sample_index]
        y_i = y[sample_index]
        prediction =  x_i @ weights
        error = prediction - y_i
        gradient = 2 * error * x_i
        weights = weights - learning_rate * gradient
    
    return weights.tolist()

    