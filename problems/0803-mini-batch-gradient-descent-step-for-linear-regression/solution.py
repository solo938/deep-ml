import numpy as np

def mini_batch_gd_step(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    batch_indices: list,
    lr: float
) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """

    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    m = len(batch_indices)

    predictions = X_batch @ weights + bias
    error = predictions - y_batch

    weight_gradient = (2 / m) * (X_batch.T @ error)
    bias_gradient = (2 / m) * np.sum(error)

    new_weights = weights - lr * weight_gradient
    new_bias = bias - lr * bias_gradient

    return np.concatenate([new_weights, [new_bias]])
    
