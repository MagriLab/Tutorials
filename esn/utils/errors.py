import numpy as np


def L2(y, y_pred, axis=None):
    # equivalent to np.sqrt(np.sum((y-y_pred)**2, axis = axis))
    # frobenius norm is default
    return np.linalg.norm(y - y_pred, axis=axis)


def rel_L2(y, y_pred, axis=None):
    # Calculate the relative L2 error.
    return L2(y, y_pred, axis=axis) / np.linalg.norm(y, axis=axis)


def mse(y, y_pred, axis=None):
    # Calculate the mean squared error.
    return np.mean((y - y_pred) ** 2, axis=axis)


def rmse(y, y_pred, axis=None):
    # Calculate the root mean squared error.
    return np.sqrt(mse(y, y_pred, axis))

def nrmse(y, y_pred, axis=None, normalize_by="rms"):
    # Calculate the normalized root mean squared error.
    if normalize_by == "rms":
        norm = np.sqrt(np.mean(y, axis=0) ** 2)
    elif normalize_by == "maxmin":
        norm = np.max(y, axis=0) - np.min(y, axis=0)
    elif normalize_by == "std":
        norm = np.std(y, axis=0)**2

    return np.mean(rmse(y, y_pred, axis=0) / norm, axis=axis)
