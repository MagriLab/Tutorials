import numpy as np
from typing import Optional

# implementation based on Backpropagation algorithms and RC in RNNs for
# the forecasting of complex spatiotemporal dynamics by Vlachas (2020)


def nrmse(pred: np.ndarray, df_test: np.ndarray, n_length: Optional[int] = None) -> float:
    """
    Compute the normalized root mean square error (NRMSE) between the network prediction and the ground truth.

    Arguments:
        pred (np.ndarray) : network prediction
        df_test (np.ndarray): ground truth values, e.g. from training data
        n_length (int, optional): length of the input sequence. If not specified, the length of the shorter input will be used

    Returns:
        The NRMSE value as a float.
    """
    if n_length == None:
        n_length = min(len(pred), df_test.shape[1])
    std = np.std(df_test[:, :n_length])
    diff = pred[:n_length, :] - df_test[:, :+n_length].T
    return np.sqrt(np.mean(diff**2 / std))


def vpt(pred: np.ndarray, df_test: np.ndarray, threshold: float) -> int:
    """Calculate the "valid prediction time" (VPT) of a given prediction.
        The VPT is defined as the maximum number of consecutive time steps
        for which the NRMSE of the prediction is below a given threshold.
    Args:
        pred (np.ndarray): network prediction
        df_test (np.ndarray): reference data
        threshold (float):  NRMSE threshold

    Returns:
        int: NRMSE index
    """
    for i in range(1, len(pred)):
        nrmse_i = nrmse(pred, df_test, n_length=i)
        if nrmse_i > threshold:
            return i - 1
    return len(pred)


def nrmse_array(pred: np.ndarray, df_test: np.ndarray) -> np.array:
    """Calculate the normalized root mean square error (NRMSE) of a given prediction.
        Return NRMSE over time
    Args:
        pred (np.ndarray): network prediction
        df_test (np.ndarray): reference data

    Returns:
        np.array: NRMSE index
    """
    nrmse_over_time = [nrmse(pred, df_test, n_length=i) for i in range(1, len(pred))]
    return nrmse_over_time
