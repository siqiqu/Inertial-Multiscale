import numpy as np


def soft_shrinkage(X, theta):
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    S_shrinked = np.maximum(S - theta, 0)
    return U @ np.diag(S_shrinked) @ Vh


def psnr(X_ref, X_est):
    mse = np.mean((X_ref - X_est) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def isnr(X_ref, X_corrupt, X_est):
    mse_before = np.mean((X_ref - X_corrupt) ** 2)
    mse_after = np.mean((X_ref - X_est) ** 2)
    if mse_after == 0:
        return float('inf')
    return 10 * np.log10(mse_before / mse_after)