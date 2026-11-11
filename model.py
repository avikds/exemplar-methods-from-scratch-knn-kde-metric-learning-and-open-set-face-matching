"""
Exemplar Methods from Scratch: kNN, KDE, Metric Learning and Open-Set Face Matching

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - pairwise_sq_dists
import numpy as np

def pairwise_sq_dists(X, Y=None, M=None):
    """Compute pairwise squared Mahalanobis distances.

    Parameters
    ----------
    X : array-like, shape (n_samples_X, n_features)
        First set of data points.

    Y : array-like, shape (n_samples_Y, n_features), optional
        Second set of data points. If None, Y = X.

    M : array-like, shape (n_features, n_features), optional
        Symmetric positive-semidefinite metric matrix.
        If None, the identity matrix is used.

    Returns
    -------
    distances : ndarray, shape (n_samples_X, n_samples_Y)
        Pairwise squared Mahalanobis distances, where

        distances[i, j] =
        (X[i] - Y[j]).T @ M @ (X[i] - Y[j])
    """

    # Convert inputs to floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    # If Y is not provided, compute distances within X.
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array.")

    # Both datasets must have the same feature dimension.
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    # Case 1: Ordinary squared Euclidean distance.
    if M is None:
        x_norm_sq = np.einsum("ij,ij->i", X, X)
        y_norm_sq = np.einsum("ij,ij->i", Y, Y)

        cross = X @ Y.T

    # Case 2: Squared Mahalanobis distance.
    else:
        M = np.asarray(M, dtype=float)

        D = X.shape[1]

        if M.ndim != 2 or M.shape != (D, D):
            raise ValueError(f"M must have shape ({D}, {D}).")

        # The metric matrix must be symmetric.
        if not np.allclose(M, M.T, rtol=1e-7, atol=1e-10):
            raise ValueError("M must be symmetric.")

        # Remove insignificant floating-point asymmetry.
        M = 0.5 * (M + M.T)

        # Compute X @ M and Y @ M.
        XM = X @ M
        YM = Y @ M

        # Compute individual quadratic terms.
        x_norm_sq = np.einsum("ij,ij->i", XM, X)
        y_norm_sq = np.einsum("ij,ij->i", YM, Y)

        # Compute the cross-product term.
        cross = XM @ Y.T

    # Use the vectorized squared-distance identity:
    #
    # d(i,j)^2 = x_i.T M x_i + y_j.T M y_j
    #            - 2 * x_i.T M y_j
    #
    # For M=None, this reduces to squared Euclidean distance.

    distances = (
        x_norm_sq[:, None]
        + y_norm_sq[None, :]
        - 2.0 * cross
    )

    # Numerical round-off can produce tiny negative values.
    # Squared distances must always be non-negative.
    np.maximum(distances, 0.0, out=distances)

    return distances

