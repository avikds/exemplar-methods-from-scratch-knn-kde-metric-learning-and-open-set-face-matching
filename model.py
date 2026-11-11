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

# Step 2 - knn_proba
def knn_proba(X_train, y_train, X_query, K, n_classes=None, M=None):
    """Return the (n_query, n_classes) array of KNN class probabilities.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training feature vectors.

    y_train : ndarray, shape (n_train,)
        Integer training labels starting from 0.

    X_query : ndarray, shape (n_query, n_features)
        Query feature vectors.

    K : int
        Number of nearest neighbours.

    n_classes : int, optional
        Total number of classes. If None, inferred as max(y_train) + 1.

    M : ndarray, shape (n_features, n_features), optional
        Mahalanobis metric matrix. If None, Euclidean distance is used.

    Returns
    -------
    proba : ndarray, shape (n_query, n_classes)
        Probability distribution over classes for every query point.
    """

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_query = np.asarray(X_query, dtype=float)

    # Validate input dimensions.
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")

    if X_query.ndim != 2:
        raise ValueError("X_query must be a 2D array.")

    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")

    n_train = X_train.shape[0]
    n_query = X_query.shape[0]

    # Ensure that every training point has a corresponding label.
    if len(y_train) != n_train:
        raise ValueError("X_train and y_train must have the same number of samples.")

    # Validate K.
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise ValueError("K must be an integer.")

    if K < 1 or K > n_train:
        raise ValueError("K must be at least 1 and cannot exceed the number of training points.")

    # Labels must be non-negative integers starting from 0.
    if not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must contain integer class labels.")

    if np.any(y_train < 0):
        raise ValueError("Class labels must be non-negative.")

    # Determine the number of classes.
    if n_classes is None:
        n_classes = int(y_train.max()) + 1

    else:
        if (
            isinstance(n_classes, (bool, np.bool_))
            or not isinstance(n_classes, (int, np.integer))
            or n_classes < 1
        ):
            raise ValueError("n_classes must be a positive integer.")

        n_classes = int(n_classes)

    if np.any(y_train >= n_classes):
        raise ValueError("Every training label must be smaller than n_classes.")

    # Compute pairwise squared distances using the function from Step 1.
    distances = pairwise_sq_dists(X_query, X_train, M=M)

    # Find the K nearest training points.
    # Stable sorting ensures that ties are resolved by the smaller
    # training-point index.
    nearest_indices = np.argsort(distances, axis=1, kind="stable")[:, :K]

    # Retrieve the class labels of the nearest neighbours.
    nearest_labels = y_train[nearest_indices]

    # Count the occurrences of each class among the K neighbours.
    proba = np.zeros((n_query, n_classes), dtype=float)

    # Use np.add.at to correctly accumulate repeated class labels.
    query_indices = np.repeat(np.arange(n_query), K)

    np.add.at(
        proba,
        (query_indices, nearest_labels.ravel()),
        1.0
    )

    # Convert class counts into probabilities.
    proba /= K

    return proba

# Step 3 - error_vs_k
def error_vs_k(X_train, y_train, X_test, y_test, ks):
    """Return (train_err, test_err), the misclassification rates for each K in ks.

    Training points are classified against the complete training set,
    including themselves, as required for reproducing the book's curves.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training feature vectors.

    y_train : ndarray, shape (n_train,)
        Integer training labels starting from 0.

    X_test : ndarray, shape (n_test, n_features)
        Test feature vectors.

    y_test : ndarray, shape (n_test,)
        Integer test labels.

    ks : iterable
        Values of K (number of nearest neighbours) to evaluate.

    Returns
    -------
    train_err : ndarray, shape (len(ks),)
        Training misclassification rate for each K.

    test_err : ndarray, shape (len(ks),)
        Test misclassification rate for each K.
    """

    # Convert inputs to NumPy arrays.
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)

    # Validate label dimensions.
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1D arrays.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of samples.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of samples.")

    if len(X_train) == 0:
        raise ValueError("The training set must not be empty.")

    if len(X_test) == 0:
        raise ValueError("The test set must not be empty.")

    # Validate class labels.
    if not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must contain integer class labels.")

    if not np.issubdtype(y_test.dtype, np.integer):
        raise ValueError("y_test must contain integer class labels.")

    if np.any(y_train < 0) or np.any(y_test < 0):
        raise ValueError("Class labels must be non-negative.")

    # Convert ks to a list so that it can be iterated over consistently.
    ks = list(ks)

    # Include every class label appearing in either dataset.
    # This also handles test classes absent from the training set.
    n_classes = int(max(y_train.max(), y_test.max())) + 1

    # Store training and test errors for each K.
    train_err = []
    test_err = []

    for K in ks:

        # Classify training points against the FULL training set.
        # Each training point is included among its own neighbours.
        train_proba = knn_proba(
            X_train,
            y_train,
            X_train,
            K,
            n_classes=n_classes
        )

        # Classify test points using the training set.
        test_proba = knn_proba(
            X_train,
            y_train,
            X_test,
            K,
            n_classes=n_classes
        )

        # Predict the class with the highest KNN probability.
        # np.argmax selects the smallest class index in case of a tie.
        y_train_pred = np.argmax(train_proba, axis=1)
        y_test_pred = np.argmax(test_proba, axis=1)

        # Calculate misclassification rates.
        train_error = np.mean(y_train_pred != y_train)
        test_error = np.mean(y_test_pred != y_test)

        train_err.append(train_error)
        test_err.append(test_error)

    # Return float NumPy arrays.
    return np.asarray(train_err, dtype=float), np.asarray(test_err, dtype=float)

