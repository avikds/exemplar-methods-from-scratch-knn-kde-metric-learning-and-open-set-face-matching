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

# Step 4 - one_nn_vs_bayes
import math


def one_nn_vs_bayes(delta, D, Ns, n_test=2000, seed=0):
    """Measure the 1-NN test error and compare it with the Bayes error.

    The data distribution is:

        y ~ Bernoulli(0.5)

        x | y=1 ~ N((+delta, 0, ..., 0), I)
        x | y=0 ~ N((-delta, 0, ..., 0), I)

    Only the first coordinate contains class information.

    Parameters
    ----------
    delta : float
        Separation between the class means along the first coordinate.

    D : int
        Number of feature dimensions.

    Ns : iterable of int
        Training sample sizes to evaluate.

    n_test : int, default=2000
        Number of test observations.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:

        'bayes'     : Exact Bayes classification error.
        'bound'     : Cover-Hart bound, 2*R*(1-R).
        'asymptote' : Exact large-sample 1-NN error, computed numerically.
        'errors'    : Measured 1-NN test error for each training size.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------

    if not np.isscalar(delta) or not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a finite non-negative scalar.")

    if (
        isinstance(D, (bool, np.bool_))
        or not isinstance(D, (int, np.integer))
        or D < 1
    ):
        raise ValueError("D must be a positive integer.")

    if (
        isinstance(n_test, (bool, np.bool_))
        or not isinstance(n_test, (int, np.integer))
        or n_test < 1
    ):
        raise ValueError("n_test must be a positive integer.")

    Ns = list(Ns)

    for N in Ns:
        if (
            isinstance(N, (bool, np.bool_))
            or not isinstance(N, (int, np.integer))
            or N < 1
        ):
            raise ValueError("Every training size in Ns must be a positive integer.")

    delta = float(delta)
    D = int(D)
    n_test = int(n_test)

    # ---------------------------------------------------------
    # 2. Compute the exact Bayes error
    # ---------------------------------------------------------

    # R* = Phi(-delta)
    # Use the complementary error function for numerical accuracy.

    bayes = 0.5 * math.erfc(delta / math.sqrt(2.0))

    # Cover-Hart asymptotic upper bound.
    bound = 2.0 * bayes * (1.0 - bayes)

    # ---------------------------------------------------------
    # 3. Compute the exact large-sample 1-NN asymptote
    # ---------------------------------------------------------

    # The posterior probability of class 1 is:
    #
    # eta(x) = P(y=1 | x)
    #        = 1 / (1 + exp(-2 * delta * x))
    #
    # Using tanh gives a numerically stable equivalent:
    #
    # eta(x) = 0.5 * (1 + tanh(delta * x))

    # The marginal density of the informative coordinate is:
    #
    # p(x) = 0.5 * N(x; +delta, 1)
    #      + 0.5 * N(x; -delta, 1)

    # Construct the required uniform integration grid.
    grid = np.linspace(-12.0, 12.0, 24001)

    # Posterior probability of class 1.
    eta = 0.5 * (1.0 + np.tanh(delta * grid))

    # Standard normal density normalization constant.
    normal_const = 1.0 / math.sqrt(2.0 * math.pi)

    # Conditional densities for the two classes.
    density_class_1 = (
        normal_const
        * np.exp(-0.5 * (grid - delta) ** 2)
    )

    density_class_0 = (
        normal_const
        * np.exp(-0.5 * (grid + delta) ** 2)
    )

    # Marginal density of the first coordinate.
    p_x = 0.5 * (density_class_1 + density_class_0)

    # Asymptotic 1-NN error:
    #
    # R_1NN = integral 2 * eta(x) * (1 - eta(x)) * p(x) dx

    integrand = 2.0 * eta * (1.0 - eta) * p_x

    # Composite trapezoidal integration.
    dx = grid[1] - grid[0]

    asymptote = dx * np.sum(
        0.5 * (integrand[:-1] + integrand[1:])
    )

    # ---------------------------------------------------------
    # 4. Initialize the random number generator
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 5. Define the data-generation function
    # ---------------------------------------------------------

    def draw_sample(n):
        """Generate n observations from the specified distribution."""

        # Draw labels first, exactly as specified.
        y = rng.integers(0, 2, size=n)

        # Generate independent standard normal coordinates.
        X = rng.normal(size=(n, D))

        # Shift the first coordinate according to the label.
        #
        # y=0 -> mean -delta
        # y=1 -> mean +delta
        X[:, 0] += np.where(y == 1, delta, -delta)

        return X, y

    # ---------------------------------------------------------
    # 6. Draw the test set FIRST
    # ---------------------------------------------------------

    X_test, y_test = draw_sample(n_test)

    # Store measured 1-NN test errors.
    errors = []

    # ---------------------------------------------------------
    # 7. Evaluate each training sample size
    # ---------------------------------------------------------

    for N in Ns:

        # Draw one fresh training set for this sample size.
        X_train, y_train = draw_sample(int(N))

        # Store predictions for all test observations.
        y_pred = np.empty(n_test, dtype=int)

        # Process test observations in batches to avoid allocating
        # an unnecessarily large distance matrix.
        batch_size = min(256, n_test)

        for start in range(0, n_test, batch_size):

            end = min(start + batch_size, n_test)

            # Compute pairwise squared Euclidean distances.
            # The Mahalanobis matrix is omitted, so M=None.
            distances = pairwise_sq_dists(
                X_test[start:end],
                X_train
            )

            # Find the index of the single nearest training point.
            # np.argmin returns the first index in case of a tie.
            nearest_indices = np.argmin(distances, axis=1)

            # Assign the label of the nearest neighbour.
            y_pred[start:end] = y_train[nearest_indices]

        # Measure test misclassification rate.
        test_error = np.mean(y_pred != y_test)

        errors.append(float(test_error))

    # ---------------------------------------------------------
    # 8. Return all theoretical and measured results
    # ---------------------------------------------------------

    return {
        "bayes": float(bayes),
        "bound": float(bound),
        "asymptote": float(asymptote),
        "errors": errors
    }

# Step 5 - cube_edge_length
def cube_edge_length(p, D, N=100000, seed=0):
    """Return (theory, empirical), the edge length of a centred cube
    holding a fraction p of uniformly distributed data.

    Parameters
    ----------
    p : float
        Fraction of data points to be contained in the cube.

    D : int
        Number of dimensions.

    N : int, default=100000
        Number of uniformly distributed samples.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    theory : float
        Theoretical cube edge length, p**(1/D).

    empirical : float
        Empirical cube edge length estimated using simulated data.
    """

    # Validate the requested fraction.
    if not np.isscalar(p) or not np.isfinite(p) or not (0 < p <= 1):
        raise ValueError("p must be a finite number in the interval (0, 1].")

    # Validate dimensionality.
    if (
        isinstance(D, (bool, np.bool_))
        or not isinstance(D, (int, np.integer))
        or D < 1
    ):
        raise ValueError("D must be a positive integer.")

    # Validate sample size.
    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N < 1
    ):
        raise ValueError("N must be a positive integer.")

    # ---------------------------------------------------------
    # 1. Compute the theoretical cube edge length.
    # ---------------------------------------------------------

    # For uniformly distributed points in the D-dimensional
    # unit cube, the probability of falling inside a centred
    # cube of edge length e is:
    #
    # p = e**D
    #
    # Therefore:
    #
    # e = p**(1/D)

    theory = p ** (1.0 / D)

    # ---------------------------------------------------------
    # 2. Generate uniformly distributed random samples.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(N, D))

    # ---------------------------------------------------------
    # 3. Compute the L-infinity (sup-norm) distance from
    #    every point to the centre of the unit cube.
    # ---------------------------------------------------------

    # The centre of the unit cube is (0.5, 0.5, ..., 0.5).
    #
    # The sup-norm distance is the maximum absolute coordinate
    # difference from the centre.

    distances = np.max(np.abs(X - 0.5), axis=1)

    # ---------------------------------------------------------
    # 4. Sort the distances in ascending order.
    # ---------------------------------------------------------

    distances = np.sort(distances)

    # ---------------------------------------------------------
    # 5. Determine the number of points to include.
    # ---------------------------------------------------------

    # Use the 1-based kth position specified in the instructions.
    k = max(1, round(p * N))

    # ---------------------------------------------------------
    # 6. Calculate the empirical cube edge length.
    # ---------------------------------------------------------

    # The distance is measured from the centre to a cube face.
    # Therefore, the complete edge length is twice this distance.

    empirical = 2.0 * distances[k - 1]

    return float(theory), float(empirical)

