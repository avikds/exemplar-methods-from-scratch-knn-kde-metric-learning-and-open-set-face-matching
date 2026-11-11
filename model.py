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

# Step 6 - noise_dimension_curve
def noise_dimension_curve(
    X_train, y_train, X_test, y_test,
    n_noise, K=5, scale=1.0, seed=0
):
    """Return KNN test accuracy after appending different numbers of noise features.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Original training features.

    y_train : ndarray, shape (n_train,)
        Training class labels.

    X_test : ndarray, shape (n_test, n_features)
        Original test features.

    y_test : ndarray, shape (n_test,)
        Test class labels.

    n_noise : iterable of int
        Numbers of noise dimensions to append.

    K : int, default=5
        Number of nearest neighbours used by KNN.

    scale : float, default=1.0
        Standard deviation multiplier for the Gaussian noise.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    accuracies : list of float
        KNN test accuracy for each number of added noise dimensions.
    """

    # Convert inputs to NumPy arrays.
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)

    # ---------------------------------------------------------
    # 1. Validate input dimensions and sample counts.
    # ---------------------------------------------------------

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1D arrays.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of samples.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of samples.")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and test sets must not be empty.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")

    # ---------------------------------------------------------
    # 2. Validate class labels.
    # ---------------------------------------------------------

    if not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must contain integer class labels.")

    if not np.issubdtype(y_test.dtype, np.integer):
        raise ValueError("y_test must contain integer class labels.")

    if np.any(y_train < 0) or np.any(y_test < 0):
        raise ValueError("Class labels must be non-negative.")

    # Include labels appearing in either dataset.
    n_classes = int(max(y_train.max(), y_test.max())) + 1

    # ---------------------------------------------------------
    # 3. Validate noise dimensions and scale.
    # ---------------------------------------------------------

    n_noise = list(n_noise)

    for d in n_noise:
        if (
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, np.integer))
            or d < 0
        ):
            raise ValueError(
                "Every entry in n_noise must be a non-negative integer."
            )

    if (
        not np.isscalar(scale)
        or not np.isfinite(scale)
        or scale < 0
    ):
        raise ValueError("scale must be a finite non-negative number.")

    # ---------------------------------------------------------
    # 4. Initialize one random number generator for the
    #    entire function call.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    # Store the accuracy corresponding to each noise dimension.
    accuracies = []

    # ---------------------------------------------------------
    # 5. Evaluate KNN for each requested number of noise features.
    # ---------------------------------------------------------

    for d in n_noise:

        # Draw training noise FIRST.
        # Each noise feature follows N(0, scale^2).
        train_noise = scale * rng.normal(
            size=(len(X_train), d)
        )

        # Draw test noise SECOND using the same generator.
        test_noise = scale * rng.normal(
            size=(len(X_test), d)
        )

        # Append the noise features to the original datasets.
        X_train_augmented = np.hstack(
            (X_train, train_noise)
        )

        X_test_augmented = np.hstack(
            (X_test, test_noise)
        )

        # Obtain KNN class probabilities.
        # M=None means ordinary squared Euclidean distance.
        probabilities = knn_proba(
            X_train_augmented,
            y_train,
            X_test_augmented,
            K,
            n_classes=n_classes
        )

        # Predict the class with the largest probability.
        # np.argmax selects the smallest class index in a tie.
        y_pred = np.argmax(probabilities, axis=1)

        # Calculate classification accuracy.
        accuracy = np.mean(y_pred == y_test)

        accuracies.append(float(accuracy))

    return accuracies

# Step 7 - kde_density
def density_kernel(u, name="gaussian"):
    """Evaluate a normalized 1-D kernel of Table 16.1 element-wise.

    Parameters
    ----------
    u : array-like
        Input values at which to evaluate the kernel.

    name : str, default="gaussian"
        Kernel name. Supported kernels:
        'gaussian', 'boxcar', 'epanechnikov', 'tricube'.

    Returns
    -------
    ndarray
        Kernel values with the same shape as u.
    """

    u = np.asarray(u, dtype=float)

    # Gaussian kernel:
    # K(u) = (2*pi)^(-1/2) * exp(-u^2 / 2)
    if name == "gaussian":
        return np.exp(-0.5 * u**2) / math.sqrt(2.0 * math.pi)

    # Boxcar (uniform) kernel:
    # K(u) = 1/2, when |u| <= 1; otherwise 0.
    elif name == "boxcar":
        return 0.5 * (np.abs(u) <= 1.0)

    # Epanechnikov kernel:
    # K(u) = 3/4 * (1 - u^2), when |u| <= 1; otherwise 0.
    elif name == "epanechnikov":
        return 0.75 * (1.0 - u**2) * (np.abs(u) <= 1.0)

    # Tricube kernel:
    # K(u) = 70/81 * (1 - |u|^3)^3, when |u| <= 1.
    elif name == "tricube":
        return (
            (70.0 / 81.0)
            * (1.0 - np.abs(u)**3)**3
            * (np.abs(u) <= 1.0)
        )

    else:
        raise ValueError(
            "Unknown kernel name. Choose from "
            "'gaussian', 'boxcar', 'epanechnikov', or 'tricube'."
        )


def kde_density(x_train, x_query, h, kernel="gaussian"):
    """Return the Parzen window density estimate (eq 16.33).

    Parameters
    ----------
    x_train : array-like
        One-dimensional training observations.

    x_query : array-like
        Query locations at which to evaluate the density.

    h : float
        Positive bandwidth parameter.

    kernel : str, default="gaussian"
        Kernel name supported by density_kernel().

    Returns
    -------
    ndarray
        KDE density estimates at each query location.
    """

    # Convert inputs to NumPy arrays and flatten them.
    x_train = np.asarray(x_train, dtype=float).ravel()
    x_query = np.asarray(x_query, dtype=float).ravel()

    # Validate the training data.
    if x_train.size == 0:
        raise ValueError("x_train must contain at least one observation.")

    if not np.all(np.isfinite(x_train)):
        raise ValueError("x_train must contain finite values.")

    if not np.all(np.isfinite(x_query)):
        raise ValueError("x_query must contain finite values.")

    # Validate the bandwidth.
    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("Bandwidth h must be a positive finite number.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("Bandwidth h must be a positive finite number.")

    # Validate the kernel name.
    # This also raises ValueError for unsupported kernel names.
    if kernel not in (
        "gaussian",
        "boxcar",
        "epanechnikov",
        "tricube"
    ):
        raise ValueError(
            "Unknown kernel name. Choose from "
            "'gaussian', 'boxcar', 'epanechnikov', or 'tricube'."
        )

    # Number of training observations.
    N = len(x_train)

    # Compute scaled distances between every query point and
    # every training observation.
    #
    # Shape: (n_query, N)
    u = (x_query[:, None] - x_train[None, :]) / h

    # Evaluate the selected kernel element-wise.
    kernel_values = density_kernel(u, name=kernel)

    # Apply the Parzen window density estimator:
    #
    # p(x | D) = (1/N) * sum_i [ (1/h) * K((x - x_i)/h) ]
    #
    # Equivalently:
    # p(x | D) = mean(K((x - x_i)/h)) / h

    density = np.mean(kernel_values, axis=1) / h

    return density

# Step 8 - silverman_bandwidth
def silverman_bandwidth(X):
    """Return Silverman's bandwidth with the MAD-based robust scale,
    combined over dimensions using a geometric mean.

    Parameters
    ----------
    X : array-like, shape (N,) or (N, D)
        Training observations. Can be one-dimensional or
        multidimensional.

    Returns
    -------
    float
        A single bandwidth obtained by taking the geometric
        mean of the individual dimension-wise bandwidths.
    """

    # Convert the input to a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Accept either a 1-D array or a 2-D array.
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    elif X.ndim != 2:
        raise ValueError("X must be either a 1-D or 2-D array.")

    # Validate the input data.
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one dimension.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    # Number of observations and dimensions.
    N, D = X.shape

    # ---------------------------------------------------------
    # 1. Compute the median of each dimension.
    # ---------------------------------------------------------

    median = np.median(X, axis=0)

    # ---------------------------------------------------------
    # 2. Compute the Median Absolute Deviation (MAD).
    # ---------------------------------------------------------

    # MAD_d = median(|x_d - median(x_d)|)
    mad = np.median(
        np.abs(X - median),
        axis=0
    )

    # ---------------------------------------------------------
    # 3. Convert MAD into a robust estimate of standard deviation.
    # ---------------------------------------------------------

    # sigma_hat_d = 1.4826 * MAD_d
    sigma_hat = 1.4826 * mad

    # A zero MAD means the robust scale is zero in that dimension.
    if np.any(mad == 0):
        raise ValueError(
            "Median absolute deviation must be non-zero in every dimension."
        )

    # ---------------------------------------------------------
    # 4. Compute the Silverman bandwidth for each dimension.
    # ---------------------------------------------------------

    # h_d = sigma_hat_d * (4 / (3*N))^(1/5)
    h_d = sigma_hat * (4.0 / (3.0 * N)) ** (1.0 / 5.0)

    # ---------------------------------------------------------
    # 5. Combine dimension-wise bandwidths using a geometric mean.
    # ---------------------------------------------------------

    # h = (product(h_d))^(1/D)
    #
    # Compute using logarithms for improved numerical stability.
    bandwidth = np.exp(np.mean(np.log(h_d)))

    return float(bandwidth)

# Step 9 - kde_logpdf
def kde_logpdf(X_train, X_query, h, leave_one_out=False):
    """Return the Gaussian KDE log density at each query (eq 16.30, 16.33).

    Parameters
    ----------
    X_train : ndarray, shape (N, D) or (N,)
        Training observations.

    X_query : ndarray, shape (M, D) or (M,)
        Query observations.

    h : float
        Positive Gaussian kernel bandwidth.

    leave_one_out : bool, default=False
        If True, exclude training point i when evaluating query point i.
        X_query must then correspond row-by-row to X_train.

    Returns
    -------
    log_density : ndarray, shape (M,)
        Gaussian KDE log density evaluated at each query point.
    """

    # Convert inputs to floating-point arrays.
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)

    # Treat 1-D inputs as one-dimensional feature vectors.
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    elif X_train.ndim != 2:
        raise ValueError("X_train must be a 1-D or 2-D array.")

    if X_query.ndim == 1:
        X_query = X_query.reshape(-1, 1)
    elif X_query.ndim != 2:
        raise ValueError("X_query must be a 1-D or 2-D array.")

    # Validate the data.
    if X_train.shape[0] == 0:
        raise ValueError("X_train must contain at least one observation.")

    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("X_train and X_query must have the same feature dimension.")

    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train must contain finite values.")

    if not np.all(np.isfinite(X_query)):
        raise ValueError("X_query must contain finite values.")

    # Validate the bandwidth.
    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("h must be a positive finite number.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    # Number of training observations and feature dimensions.
    N, D = X_train.shape
    n_query = X_query.shape[0]

    # Determine the number of training observations contributing
    # to each density estimate.
    if leave_one_out:
        if n_query != N:
            raise ValueError(
                "In leave-one-out mode, X_query must have "
                "the same number of rows as X_train."
            )

        if N < 2:
            raise ValueError(
                "Leave-one-out KDE requires at least two training observations."
            )

        n_components = N - 1

    else:
        n_components = N

    # ---------------------------------------------------------
    # 1. Compute pairwise squared Euclidean distances.
    # ---------------------------------------------------------

    # Reuse the pairwise distance function from Step 1.
    #
    # Shape: (n_query, N)
    sq_dists = pairwise_sq_dists(X_query, X_train)

    # ---------------------------------------------------------
    # 2. Compute Gaussian log-kernel values.
    # ---------------------------------------------------------

    # Gaussian KDE:
    #
    # p(x) = (1 / N) * sum_i [
    #     (1 / (h * sqrt(2*pi))^D)
    #     * exp(-||x - x_i||^2 / (2*h^2))
    # ]
    #
    # Work directly with logarithms to avoid underflow.

    log_kernel = -sq_dists / (2.0 * h**2)

    # ---------------------------------------------------------
    # 3. Apply leave-one-out exclusion when requested.
    # ---------------------------------------------------------

    if leave_one_out:
        # Row i represents query X_train[i].
        # Remove its self-contribution from the density estimate.
        np.fill_diagonal(log_kernel, -np.inf)

    # ---------------------------------------------------------
    # 4. Compute log-sum-exp in a numerically stable way.
    # ---------------------------------------------------------

    # For each query:
    #
    # log(sum_i exp(a_i))
    # = max(a_i) + log(sum_i exp(a_i - max(a_i)))
    #
    # Subtracting the maximum prevents overflow.
    # At least one exponential in each row is exactly 1,
    # preventing underflow of the entire sum.

    max_log_kernel = np.max(log_kernel, axis=1, keepdims=True)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted_sum = np.sum(
            np.exp(log_kernel - max_log_kernel),
            axis=1
        )

        log_sum_exp = (
            max_log_kernel.ravel()
            + np.log(shifted_sum)
        )

    # ---------------------------------------------------------
    # 5. Add the Gaussian normalization constants.
    # ---------------------------------------------------------

    # log p(x) =
    #
    # logsumexp(log_kernel)
    # - log(number of contributing samples)
    # - D*log(h)
    # - (D/2)*log(2*pi)

    log_density = (
        log_sum_exp
        - np.log(n_components)
        - D * np.log(h)
        - 0.5 * D * np.log(2.0 * np.pi)
    )

    return log_density


def loo_bandwidth(X, grid):
    """Return (best_h, curve) using leave-one-out KDE log likelihood.

    Parameters
    ----------
    X : ndarray, shape (N, D) or (N,)
        Training observations.

    grid : array-like
        Candidate positive bandwidth values.

    Returns
    -------
    best_h : float
        Bandwidth with the highest mean leave-one-out log density.

    curve : ndarray
        Mean leave-one-out log density for every bandwidth in grid.
    """

    # Convert the candidate bandwidths into a NumPy array.
    grid = np.asarray(grid, dtype=float)

    # Validate the bandwidth grid.
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array.")

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError(
            "Every bandwidth in grid must be positive and finite."
        )

    # Convert the input observations to an array.
    X = np.asarray(X, dtype=float)

    if X.ndim not in (1, 2):
        raise ValueError("X must be a 1-D or 2-D array.")

    if X.shape[0] < 2:
        raise ValueError(
            "Leave-one-out bandwidth selection requires at least two observations."
        )

    # Store mean leave-one-out log likelihoods.
    curve = np.empty(len(grid), dtype=float)

    # ---------------------------------------------------------
    # Evaluate each candidate bandwidth.
    # ---------------------------------------------------------

    for i, h in enumerate(grid):

        # Evaluate every observation using a KDE that excludes
        # that observation from its own density estimate.
        log_densities = kde_logpdf(
            X,
            X,
            h,
            leave_one_out=True
        )

        # Mean leave-one-out log likelihood.
        curve[i] = np.mean(log_densities)

    # ---------------------------------------------------------
    # Select the bandwidth with the largest LOO log likelihood.
    # ---------------------------------------------------------

    best_index = int(np.argmax(curve))

    best_h = float(grid[best_index])

    return best_h, curve

# Step 10 - balloon_posterior
def balloon_posterior(X_train, y_train, X_query, K, n_classes=None):
    """Return (cond, post): balloon class conditionals (eq 16.34)
    and the posterior they imply through Bayes' rule (eq 16.35).

    Parameters
    ----------
    X_train : ndarray, shape (N, D)
        Training feature vectors.

    y_train : ndarray, shape (N,)
        Integer training class labels starting from 0.

    X_query : ndarray, shape (n_query, D)
        Query feature vectors.

    K : int
        Number of nearest training neighbours.

    n_classes : int, optional
        Number of classes. If None, inferred as max(y_train) + 1.

    Returns
    -------
    cond : ndarray, shape (n_query, n_classes)
        Balloon class-conditional density estimates.

    post : ndarray, shape (n_query, n_classes)
        Posterior probabilities calculated using Bayes' rule.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs and validate dimensions.
    # ---------------------------------------------------------

    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")

    if X_query.ndim != 2:
        raise ValueError("X_query must be a 2D array.")

    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")

    N, D = X_train.shape
    n_query = X_query.shape[0]

    if N == 0 or D == 0:
        raise ValueError("X_train must contain samples and features.")

    if len(y_train) != N:
        raise ValueError("X_train and y_train must have the same number of samples.")

    if X_query.shape[1] != D:
        raise ValueError("X_train and X_query must have the same feature dimension.")

    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_query)):
        raise ValueError("Training and query data must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate K and class labels.
    # ---------------------------------------------------------

    if (
        isinstance(K, (bool, np.bool_))
        or not isinstance(K, (int, np.integer))
        or not 1 <= K <= N
    ):
        raise ValueError("K must be an integer between 1 and N.")

    if not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must contain integer labels.")

    if np.any(y_train < 0):
        raise ValueError("Class labels must be non-negative.")

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
        raise ValueError("Training labels must be smaller than n_classes.")

    # ---------------------------------------------------------
    # 3. Compute pairwise squared Euclidean distances.
    # ---------------------------------------------------------

    sq_dists = pairwise_sq_dists(X_query, X_train)

    # ---------------------------------------------------------
    # 4. Find the K nearest training observations.
    # ---------------------------------------------------------

    # Stable sorting ensures that smaller training indices
    # win when two distances are equal.

    nearest_indices = np.argsort(
        sq_dists,
        axis=1,
        kind="stable"
    )[:, :K]

    nearest_labels = y_train[nearest_indices]

    # ---------------------------------------------------------
    # 5. Count the observations belonging to each class.
    # ---------------------------------------------------------

    # Total number of training observations in each class.
    class_counts = np.bincount(
        y_train,
        minlength=n_classes
    )

    # Number of observations from each class inside
    # the K-nearest-neighbour balloon.
    local_counts = np.zeros(
        (n_query, n_classes),
        dtype=float
    )

    query_indices = np.repeat(
        np.arange(n_query),
        K
    )

    np.add.at(
        local_counts,
        (query_indices, nearest_labels.ravel()),
        1.0
    )

    # ---------------------------------------------------------
    # 6. Calculate the CORRECT K-th nearest-neighbour radius.
    # ---------------------------------------------------------

    # IMPORTANT:
    # The original distance matrix is NOT sorted.
    # Therefore, sq_dists[:, K-1] is NOT the K-th nearest
    # distance.
    #
    # Retrieve the distances corresponding to the already
    # sorted nearest-neighbour indices.

    nearest_sq_dists = np.take_along_axis(
        sq_dists,
        nearest_indices,
        axis=1
    )

    # The last entry is the squared distance to the K-th
    # nearest training observation.
    kth_sq_dist = nearest_sq_dists[:, K - 1]

    radius = np.sqrt(kth_sq_dist)

    # ---------------------------------------------------------
    # 7. Calculate the D-dimensional balloon volume.
    # ---------------------------------------------------------

    # V(x) = pi^(D/2) * r^D / Gamma(D/2 + 1)

    log_unit_ball_volume = (
        0.5 * D * math.log(math.pi)
        - math.lgamma(0.5 * D + 1.0)
    )

    with np.errstate(divide="ignore"):
        log_volume = (
            log_unit_ball_volume
            + D * np.log(radius)
        )

    # Prevent zero-volume problems when a query coincides
    # with a training observation.
    log_volume = np.clip(
        log_volume,
        math.log(1e-300),
        math.log(np.finfo(float).max)
    )

    volume = np.exp(log_volume)

    # ---------------------------------------------------------
    # 8. Compute the balloon class-conditional densities.
    # ---------------------------------------------------------

    # Equation 16.34:
    #
    # p(x | y=c, D) = N_c(x) / (N_c * V(x))

    denominator = (
        class_counts[None, :] * volume[:, None]
    )

    cond = np.zeros(
        (n_query, n_classes),
        dtype=float
    )

    np.divide(
        local_counts,
        denominator,
        out=cond,
        where=denominator > 0
    )

    # ---------------------------------------------------------
    # 9. Calculate the class priors.
    # ---------------------------------------------------------

    # p(y=c) = N_c / N

    priors = class_counts / N

    # ---------------------------------------------------------
    # 10. Calculate posterior probabilities using Bayes' rule.
    # ---------------------------------------------------------

    # Equation 16.35:
    #
    # p(y=c | x, D) =
    #     p(x | y=c, D) * p(y=c)
    #     -----------------------
    #     sum_j p(x | y=j, D) * p(y=j)

    joint = cond * priors[None, :]

    normalizer = np.sum(
        joint,
        axis=1,
        keepdims=True
    )

    post = np.zeros_like(joint)

    np.divide(
        joint,
        normalizer,
        out=post,
        where=normalizer > 0
    )

    return cond, post

# Step 11 - kde_classifier
def kde_classifier(X_train, y_train, X_query, h, n_classes=None):
    """Return the (n_query, n_classes) posterior of a generative
    classifier with one fixed-bandwidth Gaussian KDE per class.

    Parameters
    ----------
    X_train : ndarray, shape (N, D) or (N,)
        Training feature vectors.

    y_train : ndarray, shape (N,)
        Integer class labels starting from 0.

    X_query : ndarray, shape (n_query, D) or (n_query,)
        Query feature vectors.

    h : float
        Positive Gaussian KDE bandwidth.

    n_classes : int, optional
        Number of classes. If None, inferred as max(y_train) + 1.
        Every class must have at least one training observation.

    Returns
    -------
    posterior : ndarray, shape (n_query, n_classes)
        Posterior class probabilities for every query.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train)

    # Treat one-dimensional feature arrays as single-column data.
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    elif X_train.ndim != 2:
        raise ValueError("X_train must be a 1-D or 2-D array.")

    if X_query.ndim == 1:
        X_query = X_query.reshape(-1, 1)
    elif X_query.ndim != 2:
        raise ValueError("X_query must be a 1-D or 2-D array.")

    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")

    # ---------------------------------------------------------
    # 2. Validate input dimensions and values.
    # ---------------------------------------------------------

    N, D = X_train.shape
    n_query = X_query.shape[0]

    if N == 0 or D == 0:
        raise ValueError("X_train must contain observations and features.")

    if len(y_train) != N:
        raise ValueError(
            "X_train and y_train must have the same number of samples."
        )

    if X_query.shape[1] != D:
        raise ValueError(
            "X_train and X_query must have the same number of features."
        )

    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train must contain finite values.")

    if not np.all(np.isfinite(X_query)):
        raise ValueError("X_query must contain finite values.")

    # ---------------------------------------------------------
    # 3. Validate the bandwidth.
    # ---------------------------------------------------------

    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("h must be a positive finite number.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    # ---------------------------------------------------------
    # 4. Validate the training labels.
    # ---------------------------------------------------------

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
        raise ValueError(
            "Every training label must be smaller than n_classes."
        )

    # ---------------------------------------------------------
    # 5. Count the training observations in each class.
    # ---------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=n_classes
    )

    # Every class must appear in the training set.
    if np.any(class_counts == 0):
        raise ValueError(
            "Every class must have at least one training observation."
        )

    # Calculate the class prior probabilities.
    #
    # P(y=c) = N_c / N

    priors = class_counts / N

    # ---------------------------------------------------------
    # 6. Compute the class-conditional Gaussian KDEs.
    # ---------------------------------------------------------

    # Store the joint log probabilities:
    #
    # log P(y=c, x | D)
    #     = log P(x | y=c, D) + log P(y=c)

    log_joint = np.empty(
        (n_query, n_classes),
        dtype=float
    )

    for c in range(n_classes):

        # Select the training observations belonging to class c.
        X_class = X_train[y_train == c]

        # Estimate the class-conditional density using the
        # Gaussian KDE implementation from Step 9.
        log_conditional = kde_logpdf(
            X_class,
            X_query,
            h,
            leave_one_out=False
        )

        # Add the logarithm of the class prior.
        log_joint[:, c] = (
            log_conditional + np.log(priors[c])
        )

    # ---------------------------------------------------------
    # 7. Normalize the posterior probabilities in log space.
    # ---------------------------------------------------------

    # Apply the log-sum-exp normalization:
    #
    # P(y=c | x, D) =
    # exp(log_joint_c) / sum_j exp(log_joint_j)
    #
    # Subtracting the maximum prevents numerical overflow.

    # Treat numerical NaNs from extreme floating-point cases
    # as unavailable log probabilities.
    log_joint = np.where(
        np.isnan(log_joint),
        -np.inf,
        log_joint
    )

    row_max = np.max(
        log_joint,
        axis=1,
        keepdims=True
    )

    posterior = np.empty_like(log_joint)

    # Identify rows with at least one finite log probability.
    valid_rows = np.isfinite(row_max[:, 0])

    if np.any(valid_rows):

        shifted = np.exp(
            log_joint[valid_rows] - row_max[valid_rows]
        )

        normalizer = np.sum(
            shifted,
            axis=1,
            keepdims=True
        )

        posterior[valid_rows] = shifted / normalizer

    # If all class log probabilities are non-finite for an
    # extreme numerical input, return the normalized priors.
    if np.any(~valid_rows):
        posterior[~valid_rows] = priors

    return posterior

# Step 12 - nadaraya_watson
def nadaraya_watson(x_train, y_train, x_query, h, sigma=0.0):
    """Return (mean, var) of Gaussian kernel regression at the queries.

    Parameters
    ----------
    x_train : ndarray, shape (N,) or (N, D)
        Training input observations.

    y_train : ndarray, shape (N,)
        Training target values. Flattened automatically.

    x_query : ndarray, shape (n_query,) or (n_query, D)
        Query input observations.

    h : float
        Positive Gaussian kernel bandwidth.

    sigma : float, default=0.0
        Standard deviation of the observation noise.

    Returns
    -------
    mean : ndarray, shape (n_query,)
        Nadaraya-Watson conditional mean predictions.

    var : ndarray, shape (n_query,)
        Conditional predictive variance estimates.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()

    # Treat 1-D inputs as one-dimensional feature vectors.
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    elif x_train.ndim != 2:
        raise ValueError("x_train must be a 1-D or 2-D array.")

    if x_query.ndim == 1:
        x_query = x_query.reshape(-1, 1)
    elif x_query.ndim != 2:
        raise ValueError("x_query must be a 1-D or 2-D array.")

    # ---------------------------------------------------------
    # 2. Validate input dimensions and values.
    # ---------------------------------------------------------

    N, D = x_train.shape
    n_query = x_query.shape[0]

    if N == 0 or D == 0:
        raise ValueError("x_train must contain observations and features.")

    if len(y_train) != N:
        raise ValueError(
            "x_train and y_train must have the same number of observations."
        )

    if x_query.shape[1] != D:
        raise ValueError(
            "x_train and x_query must have the same number of features."
        )

    if not np.all(np.isfinite(x_train)):
        raise ValueError("x_train must contain finite values.")

    if not np.all(np.isfinite(x_query)):
        raise ValueError("x_query must contain finite values.")

    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train must contain finite values.")

    # ---------------------------------------------------------
    # 3. Validate the bandwidth and noise standard deviation.
    # ---------------------------------------------------------

    try:
        h = float(h)
        sigma = float(sigma)
    except (TypeError, ValueError):
        raise ValueError("h and sigma must be finite scalar values.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative number.")

    # ---------------------------------------------------------
    # 4. Compute pairwise squared Euclidean distances.
    # ---------------------------------------------------------

    # Reuse pairwise_sq_dists() from Step 1.
    #
    # Shape: (n_query, N)

    sq_dists = pairwise_sq_dists(
        x_query,
        x_train
    )

    # ---------------------------------------------------------
    # 5. Compute Gaussian kernel weights in log space.
    # ---------------------------------------------------------

    # Gaussian kernel:
    #
    # K_h(x - x_n) = exp(-||x - x_n||^2 / (2*h^2))
    #
    # The common Gaussian normalization constant cancels
    # when calculating normalized weights.

    log_weights = -sq_dists / (2.0 * h**2)

    # ---------------------------------------------------------
    # 6. Normalize weights using the log-sum-exp trick.
    # ---------------------------------------------------------

    # Subtract the maximum log weight in each query row
    # before taking exponentials.
    #
    # This prevents all weights from underflowing to zero
    # when queries are far from the training observations.

    if n_query == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    row_max = np.max(
        log_weights,
        axis=1,
        keepdims=True
    )

    shifted_weights = np.exp(
        log_weights - row_max
    )

    # Normalize every row so that the weights sum to one.
    weights = shifted_weights / np.sum(
        shifted_weights,
        axis=1,
        keepdims=True
    )

    # ---------------------------------------------------------
    # 7. Compute the Nadaraya-Watson conditional mean.
    # ---------------------------------------------------------

    # mu(x) = sum_n w_n(x) * y_n

    mean = weights @ y_train

    # ---------------------------------------------------------
    # 8. Compute the conditional predictive variance.
    # ---------------------------------------------------------

    # V(x) = sigma^2
    #        + sum_n w_n(x) * y_n^2
    #        - mu(x)^2

    second_moment = weights @ (y_train**2)

    var = (
        sigma**2
        + second_moment
        - mean**2
    )

    # Floating-point rounding can produce tiny negative values.
    # Variance cannot be negative, so clip at zero.
    var = np.maximum(var, 0.0)

    return mean, var

# Step 13 - loo_nw_bandwidth
def loo_nw_bandwidth(x, y, grid):
    """Return (best_h, curve) where curve contains the leave-one-out
    mean squared error of Nadaraya-Watson regression for each bandwidth.

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, D)
        Training input observations.

    y : ndarray, shape (N,)
        Training target values. Flattened automatically.

    grid : array-like, shape (n_bandwidths,)
        Candidate positive bandwidth values.

    Returns
    -------
    best_h : float
        Bandwidth with the smallest leave-one-out mean squared error.

    curve : ndarray
        Float array containing one LOO MSE per bandwidth.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float)

    # Accept either 1-D or 2-D input features.
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("x must be a 1-D or 2-D array.")

    # ---------------------------------------------------------
    # 2. Validate the input data.
    # ---------------------------------------------------------

    N, D = x.shape

    if N < 2:
        raise ValueError(
            "At least two observations are required for leave-one-out validation."
        )

    if D == 0:
        raise ValueError("x must contain at least one feature.")

    if len(y) != N:
        raise ValueError("x and y must contain the same number of observations.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite values.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain finite values.")

    # ---------------------------------------------------------
    # 3. Validate the bandwidth grid.
    # ---------------------------------------------------------

    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array.")

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError(
            "Every bandwidth in grid must be positive and finite."
        )

    # ---------------------------------------------------------
    # 4. Compute the pairwise squared-distance matrix ONCE.
    # ---------------------------------------------------------

    # Reuse pairwise_sq_dists() from Step 1.
    #
    # Shape: (N, N)

    sq_dists = pairwise_sq_dists(x)

    # Exclude each observation's self-distance.
    #
    # This ensures that every observation is predicted using
    # all other observations, rather than itself.

    np.fill_diagonal(sq_dists, np.inf)

    # ---------------------------------------------------------
    # 5. Prepare the leave-one-out calculations.
    # ---------------------------------------------------------

    # Store the minimum non-self squared distance for each row.
    #
    # Subtracting this minimum before exponentiation provides
    # numerically stable Gaussian weights.

    min_sq_dists = np.min(
        sq_dists,
        axis=1,
        keepdims=True
    )

    # Difference from the nearest non-self observation.
    shifted_dists = sq_dists - min_sq_dists

    # Store one MSE for each candidate bandwidth.
    curve = np.empty(len(grid), dtype=float)

    # ---------------------------------------------------------
    # 6. Evaluate every candidate bandwidth.
    # ---------------------------------------------------------

    for i, h in enumerate(grid):

        # Gaussian kernel exponent:
        #
        # exp(-||x_i - x_j||^2 / (2*h^2))
        #
        # Subtracting the minimum squared distance does not
        # change the normalized weights.
        #
        # Divide sequentially by h to avoid underflow in h**2
        # for extremely small positive bandwidths.

        with np.errstate(
            over="ignore",
            under="ignore",
            divide="ignore",
            invalid="ignore"
        ):
            exponent = (shifted_dists / h) / (2.0 * h)

            weights = np.exp(-exponent)

        # Normalize the weights for each left-out observation.
        row_sums = np.sum(weights, axis=1, keepdims=True)

        normalized_weights = weights / row_sums

        # -----------------------------------------------------
        # 7. Compute the leave-one-out predictions.
        # -----------------------------------------------------

        # Each row i contains the normalized weights for
        # predicting y_i using all observations except i.

        y_pred = normalized_weights @ y

        # -----------------------------------------------------
        # 8. Calculate the leave-one-out mean squared error.
        # -----------------------------------------------------

        errors = y - y_pred

        curve[i] = np.mean(errors**2)

    # ---------------------------------------------------------
    # 9. Select the bandwidth with the smallest LOO MSE.
    # ---------------------------------------------------------

    best_index = int(np.argmin(curve))

    best_h = float(grid[best_index])

    return best_h, curve

# Step 14 - loess
def loess(x_train, y_train, x_query, h, kernel="gaussian"):
    """Return locally linear regression predictions (eq 16.47).

    At every query point x0, fit a weighted least-squares line:

        y = beta[0] + beta[1] * x

    using kernel weights:

        w_i = K((x0 - x_train[i]) / h)

    Parameters
    ----------
    x_train : array-like
        One-dimensional training inputs.

    y_train : array-like
        Training target values.

    x_query : array-like
        Query points where predictions are required.

    h : float
        Positive kernel bandwidth.

    kernel : str, default="gaussian"
        Kernel name accepted by density_kernel():
        'gaussian', 'boxcar', 'epanechnikov', or 'tricube'.

    Returns
    -------
    predictions : ndarray
        One locally linear regression prediction per query point.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to flattened NumPy arrays.
    # ---------------------------------------------------------

    x_train = np.asarray(x_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    x_query = np.asarray(x_query, dtype=float).ravel()

    # ---------------------------------------------------------
    # 2. Validate input data.
    # ---------------------------------------------------------

    if x_train.size == 0:
        raise ValueError("x_train must not be empty.")

    if x_train.size != y_train.size:
        raise ValueError(
            "x_train and y_train must have the same number of observations."
        )

    if not np.all(np.isfinite(x_train)):
        raise ValueError("x_train must contain finite values.")

    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train must contain finite values.")

    if not np.all(np.isfinite(x_query)):
        raise ValueError("x_query must contain finite values.")

    # ---------------------------------------------------------
    # 3. Validate the bandwidth.
    # ---------------------------------------------------------

    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("h must be a positive finite number.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    # Validate the kernel name using the earlier function.
    density_kernel(np.array([0.0]), name=kernel)

    # ---------------------------------------------------------
    # 4. Construct the linear design matrix.
    # ---------------------------------------------------------

    # phi(x) = [1, x]
    #
    # Each training observation has two basis features:
    # an intercept and a slope.

    Phi = np.column_stack([
        np.ones(x_train.size),
        x_train
    ])

    # Store one prediction per query.
    predictions = np.empty(x_query.size, dtype=float)

    # ---------------------------------------------------------
    # 5. Perform a separate weighted linear fit for each query.
    # ---------------------------------------------------------

    for i, x0 in enumerate(x_query):

        # Compute the standardized differences.
        u = (x0 - x_train) / h

        # Evaluate the selected kernel.
        weights = density_kernel(u, name=kernel)

        # -----------------------------------------------------
        # 6. Stabilize the weights.
        # -----------------------------------------------------

        # Multiplying all weights by the same positive constant
        # does not change the weighted least-squares solution.
        #
        # Therefore, divide by the maximum weight whenever
        # at least one kernel weight is positive.

        max_weight = np.max(weights)

        if max_weight > 0:

            weights = weights / max_weight

        elif kernel == "gaussian":

            # Gaussian weights may all underflow to zero for
            # queries far away from the training observations.
            #
            # Use an equivalent log-space Gaussian expression.

            with np.errstate(over="ignore", invalid="ignore"):
                log_weights = -0.5 * u**2

            max_log_weight = np.max(log_weights)

            if np.isfinite(max_log_weight):

                weights = np.exp(
                    log_weights - max_log_weight
                )

            else:

                # Extreme floating-point inputs can overflow
                # even the squared standardized differences.
                # Retain the nearest training observation(s).
                distances = np.abs(x0 - x_train)
                minimum_distance = np.min(distances)

                weights = (
                    distances == minimum_distance
                ).astype(float)

        # -----------------------------------------------------
        # 7. Construct the weighted normal equations.
        # -----------------------------------------------------

        # Solve:
        #
        # beta = argmin sum_j w_j *
        #        (y_j - beta[0] - beta[1]*x_j)^2
        #
        # Normal equations:
        #
        # A = Phi.T @ W @ Phi
        # b = Phi.T @ W @ y
        #
        # Use weighted multiplication instead of explicitly
        # constructing the diagonal matrix W.

        A = Phi.T @ (weights[:, None] * Phi)

        b = Phi.T @ (weights * y_train)

        # -----------------------------------------------------
        # 8. Solve the normal equations using least squares.
        # -----------------------------------------------------

        # IMPORTANT:
        # Do not use np.linalg.inv(A).
        #
        # Compact kernels may leave too few observations in the
        # window, making A singular or rank deficient.
        #
        # np.linalg.lstsq handles these cases safely.

        beta = np.linalg.lstsq(
            A,
            b,
            rcond=None
        )[0]

        # -----------------------------------------------------
        # 9. Predict at the query point.
        # -----------------------------------------------------

        # mu(x0) = beta[0] + beta[1] * x0

        phi_query = np.array([1.0, x0])

        predictions[i] = phi_query @ beta

    return predictions

# Step 15 - nca_objective
def pair_gradient(W, X, A):
    """Gradient with respect to W of:

        sum_ij A[i,j] * ||W x_i - W x_j||^2

    The matrix A is treated as fixed.

    Parameters
    ----------
    W : ndarray, shape (L, D)
        Linear transformation matrix.

    X : ndarray, shape (N, D)
        Input observations.

    A : ndarray, shape (N, N)
        Fixed pairwise weights.

    Returns
    -------
    grad : ndarray, shape (L, D)
        Gradient of the weighted pairwise distance objective.
    """

    # Convert inputs to floating-point NumPy arrays.
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)

    # Validate dimensions.
    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")

    N, D = X.shape
    L = W.shape[0]

    if W.shape[1] != D:
        raise ValueError("W and X must have compatible feature dimensions.")

    if A.shape != (N, N):
        raise ValueError("A must have shape (N, N).")

    if N == 0 or D == 0 or L == 0:
        raise ValueError("W and X must have non-empty dimensions.")

    if not (
        np.all(np.isfinite(W))
        and np.all(np.isfinite(X))
        and np.all(np.isfinite(A))
    ):
        raise ValueError("W, X and A must contain finite values.")

    # ---------------------------------------------------------
    # Derive the pairwise gradient using a shared matrix identity.
    # ---------------------------------------------------------
    #
    # Let:
    #
    # S(W) = sum_ij A[i,j] ||W(x_i - x_j)||^2
    #
    # Its gradient is:
    #
    # dS/dW = 2 W sum_ij A[i,j](x_i-x_j)(x_i-x_j)^T
    #
    # Expanding the pairwise sum gives:
    #
    # B = X.T @ diag(A.sum(axis=1) + A.sum(axis=0)) @ X
    #     - X.T @ (A + A.T) @ X
    #
    # We compute this without constructing a diagonal matrix
    # and without looping over pairs of observations.

    row_col_sum = A.sum(axis=1) + A.sum(axis=0)

    term1 = X.T @ (row_col_sum[:, None] * X)

    term2 = X.T @ (A + A.T) @ X

    B = term1 - term2

    # Exact analytical gradient.
    grad = 2.0 * (W @ B)

    return grad


def nca_objective(W, X, y):
    """Return (J, grad) for neighbourhood components analysis.

    The objective is the expected number of correctly classified
    points under the stochastic nearest-neighbour rule.

    Parameters
    ----------
    W : ndarray, shape (L, D)
        Linear transformation matrix.

    X : ndarray, shape (N, D)
        Training observations.

    y : ndarray, shape (N,)
        Class labels.

    Returns
    -------
    J : float
        NCA objective value.

    grad : ndarray, shape (L, D)
        Exact analytical gradient dJ/dW.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate inputs.
    # ---------------------------------------------------------

    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    N, D = X.shape
    L = W.shape[0]

    if N < 2:
        raise ValueError("NCA requires at least two observations.")

    if D == 0 or L == 0:
        raise ValueError("W and X must have non-empty feature dimensions.")

    if W.shape[1] != D:
        raise ValueError("W must have shape (L, D).")

    if len(y) != N:
        raise ValueError("X and y must contain the same number of observations.")

    if not (
        np.all(np.isfinite(W))
        and np.all(np.isfinite(X))
    ):
        raise ValueError("W and X must contain finite values.")

    # ---------------------------------------------------------
    # 2. Transform the observations using W.
    # ---------------------------------------------------------

    # Each transformed observation is:
    #
    # z_i = W x_i
    #
    # Since observations are stored row-wise:
    #
    # Z = X @ W.T

    Z = X @ W.T

    # ---------------------------------------------------------
    # 3. Compute pairwise squared transformed distances.
    # ---------------------------------------------------------

    # Reuse pairwise_sq_dists() from Step 1.
    #
    # d_ij = ||W x_i - W x_j||^2

    distances = pairwise_sq_dists(Z)

    # Exclude self-neighbours.
    # NCA considers only j != i.
    np.fill_diagonal(distances, np.inf)

    # ---------------------------------------------------------
    # 4. Compute stable neighbour-selection probabilities.
    # ---------------------------------------------------------

    # p_ij = exp(-d_ij) / sum_{k != i} exp(-d_ik)
    #
    # Direct exponentiation can underflow when W is large.
    # Subtract the smallest distance in each row before
    # exponentiation. This is equivalent to log-sum-exp
    # normalization because the common factor cancels.

    min_distances = np.min(
        distances,
        axis=1,
        keepdims=True
    )

    shifted_distances = distances - min_distances

    with np.errstate(
        under="ignore",
        over="ignore",
        invalid="ignore"
    ):
        exp_weights = np.exp(-shifted_distances)

    # Normalize each row.
    probabilities = exp_weights / np.sum(
        exp_weights,
        axis=1,
        keepdims=True
    )

    # Explicitly ensure self-selection probabilities are zero.
    np.fill_diagonal(probabilities, 0.0)

    # ---------------------------------------------------------
    # 5. Compute the NCA objective.
    # ---------------------------------------------------------

    # same[i,j] = 1 if observations i and j belong
    # to the same class; otherwise 0.
    same = (y[:, None] == y[None, :]).astype(float)

    # Self-neighbours are excluded.
    np.fill_diagonal(same, 0.0)

    # J(W) = sum_i sum_{j:y_j=y_i} p_ij
    J = float(np.sum(probabilities * same))

    # ---------------------------------------------------------
    # 6. Compute the probability of selecting a same-class
    #    neighbour for each observation.
    # ---------------------------------------------------------

    # q_i = sum_{j:y_j=y_i} p_ij

    p_same = np.sum(
        probabilities * same,
        axis=1,
        keepdims=True
    )

    # ---------------------------------------------------------
    # 7. Construct the fixed pairwise weight matrix A.
    # ---------------------------------------------------------

    # The NCA gradient can be expressed as:
    #
    # dJ/dW = - pair_gradient(W, X, A)
    #
    # where:
    #
    # A_ij = p_ij * (same_ij - q_i)
    #
    # This matrix captures the difference between selecting
    # a same-class neighbour and selecting an arbitrary neighbour.

    A = probabilities * (same - p_same)

    # ---------------------------------------------------------
    # 8. Calculate the exact analytical gradient.
    # ---------------------------------------------------------

    grad = -pair_gradient(W, X, A)

    return J, grad

# Step 16 - lmnn_objective
def target_neighbors(X, y, K):
    """Return the (N, K) array of same-class nearest neighbours.

    Neighbours are selected using Euclidean distance, excluding
    the observation itself. Ties are resolved by smaller index.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Training observations.

    y : ndarray, shape (N,)
        Class labels.

    K : int
        Number of same-class target neighbours.

    Returns
    -------
    targets : ndarray, shape (N, K)
        Indices of the K nearest same-class neighbours for each
        training observation.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate inputs.
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    N, D = X.shape

    if N == 0 or D == 0:
        raise ValueError("X must contain observations and features.")

    if len(y) != N:
        raise ValueError("X and y must have the same number of observations.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    if (
        isinstance(K, (bool, np.bool_))
        or not isinstance(K, (int, np.integer))
        or K < 1
    ):
        raise ValueError("K must be a positive integer.")

    # ---------------------------------------------------------
    # 2. Calculate pairwise squared Euclidean distances.
    # ---------------------------------------------------------

    # Reuse pairwise_sq_dists() from Step 1.
    distances = pairwise_sq_dists(X)

    # ---------------------------------------------------------
    # 3. Identify same-class neighbours.
    # ---------------------------------------------------------

    # same_class[i, j] is True when observations i and j
    # belong to the same class.

    same_class = y[:, None] == y[None, :]

    # An observation cannot be its own neighbour.
    np.fill_diagonal(same_class, False)

    # Each observation must have at least K other observations
    # belonging to its class.
    if np.any(np.sum(same_class, axis=1) < K):
        raise ValueError(
            "Every observation must have at least K other observations "
            "belonging to the same class."
        )

    # Replace distances to different-class observations
    # with infinity so that they cannot be selected.

    masked_distances = np.where(
        same_class,
        distances,
        np.inf
    )

    # ---------------------------------------------------------
    # 4. Select the K nearest same-class observations.
    # ---------------------------------------------------------

    # Stable sorting ensures that smaller training indices
    # win whenever distances are equal.

    sorted_indices = np.argsort(
        masked_distances,
        axis=1,
        kind="stable"
    )

    targets = sorted_indices[:, :K]

    return targets


def lmnn_objective(W, X, y, targets, m=1.0, lam=0.5):
    """Return (loss, grad, n_active) for LMNN (eq 16.4, 16.5).

    The LMNN objective combines:
      - Pull loss: brings target neighbours closer.
      - Push loss: separates impostors using a margin constraint.

    Parameters
    ----------
    W : ndarray, shape (L, D)
        Linear transformation matrix.

    X : ndarray, shape (N, D)
        Training observations.

    y : ndarray, shape (N,)
        Training class labels.

    targets : ndarray, shape (N, K)
        Fixed target-neighbour indices, computed on the original
        feature space using target_neighbors().

    m : float, default=1.0
        Margin used in the impostor constraints.

    lam : float, default=0.5
        Weight assigned to the push loss. Must lie in [0, 1].

    Returns
    -------
    loss : float
        Total LMNN objective.

    grad : ndarray, shape (L, D)
        Exact gradient of the objective with the active hinge
        terms held fixed.

    n_active : int
        Number of triplets having strictly positive hinge loss.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs and validate dimensions.
    # ---------------------------------------------------------

    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    targets = np.asarray(targets)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    N, D = X.shape
    L = W.shape[0]

    if N == 0 or D == 0 or L == 0:
        raise ValueError("W and X must have non-empty dimensions.")

    if W.shape[1] != D:
        raise ValueError("W must have shape (L, D).")

    if len(y) != N:
        raise ValueError("X and y must have the same number of observations.")

    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(X)):
        raise ValueError("W and X must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate target-neighbour indices.
    # ---------------------------------------------------------

    if targets.ndim != 2 or targets.shape[0] != N:
        raise ValueError("targets must have shape (N, K).")

    if targets.shape[1] < 1:
        raise ValueError("targets must contain at least one neighbour per row.")

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets must contain integer indices.")

    K = targets.shape[1]

    if np.any(targets < 0) or np.any(targets >= N):
        raise ValueError("Every target index must be between 0 and N-1.")

    # Target neighbours must not include the observation itself.
    if np.any(targets == np.arange(N)[:, None]):
        raise ValueError("An observation cannot be its own target neighbour.")

    # Target neighbours must belong to the same class.
    if np.any(y[targets] != y[:, None]):
        raise ValueError("Every target neighbour must have the same class.")

    # Prevent duplicate target neighbours within a row.
    if np.any(np.diff(np.sort(targets, axis=1), axis=1) == 0):
        raise ValueError("Target neighbours must be unique within each row.")

    # ---------------------------------------------------------
    # 3. Validate margin and regularization weight.
    # ---------------------------------------------------------

    try:
        m = float(m)
        lam = float(lam)
    except (TypeError, ValueError):
        raise ValueError("m and lam must be finite scalar values.")

    if not np.isfinite(m) or m < 0:
        raise ValueError("m must be a finite non-negative number.")

    if not np.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise ValueError("lam must lie between 0 and 1.")

    # ---------------------------------------------------------
    # 4. Transform the training observations.
    # ---------------------------------------------------------

    # z_i = W @ x_i
    #
    # Since observations are stored row-wise:
    #
    # Z = X @ W.T

    Z = X @ W.T

    # ---------------------------------------------------------
    # 5. Compute all transformed pairwise squared distances.
    # ---------------------------------------------------------

    # d_ij^2 = ||W x_i - W x_j||^2

    distances = pairwise_sq_dists(Z)

    # Row indices for vectorized access to target distances.
    rows = np.arange(N)

    # ---------------------------------------------------------
    # 6. Initialize the pull loss and push loss.
    # ---------------------------------------------------------

    # Pull loss:
    #
    # L_pull = sum_i sum_{j in N_i} d_ij^2

    target_distances = distances[rows[:, None], targets]

    L_pull = float(np.sum(target_distances))

    # Push loss:
    #
    # L_push = sum_i sum_{j in N_i} sum_{l:y_l != y_i}
    #          [m + d_ij^2 - d_il^2]_+

    L_push = 0.0

    # Count triplets with strictly positive hinge loss.
    n_active = 0

    # ---------------------------------------------------------
    # 7. Construct the fixed pairwise weight matrix A.
    # ---------------------------------------------------------

    # The entire loss gradient can be represented through:
    #
    # grad = pair_gradient(W, X, A)
    #
    # A has shape (N, N).

    A = np.zeros((N, N), dtype=float)

    # Difference labels identify impostor observations.
    impostors = y[:, None] != y[None, :]

    # ---------------------------------------------------------
    # 8. Process each target-neighbour column.
    # ---------------------------------------------------------

    # This is the only loop over target neighbours.
    # There are no loops over individual training observations.

    for k in range(K):

        # Target neighbour of each anchor observation.
        j = targets[:, k]

        # Squared distance between every anchor and its
        # corresponding target neighbour.
        d_target = distances[rows, j]

        # -----------------------------------------------------
        # 8a. Calculate the impostor hinge terms.
        # -----------------------------------------------------

        # hinge[i, l] = m + d_ij^2 - d_il^2

        hinge = (
            m
            + d_target[:, None]
            - distances
        )

        # A triplet is active only when:
        # 1. l belongs to a different class.
        # 2. Its hinge value is strictly positive.

        active = (hinge > 0.0) & impostors

        # -----------------------------------------------------
        # 8b. Accumulate the push loss.
        # -----------------------------------------------------

        positive_hinge = np.where(
            active,
            hinge,
            0.0
        )

        L_push += float(np.sum(positive_hinge))

        # Count active triplets.
        n_active += int(np.count_nonzero(active))

        # -----------------------------------------------------
        # 8c. Accumulate the pull and push gradient weights.
        # -----------------------------------------------------

        # Pull contribution:
        # (1 - lam) * d_ij^2

        # Push contribution for the target pair:
        # +lam * d_ij^2 for each active impostor triplet.

        active_per_anchor = np.sum(
            active,
            axis=1
        )

        A[rows, j] += (
            (1.0 - lam)
            + lam * active_per_anchor
        )

        # Push contribution for each impostor pair:
        # -lam * d_il^2 for every active triplet.

        A -= lam * active

    # ---------------------------------------------------------
    # 9. Compute the total LMNN objective.
    # ---------------------------------------------------------

    # loss = (1 - lam) * L_pull + lam * L_push

    loss = (
        (1.0 - lam) * L_pull
        + lam * L_push
    )

    # ---------------------------------------------------------
    # 10. Compute the exact analytical gradient.
    # ---------------------------------------------------------

    # Reuse the shared pairwise gradient helper from Step 15.

    grad = pair_gradient(
        W,
        X,
        A
    )

    return float(loss), grad, int(n_active)

# Step 17 - fit_linear_metric
def fit_linear_metric(objective, W0, steps=200, lr=0.01):
    """Minimise objective(W) -> (loss, grad) with Adam.

    Parameters
    ----------
    objective : callable
        Function accepting W and returning (loss, grad).
        The gradient must have the same shape as W.

    W0 : ndarray, shape (L, D)
        Initial transformation matrix. It is not modified.

    steps : int, default=200
        Number of Adam optimization iterations.

    lr : float, default=0.01
        Learning rate.

    Returns
    -------
    W : ndarray
        Optimized transformation matrix.

    history : list of float
        Objective loss at the start of each iteration.
    """

    # ---------------------------------------------------------
    # 1. Validate the initial matrix and optimizer parameters.
    # ---------------------------------------------------------

    W0 = np.asarray(W0, dtype=float)

    if W0.ndim != 2:
        raise ValueError("W0 must be a 2D matrix.")

    if W0.size == 0:
        raise ValueError("W0 must not be empty.")

    if not np.all(np.isfinite(W0)):
        raise ValueError("W0 must contain finite values.")

    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 0
    ):
        raise ValueError("steps must be a non-negative integer.")

    try:
        lr = float(lr)
    except (TypeError, ValueError):
        raise ValueError("lr must be a positive finite number.")

    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a positive finite number.")

    # Make a copy so that the original W0 remains unchanged.
    W = W0.copy()

    # ---------------------------------------------------------
    # 2. Initialize Adam hyperparameters.
    # ---------------------------------------------------------

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # First and second moment estimates.
    m = np.zeros_like(W)
    v = np.zeros_like(W)

    # Store the loss at the start of each iteration.
    history = []

    # ---------------------------------------------------------
    # 3. Perform Adam optimization.
    # ---------------------------------------------------------

    for t in range(1, steps + 1):

        # Evaluate the objective exactly once per iteration.
        # This is important because the objective may generate
        # a fresh minibatch on every call.
        loss, grad = objective(W)

        loss = float(loss)
        grad = np.asarray(grad, dtype=float)

        # Validate the returned objective and gradient.
        if not np.isfinite(loss):
            raise FloatingPointError(
                "The objective returned a non-finite loss."
            )

        if grad.shape != W.shape:
            raise ValueError(
                "The gradient must have the same shape as W."
            )

        if not np.all(np.isfinite(grad)):
            raise FloatingPointError(
                "The objective returned a non-finite gradient."
            )

        # Record the loss BEFORE applying this iteration's update.
        history.append(loss)

        # -----------------------------------------------------
        # 4. Update the first moment estimate.
        # -----------------------------------------------------

        # m_t = beta1 * m_(t-1) + (1 - beta1) * g_t

        m = beta1 * m + (1.0 - beta1) * grad

        # -----------------------------------------------------
        # 5. Update the second moment estimate.
        # -----------------------------------------------------

        # v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2

        v = beta2 * v + (1.0 - beta2) * grad**2

        # -----------------------------------------------------
        # 6. Apply bias correction.
        # -----------------------------------------------------

        # Corrected first moment.
        m_hat = m / (1.0 - beta1**t)

        # Corrected second moment.
        v_hat = v / (1.0 - beta2**t)

        # -----------------------------------------------------
        # 7. Update the parameter matrix.
        # -----------------------------------------------------

        # W = W - lr * m_hat / sqrt(v_hat + epsilon)

        W = W - lr * m_hat / np.sqrt(v_hat + epsilon)

    return W, history

# Step 18 - whitened_pca
def split_identities(n_people, n_train, n_calib, n_enrolled, n_impostors, seed=0):
    """Split person IDs into train, calib, enrolled and impostors groups.

    A seeded random permutation is divided into four consecutive
    groups in the specified order.

    Parameters
    ----------
    n_people : int
        Total number of available identities.

    n_train : int
        Number of identities assigned to training.

    n_calib : int
        Number of identities assigned to calibration.

    n_enrolled : int
        Number of identities assigned to enrollment.

    n_impostors : int
        Number of identities assigned to the impostor group.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing integer arrays with keys:
        'train', 'calib', 'enrolled', and 'impostors'.
    """

    # ---------------------------------------------------------
    # 1. Validate the identity counts.
    # ---------------------------------------------------------

    counts = {
        "n_people": n_people,
        "n_train": n_train,
        "n_calib": n_calib,
        "n_enrolled": n_enrolled,
        "n_impostors": n_impostors
    }

    for name, value in counts.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    n_people = int(n_people)
    n_train = int(n_train)
    n_calib = int(n_calib)
    n_enrolled = int(n_enrolled)
    n_impostors = int(n_impostors)

    # The requested groups cannot contain more identities
    # than are available.
    total_requested = (
        n_train + n_calib + n_enrolled + n_impostors
    )

    if total_requested > n_people:
        raise ValueError(
            "The total number of requested identities "
            "cannot exceed n_people."
        )

    # ---------------------------------------------------------
    # 2. Generate a reproducible permutation of person IDs.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    perm = rng.permutation(n_people)

    # ---------------------------------------------------------
    # 3. Divide the permutation into the four groups.
    # ---------------------------------------------------------

    start = 0

    train_ids = perm[start:start + n_train]
    start += n_train

    calib_ids = perm[start:start + n_calib]
    start += n_calib

    enrolled_ids = perm[start:start + n_enrolled]
    start += n_enrolled

    impostor_ids = perm[start:start + n_impostors]

    # Return mutually exclusive identity groups.
    return {
        "train": train_ids,
        "calib": calib_ids,
        "enrolled": enrolled_ids,
        "impostors": impostor_ids
    }


def whitened_pca(X, k):
    """Fit PCA whitening on the rows of X.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Training observations, with samples in rows.

    k : int
        Number of principal components to retain.

    Returns
    -------
    mean : ndarray, shape (D,)
        Column-wise training mean.

    components : ndarray, shape (k, D)
        Top k right singular vectors of the centred data.

    scale : ndarray, shape (k,)
        Corresponding singular values divided by sqrt(N).
    """

    # ---------------------------------------------------------
    # 1. Convert and validate the input data.
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    N, D = X.shape

    if N == 0 or D == 0:
        raise ValueError("X must contain observations and features.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate the number of components.
    # ---------------------------------------------------------

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
    ):
        raise ValueError("k must be a positive integer.")

    if k >= N:
        raise ValueError("k must be smaller than the number of samples.")

    if k > D:
        raise ValueError("k cannot exceed the number of features.")

    # ---------------------------------------------------------
    # 3. Calculate the column-wise mean.
    # ---------------------------------------------------------

    mean = np.mean(X, axis=0)

    # Centre the observations using the training mean.
    X_centered = X - mean

    # ---------------------------------------------------------
    # 4. Compute the Singular Value Decomposition.
    # ---------------------------------------------------------

    # X_centered = U @ diag(S) @ Vt
    #
    # The rows of Vt contain the right singular vectors.
    #
    # Use compact SVD to avoid unnecessarily large matrices.

    U, singular_values, Vt = np.linalg.svd(
        X_centered,
        full_matrices=False
    )

    # ---------------------------------------------------------
    # 5. Select the top k principal components.
    # ---------------------------------------------------------

    components = Vt[:k]

    # ---------------------------------------------------------
    # 6. Calculate the whitening scales.
    # ---------------------------------------------------------

    # scale_j = singular_value_j / sqrt(N)
    #
    # This uses the population variance convention:
    #
    # Var(X_j) = singular_value_j^2 / N

    scale = singular_values[:k] / np.sqrt(N)

    # A zero-variance component cannot be whitened.
    if np.any(scale <= 0):
        raise ValueError(
            "The selected principal components must have "
            "strictly positive variance."
        )

    return mean, components, scale


def apply_whitening(X, mean, components, scale):
    """Centre, project and scale X into whitened PCA coordinates.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Observations to transform.

    mean : ndarray, shape (D,)
        Mean learned from the PCA fitting data.

    components : ndarray, shape (k, D)
        Principal component vectors.

    scale : ndarray, shape (k,)
        Standard deviations corresponding to the retained
        principal components.

    Returns
    -------
    ndarray, shape (N, k)
        Whitened observations.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    components = np.asarray(components, dtype=float)
    scale = np.asarray(scale, dtype=float)

    # ---------------------------------------------------------
    # 2. Validate dimensions.
    # ---------------------------------------------------------

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if mean.ndim != 1:
        raise ValueError("mean must be a 1D array.")

    if components.ndim != 2:
        raise ValueError("components must be a 2D array.")

    if scale.ndim != 1:
        raise ValueError("scale must be a 1D array.")

    N, D = X.shape
    k, component_D = components.shape

    if D != len(mean) or D != component_D:
        raise ValueError(
            "The feature dimensions of X, mean and components must match."
        )

    if len(scale) != k:
        raise ValueError(
            "scale must contain one value per principal component."
        )

    # ---------------------------------------------------------
    # 3. Validate values and scales.
    # ---------------------------------------------------------

    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(components))
        and np.all(np.isfinite(scale))
    ):
        raise ValueError("All inputs must contain finite values.")

    if np.any(scale <= 0):
        raise ValueError("Every whitening scale must be positive.")

    # ---------------------------------------------------------
    # 4. Centre the observations.
    # ---------------------------------------------------------

    X_centered = X - mean

    # ---------------------------------------------------------
    # 5. Project onto the retained principal components.
    # ---------------------------------------------------------

    projected = X_centered @ components.T

    # ---------------------------------------------------------
    # 6. Divide by the corresponding standard deviations.
    # ---------------------------------------------------------

    whitened = projected / scale

    return whitened

# Step 19 - contrastive_loss
def unit_rows(E):
    """Return (E_hat, norms), with each row scaled to unit length.

    Rows with zero length are handled safely by flooring the
    divisor at 1e-12.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding matrix.

    Returns
    -------
    E_hat : ndarray, shape (N, L)
        Row-normalized embedding matrix.

    norms : ndarray, shape (N, 1)
        Original Euclidean norm of each row.
    """

    # Convert the embedding matrix to a floating-point array.
    E = np.asarray(E, dtype=float)

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    if not np.all(np.isfinite(E)):
        raise ValueError("E must contain finite values.")

    # Calculate the Euclidean norm of each embedding.
    # keepdims=True preserves the shape (N, 1).
    norms = np.linalg.norm(
        E,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero for zero-length embeddings.
    safe_norms = np.maximum(norms, 1e-12)

    # Normalize every row independently.
    E_hat = E / safe_norms

    return E_hat, norms


def contrastive_loss(E, y, m=1.0):
    """Return (loss, G): mean pairwise contrastive loss
    and its exact gradient with respect to E.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding matrix. E is treated as a free variable
        and is not normalized inside this function.

    y : ndarray, shape (N,)
        Class labels.

    m : float, default=1.0
        Margin for negative pairs.

    Returns
    -------
    loss : float
        Mean contrastive loss over all unordered pairs.

    G : ndarray, shape (N, L)
        Gradient of the loss with respect to E.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate inputs.
    # ---------------------------------------------------------

    E = np.asarray(E, dtype=float)
    y = np.asarray(y).ravel()

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    N, L = E.shape

    if N < 2:
        raise ValueError(
            "At least two embeddings are required to form pairs."
        )

    if L < 1:
        raise ValueError("E must contain at least one embedding dimension.")

    if len(y) != N:
        raise ValueError(
            "E and y must contain the same number of observations."
        )

    if not np.all(np.isfinite(E)):
        raise ValueError("E must contain finite values.")

    try:
        m = float(m)
    except (TypeError, ValueError):
        raise ValueError("m must be a finite non-negative number.")

    if not np.isfinite(m) or m < 0:
        raise ValueError("m must be a finite non-negative number.")

    # ---------------------------------------------------------
    # 2. Generate all unordered pairs i < j.
    # ---------------------------------------------------------

    # Only the upper triangular pairs are used.
    # This avoids counting each pair twice.

    i, j = np.triu_indices(N, k=1)

    # Total number of unordered pairs.
    n_pairs = len(i)

    # ---------------------------------------------------------
    # 3. Compute pairwise embedding differences and distances.
    # ---------------------------------------------------------

    # Difference vector for every unordered pair:
    #
    # delta_ij = e_i - e_j

    diff = E[i] - E[j]

    # Euclidean distance:
    #
    # d_ij = ||e_i - e_j||_2

    distances = np.linalg.norm(
        diff,
        axis=1
    )

    # ---------------------------------------------------------
    # 4. Identify positive and negative pairs.
    # ---------------------------------------------------------

    # Positive pair: same class.
    # Negative pair: different classes.

    positive = y[i] == y[j]
    negative = ~positive

    # ---------------------------------------------------------
    # 5. Calculate the contrastive loss for every pair.
    # ---------------------------------------------------------

    # Positive-pair loss:
    #
    # l_ij = d_ij^2
    #
    # Negative-pair loss:
    #
    # l_ij = max(m - d_ij, 0)^2

    pair_losses = np.zeros(n_pairs, dtype=float)

    # Positive pairs.
    pair_losses[positive] = distances[positive] ** 2

    # Negative pairs.
    negative_hinge = np.maximum(
        m - distances[negative],
        0.0
    )

    pair_losses[negative] = negative_hinge ** 2

    # Average over all unordered pairs.
    loss = float(np.mean(pair_losses))

    # ---------------------------------------------------------
    # 6. Calculate the gradient coefficient for each pair.
    # ---------------------------------------------------------

    # For a positive pair:
    #
    # d l_ij / d delta_ij = 2 * delta_ij
    #
    # Therefore, coefficient = 2.

    coefficients = np.zeros(n_pairs, dtype=float)

    coefficients[positive] = 2.0

    # For a negative pair with an active hinge:
    #
    # l_ij = (m - d_ij)^2
    #
    # d l_ij / d delta_ij =
    #
    # -2(m - d_ij) * delta_ij / d_ij
    #
    # coefficient = -2(m - d_ij) / d_ij

    active_negative = negative & (distances < m)

    # When d_ij = 0 for a negative pair, the distance direction
    # is undefined. The required subgradient is explicitly zero.

    valid_negative = active_negative & (distances > 0.0)

    coefficients[valid_negative] = (
        -2.0
        * (m - distances[valid_negative])
        / distances[valid_negative]
    )

    # Negative pairs with zero distance retain coefficient zero.

    # ---------------------------------------------------------
    # 7. Calculate pairwise gradient contributions.
    # ---------------------------------------------------------

    # Normalize by the number of unordered pairs because
    # the objective is the mean pairwise loss.

    pair_grad = (
        coefficients[:, None] * diff
    ) / n_pairs

    # ---------------------------------------------------------
    # 8. Accumulate gradients for both endpoints of every pair.
    # ---------------------------------------------------------

    G = np.zeros_like(E)

    # Gradient contribution for the first endpoint.
    np.add.at(G, i, pair_grad)

    # Gradient contribution for the second endpoint.
    np.add.at(G, j, -pair_grad)

    return loss, G

# Step 20 - mine_triplets
def mine_triplets(E, y, m=0.2, mode="semi-hard"):
    """Return a (T, 3) integer array of (anchor, positive, negative)
    triplets selected according to the specified mining strategy.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding vectors, with observations stored in rows.

    y : ndarray, shape (N,)
        Class labels.

    m : float, default=0.2
        Margin used for semi-hard and easy negative mining.

    mode : str, default="semi-hard"
        Negative mining strategy:
            'all'       : Every different-class negative.
            'hard'      : d(a,n) < d(a,p).
            'semi-hard' : d(a,p) < d(a,n) < d(a,p) + m.
            'easy'      : d(a,n) > d(a,p) + m.

    Returns
    -------
    triplets : ndarray, shape (T, 3)
        Integer array containing (anchor, positive, negative)
        indices, sorted by anchor, positive, and negative indices.
        Returns an empty array of shape (0, 3) if no triplets exist.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate the inputs.
    # ---------------------------------------------------------

    E = np.asarray(E, dtype=float)
    y = np.asarray(y)

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    N, L = E.shape

    if L == 0:
        raise ValueError("E must contain at least one embedding dimension.")

    if len(y) != N:
        raise ValueError("E and y must have the same number of observations.")

    if not np.all(np.isfinite(E)):
        raise ValueError("E must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate the mining mode.
    # ---------------------------------------------------------

    valid_modes = ("all", "hard", "semi-hard", "easy")

    if mode not in valid_modes:
        raise ValueError(
            "Invalid mode. Choose from 'all', 'hard', "
            "'semi-hard', or 'easy'."
        )

    # ---------------------------------------------------------
    # 3. Validate the margin.
    # ---------------------------------------------------------

    try:
        m = float(m)
    except (TypeError, ValueError):
        raise ValueError("m must be a finite non-negative number.")

    if not np.isfinite(m) or m < 0:
        raise ValueError("m must be a finite non-negative number.")

    # ---------------------------------------------------------
    # 4. Compute pairwise Euclidean distances.
    # ---------------------------------------------------------

    # Reuse pairwise_sq_dists() from Step 1.
    #
    # The function returns squared distances, so take
    # the square root to obtain ordinary Euclidean distances.
    #
    # Shape: (N, N)

    sq_dists = pairwise_sq_dists(E)

    distances = np.sqrt(sq_dists)

    # ---------------------------------------------------------
    # 5. Identify all ordered anchor-positive pairs.
    # ---------------------------------------------------------

    # An anchor and positive must:
    #   1. Have the same class label.
    #   2. Be different observations.
    #
    # np.argwhere returns pairs in row-major order, meaning
    # anchors are sorted first, followed by positive indices.

    same_class = y[:, None] == y[None, :]

    # Exclude self-pairs.
    np.fill_diagonal(same_class, False)

    positive_pairs = np.argwhere(same_class)

    # ---------------------------------------------------------
    # 6. Identify negatives for each anchor-positive pair.
    # ---------------------------------------------------------

    triplets = []

    # Process anchor-positive pairs in their sorted order.
    for a, p in positive_pairs:

        # Positive distance between the anchor and positive.
        d_ap = distances[a, p]

        # Negative observations must have a different class
        # from the anchor.
        negative_mask = y != y[a]

        # Obtain negative indices in ascending order.
        negative_indices = np.flatnonzero(negative_mask)

        # -----------------------------------------------------
        # 7. Apply the requested mining strategy.
        # -----------------------------------------------------

        if mode == "all":

            selected_negatives = negative_indices

        else:

            # Euclidean distances from the anchor to all
            # candidate negatives.
            d_an = distances[a, negative_indices]

            if mode == "hard":

                # Hard negatives:
                # d(a,n) < d(a,p)

                keep = d_an < d_ap

            elif mode == "semi-hard":

                # Semi-hard negatives:
                # d(a,p) < d(a,n) < d(a,p) + m

                keep = (
                    (d_an > d_ap)
                    & (d_an < d_ap + m)
                )

            else:  # mode == "easy"

                # Easy negatives:
                # d(a,n) > d(a,p) + m

                keep = d_an > d_ap + m

            selected_negatives = negative_indices[keep]

        # -----------------------------------------------------
        # 8. Append the selected triplets.
        # -----------------------------------------------------

        for n in selected_negatives:
            triplets.append((a, p, n))

    # ---------------------------------------------------------
    # 9. Return a correctly shaped integer array.
    # ---------------------------------------------------------

    # Explicitly reshape so that an empty result has shape (0, 3).
    return np.asarray(
        triplets,
        dtype=int
    ).reshape(-1, 3)

# Step 21 - triplet_loss
def triplet_loss(E, triplets, m=0.2):
    """Return (loss, G): the mean triplet loss (eq 16.11)
    over the given triples and its exact gradient with respect to E.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding matrix. Each row represents an embedding.

    triplets : ndarray, shape (T, 3)
        Integer array containing (anchor, positive, negative)
        indices.

    m : float, default=0.2
        Non-negative triplet margin.

    Returns
    -------
    loss : float
        Mean triplet loss over all supplied triplets.

    G : ndarray, shape (N, L)
        Gradient of the loss with respect to E.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate the embedding matrix.
    # ---------------------------------------------------------

    E = np.asarray(E, dtype=float)

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    N, L = E.shape

    if L == 0:
        raise ValueError("E must contain at least one embedding dimension.")

    if not np.all(np.isfinite(E)):
        raise ValueError("E must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate the margin.
    # ---------------------------------------------------------

    try:
        m = float(m)
    except (TypeError, ValueError):
        raise ValueError("m must be a finite non-negative number.")

    if not np.isfinite(m) or m < 0:
        raise ValueError("m must be a finite non-negative number.")

    # ---------------------------------------------------------
    # 3. Validate triplets.
    # ---------------------------------------------------------

    triplets = np.asarray(triplets)

    # Handle empty triplet input.
    if triplets.size == 0:
        if triplets.shape not in ((0,), (0, 3)):
            raise ValueError("triplets must have shape (T, 3).")

        return 0.0, np.zeros_like(E)

    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError("triplets must have shape (T, 3).")

    if not np.issubdtype(triplets.dtype, np.integer):
        raise ValueError("Triplet indices must be integers.")

    if np.any(triplets < 0) or np.any(triplets >= N):
        raise ValueError("Triplet indices must be valid rows of E.")

    # Extract anchor, positive and negative indices.
    a = triplets[:, 0]
    p = triplets[:, 1]
    n = triplets[:, 2]

    # ---------------------------------------------------------
    # 4. Calculate embedding differences.
    # ---------------------------------------------------------

    diff_ap = E[a] - E[p]
    diff_an = E[a] - E[n]

    # ---------------------------------------------------------
    # 5. Compute squared Euclidean distances.
    # ---------------------------------------------------------

    d_ap_sq = np.sum(diff_ap ** 2, axis=1)
    d_an_sq = np.sum(diff_an ** 2, axis=1)

    # ---------------------------------------------------------
    # 6. Compute the triplet hinge loss.
    # ---------------------------------------------------------

    # loss_i = max(d_ap^2 - d_an^2 + m, 0)

    hinge = d_ap_sq - d_an_sq + m

    # Only strictly positive hinge terms contribute to gradients.
    active = hinge > 0.0

    triplet_losses = np.maximum(hinge, 0.0)

    # Average over ALL supplied triplets, including inactive ones.
    T = len(triplets)

    loss = float(np.mean(triplet_losses))

    # ---------------------------------------------------------
    # 7. Initialize the gradient.
    # ---------------------------------------------------------

    G = np.zeros_like(E)

    # No active triplets means zero gradient.
    if not np.any(active):
        return loss, G

    # Select active triplets.
    a_active = a[active]
    p_active = p[active]
    n_active = n[active]

    diff_ap_active = diff_ap[active]
    diff_an_active = diff_an[active]

    # Account for the mean over ALL triplets.
    factor = 2.0 / T

    # ---------------------------------------------------------
    # 8. Compute the exact analytical gradients.
    # ---------------------------------------------------------

    # For an active triplet:
    #
    # dL / d(e_anchor) =
    #     2 * [(e_anchor - e_positive)
    #          - (e_anchor - e_negative)]
    #
    # IMPORTANT: The anchor gradient uses diff_ap - diff_an.

    grad_anchor = factor * (
        diff_ap_active - diff_an_active
    )

    # dL / d(e_positive) =
    #     -2 * (e_anchor - e_positive)

    grad_positive = -factor * diff_ap_active

    # dL / d(e_negative) =
    #      2 * (e_anchor - e_negative)

    grad_negative = factor * diff_an_active

    # ---------------------------------------------------------
    # 9. Accumulate gradient contributions.
    # ---------------------------------------------------------

    # An embedding may appear in multiple triplets and in
    # multiple roles. np.add.at correctly accumulates all
    # contributions, including repeated indices.

    np.add.at(G, a_active, grad_anchor)
    np.add.at(G, p_active, grad_positive)
    np.add.at(G, n_active, grad_negative)

    return loss, G

# Step 22 - npairs_loss
def npairs_loss(E, y, tau=1.0):
    """Return (loss, G): mean N-pairs loss (eq 16.12)
    over ordered positive pairs and its gradient with respect to E.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding matrix, with observations stored in rows.

    y : ndarray, shape (N,)
        Class labels.

    tau : float, default=1.0
        Positive temperature parameter.

    Returns
    -------
    loss : float
        Mean N-pairs loss over all ordered positive pairs.

    G : ndarray, shape (N, L)
        Exact gradient of the loss with respect to E.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs and validate dimensions.
    # ---------------------------------------------------------

    E = np.asarray(E, dtype=float)
    y = np.asarray(y)

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    N, L = E.shape

    if N < 2 or L == 0:
        raise ValueError("E must contain at least two embeddings.")

    if len(y) != N:
        raise ValueError(
            "E and y must have the same number of observations."
        )

    if not np.all(np.isfinite(E)):
        raise ValueError("E must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate the temperature.
    # ---------------------------------------------------------

    try:
        tau = float(tau)
    except (TypeError, ValueError):
        raise ValueError("tau must be a positive finite number.")

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number.")

    # ---------------------------------------------------------
    # 3. Find all ordered positive pairs.
    # ---------------------------------------------------------

    # A positive pair (i, j) satisfies:
    #
    # i != j and y[i] == y[j]
    #
    # Both (i, j) and (j, i) are included.

    same_class = y[:, None] == y[None, :]

    # Exclude self-pairs.
    np.fill_diagonal(same_class, False)

    # Obtain all ordered positive pairs.
    anchors, positives = np.where(same_class)

    n_pairs = len(anchors)

    if n_pairs == 0:
        raise ValueError(
            "The batch must contain at least one ordered positive pair."
        )

    # Every anchor must have at least one negative observation.
    negative_mask = y[anchors, None] != y[None, :]

    if not np.all(np.any(negative_mask, axis=1)):
        raise ValueError(
            "Every anchor must have at least one negative observation."
        )

    # ---------------------------------------------------------
    # 4. Compute the positive-pair similarity.
    # ---------------------------------------------------------

    # Similarity between each anchor and its corresponding
    # positive observation:
    #
    # s_ij = e_i.T @ e_j

    positive_similarity = np.sum(
        E[anchors] * E[positives],
        axis=1
    )

    # ---------------------------------------------------------
    # 5. Compute all anchor-negative similarities.
    # ---------------------------------------------------------

    # Similarities between every anchor and every batch point:
    #
    # s_ik = e_i.T @ e_k

    all_similarity = E[anchors] @ E.T

    # ---------------------------------------------------------
    # 6. Compute the N-pairs logits.
    # ---------------------------------------------------------

    # For each ordered positive pair (i, j), and negative k:
    #
    # z_ijk = (e_i.T @ e_k - e_i.T @ e_j) / tau

    logits = (
        all_similarity
        - positive_similarity[:, None]
    ) / tau

    # Only observations belonging to a different class
    # from the anchor are negatives.
    #
    # All other entries are excluded from the loss.

    logits = np.where(
        negative_mask,
        logits,
        -np.inf
    )

    # ---------------------------------------------------------
    # 7. Calculate the loss using stable log-sum-exp.
    # ---------------------------------------------------------

    # l_ij = log(1 + sum_k exp(z_ijk))
    #
    # Include the constant 1 as a zero logit.
    #
    # For numerical stability, subtract the maximum between
    # zero and all valid negative logits.

    row_max = np.maximum(
        0.0,
        np.max(logits, axis=1)
    )

    # Exponentiate only shifted logits.
    # Invalid negative positions remain zero.
    shifted_exp = np.exp(
        logits - row_max[:, None]
    )

    # Denominator after exponent shifting:
    #
    # exp(-row_max) + sum_k exp(logit_k - row_max)

    denominator = (
        np.exp(-row_max)
        + np.sum(shifted_exp, axis=1)
    )

    # Recover the log denominator.
    log_denominator = (
        row_max + np.log(denominator)
    )

    # Mean N-pairs loss over all ordered positive pairs.
    loss = float(np.mean(log_denominator))

    # ---------------------------------------------------------
    # 8. Compute the negative probabilities.
    # ---------------------------------------------------------

    # q_ijk = exp(z_ijk) / (1 + sum_k exp(z_ijk))
    #
    # Use the shifted exponentials to avoid numerical
    # overflow and underflow.

    q = shifted_exp / denominator[:, None]

    # Sum of negative probabilities for every positive pair.
    q_sum = np.sum(
        q,
        axis=1
    )

    # ---------------------------------------------------------
    # 9. Calculate the gradient contribution for anchors.
    # ---------------------------------------------------------

    # For each ordered pair (i, j):
    #
    # d l_ij / d e_i =
    #     (1/tau) * sum_k q_ijk * (e_k - e_j)

    grad_anchor = (
        q @ E
        - q_sum[:, None] * E[positives]
    ) / tau

    # ---------------------------------------------------------
    # 10. Calculate the gradient contribution for positives.
    # ---------------------------------------------------------

    # d l_ij / d e_j =
    #     -(1/tau) * sum_k q_ijk * e_i

    grad_positive = (
        -q_sum[:, None] * E[anchors]
    ) / tau

    # ---------------------------------------------------------
    # 11. Calculate the gradient contribution for negatives.
    # ---------------------------------------------------------

    # d l_ij / d e_k =
    #     q_ijk * e_i / tau

    # Summing over all ordered positive pairs gives:
    #
    # G_negative = q.T @ E[anchors] / tau

    grad_negative = (
        q.T @ E[anchors]
    ) / tau

    # ---------------------------------------------------------
    # 12. Accumulate all gradient contributions.
    # ---------------------------------------------------------

    G = np.zeros_like(E)

    # Multiple pairs can share the same anchor or positive.
    # Use np.add.at to accumulate all contributions correctly.

    np.add.at(
        G,
        anchors,
        grad_anchor
    )

    np.add.at(
        G,
        positives,
        grad_positive
    )

    # The matrix multiplication already accumulates all
    # negative-role contributions for each observation.
    G += grad_negative

    # ---------------------------------------------------------
    # 13. Average the gradient over ordered positive pairs.
    # ---------------------------------------------------------

    G /= n_pairs

    return loss, G

# Step 23 - centroid_loss
def centroid_loss(E, y, centroids):
    """Return (loss, G): the per-point bracket of the centroid
    upper-bound loss (eq 16.22), averaged over points, and its
    gradient with respect to E.

    Parameters
    ----------
    E : ndarray, shape (N, L)
        Embedding matrix.

    y : ndarray, shape (N,)
        Integer class labels from 0 to C-1.

    centroids : ndarray, shape (C, L)
        Class centroids, with one centroid per class.

    Returns
    -------
    loss : float
        Mean centroid loss over all observations.

    G : ndarray, shape (N, L)
        Exact gradient with respect to E.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    E = np.asarray(E, dtype=float)
    y = np.asarray(y)
    centroids = np.asarray(centroids, dtype=float)

    # ---------------------------------------------------------
    # 2. Validate input dimensions.
    # ---------------------------------------------------------

    if E.ndim != 2:
        raise ValueError("E must be a 2D array.")

    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    N, L = E.shape
    C, centroid_L = centroids.shape

    if N == 0 or L == 0:
        raise ValueError("E must contain observations and features.")

    if C < 2:
        raise ValueError("At least two class centroids are required.")

    if centroid_L != L:
        raise ValueError(
            "E and centroids must have the same embedding dimension."
        )

    if len(y) != N:
        raise ValueError(
            "E and y must have the same number of observations."
        )

    # ---------------------------------------------------------
    # 3. Validate the input values and labels.
    # ---------------------------------------------------------

    if not (
        np.all(np.isfinite(E))
        and np.all(np.isfinite(centroids))
    ):
        raise ValueError("E and centroids must contain finite values.")

    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integer class labels.")

    if np.any(y < 0) or np.any(y >= C):
        raise ValueError("Labels must be between 0 and C-1.")

    # ---------------------------------------------------------
    # 4. Compute distances from every embedding to every centroid.
    # ---------------------------------------------------------

    # diff[i, c] = E[i] - centroids[c]
    #
    # Shape: (N, C, L)

    diff = E[:, None, :] - centroids[None, :, :]

    # Euclidean distances.
    # Shape: (N, C)

    distances = np.linalg.norm(diff, axis=2)

    # ---------------------------------------------------------
    # 5. Extract the distance to each point's own centroid.
    # ---------------------------------------------------------

    rows = np.arange(N)

    own_distances = distances[rows, y]

    # ---------------------------------------------------------
    # 6. Calculate distances to all other class centroids.
    # ---------------------------------------------------------

    other_mask = np.ones((N, C), dtype=bool)

    other_mask[rows, y] = False

    other_distance_sum = np.sum(
        np.where(other_mask, distances, 0.0),
        axis=1
    )

    # ---------------------------------------------------------
    # 7. Calculate the correctly normalized centroid loss.
    # ---------------------------------------------------------

    # IMPORTANT:
    # Equation 16.22 uses:
    #
    # alpha = 1 / (3 * (C - 1))
    #
    # There is NO additional factor of C in the denominator.

    alpha = 1.0 / (3.0 * (C - 1))

    # Per-point loss:
    #
    # L_i = ||e_i - c_yi||
    #       - alpha * sum_{m != yi} ||e_i - c_m||

    point_losses = (
        own_distances
        - alpha * other_distance_sum
    )

    # Mean over all observations.
    loss = float(np.mean(point_losses))

    # ---------------------------------------------------------
    # 8. Calculate unit direction vectors.
    # ---------------------------------------------------------

    # For a nonzero distance:
    #
    # d ||e_i - c_m|| / d e_i
    #     = (e_i - c_m) / ||e_i - c_m||
    #
    # At exactly zero distance, use a zero gradient contribution.

    directions = np.zeros_like(diff)

    np.divide(
        diff,
        distances[:, :, None],
        out=directions,
        where=distances[:, :, None] > 0.0
    )

    # ---------------------------------------------------------
    # 9. Gradient contribution from own centroids.
    # ---------------------------------------------------------

    own_directions = directions[rows, y]

    # ---------------------------------------------------------
    # 10. Gradient contribution from other centroids.
    # ---------------------------------------------------------

    other_directions = np.where(
        other_mask[:, :, None],
        directions,
        0.0
    )

    other_direction_sum = np.sum(
        other_directions,
        axis=1
    )

    # ---------------------------------------------------------
    # 11. Combine gradient contributions.
    # ---------------------------------------------------------

    # dL/dE_i =
    #     (1/N) * (
    #         own_direction
    #         - alpha * sum(other_directions)
    #     )

    G = (
        own_directions
        - alpha * other_direction_sum
    ) / N

    return loss, G

# Step 24 - train_embedding
def pk_batch(y, n_classes, n_per_class, rng):
    """Return indices of a class-balanced P-K batch.

    Select min(n_classes, number of available classes) distinct
    classes, then sample n_per_class different observations
    from each selected class.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Original class labels. Labels need not be contiguous.

    n_classes : int
        Maximum number of distinct classes to select.

    n_per_class : int
        Number of observations sampled from each selected class.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    indices : ndarray, shape (P * K,)
        Indices of the selected training observations.
    """

    # Convert labels to a NumPy array.
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if y.size == 0:
        raise ValueError("y must contain at least one observation.")

    # Validate the requested number of classes.
    if (
        isinstance(n_classes, (bool, np.bool_))
        or not isinstance(n_classes, (int, np.integer))
        or n_classes < 1
    ):
        raise ValueError("n_classes must be a positive integer.")

    # Validate the number of observations per class.
    if (
        isinstance(n_per_class, (bool, np.bool_))
        or not isinstance(n_per_class, (int, np.integer))
        or n_per_class < 1
    ):
        raise ValueError("n_per_class must be a positive integer.")

    # ---------------------------------------------------------
    # 1. Identify all available classes.
    # ---------------------------------------------------------

    classes = np.unique(y)

    # Select the requested number of classes, without replacement.
    P = min(int(n_classes), len(classes))

    selected_classes = rng.choice(
        classes,
        size=P,
        replace=False
    )

    # ---------------------------------------------------------
    # 2. Sample n_per_class observations from each class.
    # ---------------------------------------------------------

    batch_indices = []

    for c in selected_classes:

        # Find all training indices belonging to class c.
        class_indices = np.flatnonzero(y == c)

        if len(class_indices) < n_per_class:
            raise ValueError(
                f"Class {c} contains fewer than n_per_class observations."
            )

        # Sample distinct observations without replacement.
        sampled_indices = rng.choice(
            class_indices,
            size=n_per_class,
            replace=False
        )

        batch_indices.append(sampled_indices)

    # Concatenate the selected observations into one batch.
    return np.concatenate(batch_indices).astype(int)


def sphere_gradient(W, X, G):
    """Apply the chain rule through row-wise unit normalization.

    Given:
        Z = X @ W.T
        E_hat = unit_rows(Z)
        G = dL/dE_hat

    Return:
        dL/dW

    Parameters
    ----------
    W : ndarray, shape (L, D)
        Linear transformation matrix.

    X : ndarray, shape (N, D)
        Input observations.

    G : ndarray, shape (N, L)
        Gradient with respect to normalized embeddings.

    Returns
    -------
    grad_W : ndarray, shape (L, D)
        Gradient with respect to W.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate inputs.
    # ---------------------------------------------------------

    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    G = np.asarray(G, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if G.ndim != 2:
        raise ValueError("G must be a 2D array.")

    L, D = W.shape
    N = X.shape[0]

    if X.shape[1] != D:
        raise ValueError("X and W have incompatible feature dimensions.")

    if G.shape != (N, L):
        raise ValueError("G must have shape (N, L).")

    if not (
        np.all(np.isfinite(W))
        and np.all(np.isfinite(X))
        and np.all(np.isfinite(G))
    ):
        raise ValueError("W, X and G must contain finite values.")

    # ---------------------------------------------------------
    # 2. Compute the unnormalized embeddings.
    # ---------------------------------------------------------

    # Z = X @ W.T
    #
    # Each row contains one transformed observation.

    Z = X @ W.T

    # ---------------------------------------------------------
    # 3. Normalize every embedding using the earlier unit_rows().
    # ---------------------------------------------------------

    E_hat, norms = unit_rows(Z)

    # Use the same minimum divisor as unit_rows().
    eps = 1e-12

    safe_norms = np.maximum(norms, eps)

    # ---------------------------------------------------------
    # 4. Apply the derivative of L2 normalization.
    # ---------------------------------------------------------

    # For a nonzero embedding with norm > eps:
    #
    # e_hat = z / ||z||
    #
    # dL/dz = (G - (G . e_hat) * e_hat) / ||z||
    #
    # For norms <= eps, unit_rows uses a constant divisor eps.
    # Its Jacobian is therefore I / eps.

    grad_Z = np.empty_like(G)

    small_norm = norms[:, 0] <= eps
    regular_norm = ~small_norm

    # Handle ordinary nonzero embeddings.
    if np.any(regular_norm):

        G_regular = G[regular_norm]
        E_regular = E_hat[regular_norm]

        norm_regular = safe_norms[regular_norm]

        # Projection of G onto the normalized embedding.
        projection = np.sum(
            G_regular * E_regular,
            axis=1,
            keepdims=True
        )

        grad_Z[regular_norm] = (
            G_regular - projection * E_regular
        ) / norm_regular

    # Handle zero or extremely small embeddings.
    if np.any(small_norm):

        grad_Z[small_norm] = G[small_norm] / eps

    # ---------------------------------------------------------
    # 5. Apply the chain rule through the linear transformation.
    # ---------------------------------------------------------

    # Z = X @ W.T
    #
    # Therefore:
    #
    # dL/dW = (dL/dZ).T @ X

    grad_W = grad_Z.T @ X

    return grad_W


def train_embedding(
    X,
    y,
    W0,
    loss="triplet",
    steps=300,
    lr=0.003,
    m=0.2,
    tau=0.1,
    mine="semi-hard",
    n_classes=10,
    n_per_class=2,
    seed=0
):
    """Train a linear embedding on the unit sphere using class-balanced
    mini-batches and one of the supported ranking losses.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input training observations.

    y : ndarray, shape (N,)
        Original class labels. Labels may be non-contiguous.

    W0 : ndarray, shape (L, D)
        Initial linear transformation matrix.

    loss : str, default="triplet"
        Ranking loss:
            'contrastive'
            'triplet'
            'npairs'
            'centroid'

    steps : int, default=300
        Number of Adam optimization iterations.

    lr : float, default=0.003
        Adam learning rate.

    m : float, default=0.2
        Contrastive or triplet margin.

    tau : float, default=0.1
        N-pairs temperature.

    mine : str, default="semi-hard"
        Triplet mining strategy.

    n_classes : int, default=10
        Maximum number of classes per mini-batch.

    n_per_class : int, default=2
        Number of observations sampled per selected class.

    seed : int, default=0
        Random seed.

    Returns
    -------
    W : ndarray, shape (L, D)
        Optimized linear transformation matrix.

    history : list of float
        Training objective values recorded by Adam.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate the input data.
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    W0 = np.asarray(W0, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")

    if W0.ndim != 2:
        raise ValueError("W0 must be a 2D array.")

    N, D = X.shape
    L, W_D = W0.shape

    if N == 0 or D == 0:
        raise ValueError("X must contain observations and features.")

    if L == 0:
        raise ValueError("W0 must contain at least one embedding dimension.")

    if len(y) != N:
        raise ValueError("X and y must have the same number of observations.")

    if W_D != D:
        raise ValueError("W0 must have shape (L, D).")

    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(W0))
    ):
        raise ValueError("X and W0 must contain finite values.")

    # ---------------------------------------------------------
    # 2. Validate the selected loss.
    # ---------------------------------------------------------

    valid_losses = (
        "contrastive",
        "triplet",
        "npairs",
        "centroid"
    )

    if loss not in valid_losses:
        raise ValueError(
            "Unknown loss. Choose 'contrastive', 'triplet', "
            "'npairs', or 'centroid'."
        )

    # ---------------------------------------------------------
    # 3. Map original labels to contiguous integer labels.
    # ---------------------------------------------------------

    # Real identity IDs may not be contiguous.
    #
    # Example:
    # Original labels: [10, 10, 25, 25, 40, 40]
    # Mapped labels:   [ 0,  0,  1,  1,  2,  2]

    unique_labels, y_mapped = np.unique(
        y,
        return_inverse=True
    )

    C = len(unique_labels)

    if C < 2:
        raise ValueError("At least two distinct classes are required.")

    # ---------------------------------------------------------
    # 4. Validate the batch configuration.
    # ---------------------------------------------------------

    if (
        isinstance(n_classes, (bool, np.bool_))
        or not isinstance(n_classes, (int, np.integer))
        or n_classes < 1
    ):
        raise ValueError("n_classes must be a positive integer.")

    if (
        isinstance(n_per_class, (bool, np.bool_))
        or not isinstance(n_per_class, (int, np.integer))
        or n_per_class < 1
    ):
        raise ValueError("n_per_class must be a positive integer.")

    # ---------------------------------------------------------
    # 5. Validate the centroid embedding dimension.
    # ---------------------------------------------------------

    # Centroid loss uses one-hot class centroids:
    #
    # centroids = np.eye(C)
    #
    # Therefore, the embedding dimension must equal C.

    if loss == "centroid" and L != C:
        raise ValueError(
            "For centroid loss, W0 must have exactly C rows, "
            "where C is the number of distinct classes."
        )

    # ---------------------------------------------------------
    # 6. Create one random number generator for the whole call.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 7. Define the stochastic mini-batch objective.
    # ---------------------------------------------------------

    def objective(W):
        """Draw a fresh class-balanced batch and evaluate its loss."""

        # Draw exactly one new batch per objective evaluation.
        batch_indices = pk_batch(
            y_mapped,
            n_classes,
            n_per_class,
            rng
        )

        X_batch = X[batch_indices]
        y_batch = y_mapped[batch_indices]

        # -----------------------------------------------------
        # 7a. Compute and normalize the batch embeddings.
        # -----------------------------------------------------

        # Z = X_batch @ W.T
        Z = X_batch @ W.T

        # Project every embedding onto the unit sphere.
        E_hat, _ = unit_rows(Z)

        # -----------------------------------------------------
        # 7b. Compute the selected ranking loss.
        # -----------------------------------------------------

        if loss == "contrastive":

            batch_loss, G = contrastive_loss(
                E_hat,
                y_batch,
                m=m
            )

        elif loss == "triplet":

            # Mine triplets using the selected mining strategy.
            triplets = mine_triplets(
                E_hat,
                y_batch,
                m=m,
                mode=mine
            )

            batch_loss, G = triplet_loss(
                E_hat,
                triplets,
                m=m
            )

        elif loss == "npairs":

            batch_loss, G = npairs_loss(
                E_hat,
                y_batch,
                tau=tau
            )

        else:  # loss == "centroid"

            # One-hot centroids for all C original classes.
            centroids = np.eye(C)

            batch_loss, G = centroid_loss(
                E_hat,
                y_batch,
                centroids
            )

        # -----------------------------------------------------
        # 7c. Backpropagate through unit-sphere normalization.
        # -----------------------------------------------------

        grad_W = sphere_gradient(
            W,
            X_batch,
            G
        )

        return batch_loss, grad_W

    # ---------------------------------------------------------
    # 8. Optimize the transformation matrix using Adam.
    # ---------------------------------------------------------

    W, history = fit_linear_metric(
        objective,
        W0,
        steps=steps,
        lr=lr
    )

    return W, history

# Step 25 - open_set_predict
def open_set_predict(gallery, gallery_ids, probes, tau):
    """Return (pred, scores): the best cosine match per probe,
    with pred = -1 when the best similarity is below tau.

    Parameters
    ----------
    gallery : ndarray, shape (N_gallery, L)
        Gallery embeddings.

    gallery_ids : ndarray, shape (N_gallery,)
        Identity ID associated with each gallery row.

    probes : ndarray, shape (N_probes, L)
        Probe embeddings.

    tau : float
        Cosine similarity rejection threshold.
        Positive and negative infinity are allowed.

    Returns
    -------
    pred : ndarray, shape (N_probes,)
        Predicted identity IDs, or -1 for rejected probes.

    scores : ndarray, shape (N_probes,)
        Maximum cosine similarity for each probe.
    """

    # ---------------------------------------------------------
    # 1. Convert inputs to NumPy arrays.
    # ---------------------------------------------------------

    gallery = np.asarray(gallery, dtype=float)
    probes = np.asarray(probes, dtype=float)
    gallery_ids = np.asarray(gallery_ids)

    if gallery.ndim != 2:
        raise ValueError("gallery must be a 2D array.")

    if probes.ndim != 2:
        raise ValueError("probes must be a 2D array.")

    if gallery_ids.ndim != 1:
        raise ValueError("gallery_ids must be a 1D array.")

    # ---------------------------------------------------------
    # 2. Validate dimensions and input values.
    # ---------------------------------------------------------

    n_gallery, L = gallery.shape
    n_probes = probes.shape[0]

    if n_gallery == 0 or L == 0:
        raise ValueError("gallery must contain at least one embedding.")

    if len(gallery_ids) != n_gallery:
        raise ValueError(
            "gallery_ids must contain one ID per gallery embedding."
        )

    if probes.shape[1] != L:
        raise ValueError(
            "gallery and probes must have the same embedding dimension."
        )

    if not np.all(np.isfinite(gallery)):
        raise ValueError("gallery must contain finite values.")

    if not np.all(np.isfinite(probes)):
        raise ValueError("probes must contain finite values.")

    if not np.issubdtype(gallery_ids.dtype, np.integer):
        raise ValueError("gallery_ids must contain integer IDs.")

    if np.any(gallery_ids == -1):
        raise ValueError("-1 is reserved for rejected predictions.")

    # ---------------------------------------------------------
    # 3. Validate the threshold.
    # ---------------------------------------------------------

    try:
        tau = float(tau)
    except (TypeError, ValueError):
        raise ValueError("tau must be a numeric scalar.")

    # IMPORTANT:
    # Allow +inf and -inf because they are useful for
    # accepting all probes or rejecting all probes.
    # Only NaN is invalid.
    if np.isnan(tau):
        raise ValueError("tau must not be NaN.")

    # ---------------------------------------------------------
    # 4. Normalize gallery and probe embeddings.
    # ---------------------------------------------------------

    # Reuse unit_rows() from Step 19.
    gallery_hat, _ = unit_rows(gallery)
    probes_hat, _ = unit_rows(probes)

    # ---------------------------------------------------------
    # 5. Compute cosine similarities.
    # ---------------------------------------------------------

    # For normalized vectors, cosine similarity is their
    # ordinary dot product.
    #
    # Shape: (n_probes, n_gallery)

    similarities = probes_hat @ gallery_hat.T

    # Protect against floating-point rounding.
    similarities = np.clip(
        similarities,
        -1.0,
        1.0
    )

    # ---------------------------------------------------------
    # 6. Find the best matching gallery row.
    # ---------------------------------------------------------

    # np.argmax returns the first occurrence of the maximum,
    # so the earlier gallery row wins in a tie.

    best_indices = np.argmax(
        similarities,
        axis=1
    )

    scores = similarities[
        np.arange(n_probes),
        best_indices
    ]

    # ---------------------------------------------------------
    # 7. Apply the rejection threshold.
    # ---------------------------------------------------------

    # Accept when score >= tau.
    # Reject when score < tau.

    accepted = scores >= tau

    pred = np.full(
        n_probes,
        -1,
        dtype=np.int64
    )

    pred[accepted] = gallery_ids[
        best_indices[accepted]
    ].astype(np.int64)

    return pred, scores


def open_set_metrics(pred, true):
    """Return a dictionary containing DIR, FAR and balanced score.

    A true identity of -1 represents a non-enrolled person.

    Parameters
    ----------
    pred : ndarray, shape (N,)
        Predicted identity IDs. -1 indicates rejection.

    true : ndarray, shape (N,)
        Actual identity IDs. -1 indicates a non-enrolled person.

    Returns
    -------
    dict
        'dir'      : Detection and identification rate.
        'far'      : False acceptance rate.
        'balanced' : 0.5 * (dir + 1 - far).
    """

    pred = np.asarray(pred)
    true = np.asarray(true)

    # Validate input dimensions.
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError("pred and true must be 1D arrays.")

    if len(pred) != len(true):
        raise ValueError(
            "pred and true must have the same number of observations."
        )

    if not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("pred must contain integer identity IDs.")

    if not np.issubdtype(true.dtype, np.integer):
        raise ValueError("true must contain integer identity IDs.")

    if len(true) == 0:
        raise ValueError("At least one probe is required.")

    # Separate enrolled and non-enrolled probes.
    enrolled = true != -1
    non_enrolled = true == -1

    # Both types of probes must be present.
    if not np.any(enrolled):
        raise ValueError("At least one enrolled probe is required.")

    if not np.any(non_enrolled):
        raise ValueError("At least one non-enrolled probe is required.")

    # Detection and identification rate:
    # Correctly identified enrolled probes / all enrolled probes.
    dir_rate = np.mean(
        pred[enrolled] == true[enrolled]
    )

    # False acceptance rate:
    # Accepted non-enrolled probes / all non-enrolled probes.
    far_rate = np.mean(
        pred[non_enrolled] != -1
    )

    # Balanced score.
    balanced = 0.5 * (
        dir_rate + 1.0 - far_rate
    )

    return {
        "dir": float(dir_rate),
        "far": float(far_rate),
        "balanced": float(balanced)
    }

# Step 26 - tune_threshold
def tune_threshold(gallery, gallery_ids, probes, true):
    """Return (tau, best): the threshold maximizing the balanced
    open-set recognition score on a calibration set.

    Parameters
    ----------
    gallery : ndarray, shape (N_gallery, L)
        Embeddings of enrolled identities.

    gallery_ids : ndarray, shape (N_gallery,)
        Identity IDs corresponding to gallery embeddings.

    probes : ndarray, shape (N_probes, L)
        Calibration probe embeddings.

    true : ndarray, shape (N_probes,)
        Actual identity IDs. A value of -1 indicates a
        non-enrolled person.

    Returns
    -------
    tau : float
        Selected rejection threshold. If multiple candidates
        achieve the best score within 1e-12, their median
        threshold is returned.

    best : float
        Highest balanced score achieved on the calibration set.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate the true identity labels.
    # ---------------------------------------------------------

    true = np.asarray(true)

    if true.ndim != 1:
        raise ValueError("true must be a 1D array.")

    if not np.issubdtype(true.dtype, np.integer):
        raise ValueError("true must contain integer identity IDs.")

    if true.size == 0:
        raise ValueError("The calibration set must not be empty.")

    # Both enrolled and non-enrolled probes are required.
    if not np.any(true != -1):
        raise ValueError("At least one enrolled probe is required.")

    if not np.any(true == -1):
        raise ValueError("At least one non-enrolled probe is required.")

    # ---------------------------------------------------------
    # 2. Compute the best gallery match and cosine score ONCE.
    # ---------------------------------------------------------

    # tau = -inf ensures that every probe is accepted.
    #
    # We retain these predictions as the best gallery identity
    # for each probe. Candidate thresholds will only change
    # whether a probe is accepted or rejected.
    #
    # Similarities must NOT be recomputed for each candidate.

    best_ids, scores = open_set_predict(
        gallery,
        gallery_ids,
        probes,
        -np.inf
    )

    # ---------------------------------------------------------
    # 3. Find the distinct cosine similarity scores.
    # ---------------------------------------------------------

    distinct_scores = np.unique(scores)

    if distinct_scores.size == 0:
        raise ValueError("No calibration similarity scores are available.")

    # ---------------------------------------------------------
    # 4. Construct all candidate thresholds.
    # ---------------------------------------------------------

    # Candidate 1:
    # 1e-6 below the smallest distinct similarity score.

    lower_threshold = distinct_scores[0] - 1e-6

    # Candidate thresholds:
    # Midpoints between consecutive distinct scores.

    midpoints = (
        distinct_scores[:-1] + distinct_scores[1:]
    ) / 2.0

    # Final candidate:
    # 1e-6 above the largest distinct similarity score.

    upper_threshold = distinct_scores[-1] + 1e-6

    # Combine all candidates in ascending order.
    candidates = np.concatenate((
        np.array([lower_threshold]),
        midpoints,
        np.array([upper_threshold])
    ))

    # ---------------------------------------------------------
    # 5. Evaluate the balanced score for every candidate.
    # ---------------------------------------------------------

    balanced_scores = np.empty(
        len(candidates),
        dtype=float
    )

    for i, threshold in enumerate(candidates):

        # Reuse the previously computed best gallery IDs.
        # No additional cosine similarity computation is needed.

        pred = np.where(
            scores >= threshold,
            best_ids,
            -1
        )

        # Evaluate the open-set metrics.
        metrics = open_set_metrics(
            pred,
            true
        )

        balanced_scores[i] = metrics["balanced"]

    # ---------------------------------------------------------
    # 6. Find the highest balanced score.
    # ---------------------------------------------------------

    best = float(np.max(balanced_scores))

    # ---------------------------------------------------------
    # 7. Handle multiple thresholds achieving the best score.
    # ---------------------------------------------------------

    # Include all candidate thresholds whose balanced scores
    # are within 1e-12 of the maximum.

    tied = np.abs(
        balanced_scores - best
    ) <= 1e-12

    # Select the median threshold among tied candidates.
    tau = float(np.median(candidates[tied]))

    return tau, best

