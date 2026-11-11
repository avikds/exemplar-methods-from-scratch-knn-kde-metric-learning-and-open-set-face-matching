"""
Loads the Olivetti Faces dataset using scikit-learn.
"""

from sklearn.datasets import fetch_olivetti_faces


def load(name):
    """
    Load a supported dataset.

    Parameters
    ----------
    name : str
        Dataset name. Currently supports 'olivetti-faces'.

    Returns
    -------
    sklearn.utils.Bunch
        Dataset containing .data and .target.
    """

    if name not in ("olivetti-faces", "olivetti_faces"):
        raise ValueError(f"Unknown dataset: {name}")

    return fetch_olivetti_faces(
        data_home=None,
        shuffle=False,
        download_if_missing=True
    )