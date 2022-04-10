import numpy as np


def generate_correlated_data():
    rng = np.random.RandomState(0)
    n_samples = 500
    cov = [[3, 4], [3, 3]]
    X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
    return X


def standardize_vector(x, scale=True):
    assert isinstance(x, np.ndarray) and len(x.shape) == 1
    centered = x - np.mean(x, axis=0)
    if scale:
        return centered / np.std(x, axis=0)
    else:
        return centered

def standartize_2d_matrix(X, scale=True):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2
    out = np.zeros_like(X)
    features_cnt = X.shape[1]
    for idx in range(features_cnt):
        out.T[idx] = standardize_vector(X.T[idx], scale=scale)
    return out


def covariance(x1, x2, standartize=True):
    assert isinstance(x1, np.ndarray)
    assert isinstance(x2, np.ndarray)
    assert x1.shape == x2.shape and len(x1.shape) == 1
    sample_length = len(x1)
    assert sample_length
    if standartize:
        x1 = standardize_vector(x1, scale=False)
        x2 = standardize_vector(x2, scale=False)
    return np.dot(x1, x2) / sample_length


def covariance_matrix(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2
    sample_size, features = X.shape
    assert sample_size and features
    out = np.zeros(shape=(features, features), dtype=np.float32)
    for j in range(features):
        for i in range(j, features):
            out[j, i] = covariance(X[:, i], X[:, j])
        for i in range(j):
            out[j, i] = out[i, j]
    return out
