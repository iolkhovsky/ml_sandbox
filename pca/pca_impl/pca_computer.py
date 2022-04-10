import numpy as np

from .utils import covariance_matrix, standartize_2d_matrix


class PCAComputer:
    def __init__(self, n_components) -> None:
        self.n_components_ = n_components
        self.principal_components_ = None
        self.eigen_values_ = None
    
    def fit(self, X):
        n_components = min(self.n_components_, X.shape[1])
        cov_mat = covariance_matrix(X)
        # covariance matrix is symmetric by definition so we may use eigh instead eig
        eigen_values, eigen_vectors_T = np.linalg.eigh(cov_mat)
        # Also we may use SVD to extract eigen values and vectors
        # directly as cov_mat is symmetric:
        # vectorsT, values, vectorsT = np.linalg.svd(cov_mat)
        self.eigen_values_, sorted_vectors = zip(
            *sorted(
                zip(eigen_values, eigen_vectors_T.T),
                key=lambda x: x[0],
                reverse=True)
        )
        self.principal_components_ = np.vstack(sorted_vectors[:n_components])
    
    def transform(self, X):
        assert self.principal_components_ is not None
        X_centered = standartize_2d_matrix(X, scale=False)
        X_transformed = np.transpose(
            np.dot(self.principal_components_, X_centered.T))
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __call__(self, X):
        return self.transform(X)
    
    def __str__(self) -> str:
        return f"PCA computer for {self.n_components_} components"
