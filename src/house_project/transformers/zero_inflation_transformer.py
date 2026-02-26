import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroInflationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, prefix="has_"):
        self.prefix = prefix

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = np.asarray(X)
        indicator = (X != 0).astype(int)
        return np.hstack([indicator])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix}{f}" for f in input_features], dtype=object)