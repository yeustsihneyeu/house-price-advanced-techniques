import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, x1, x2, name):
        self.x1 = x1
        self.x2 = x2
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X[self.x1] - X[self.x2]).to_frame(self.name)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.name])