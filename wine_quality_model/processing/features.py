from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Пример кастомного трансформера (если нужен)
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)