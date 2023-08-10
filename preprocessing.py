import numpy as np

from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler


class LogitScaler(MinMaxScaler):
    """Preprocessing scaler that performs a logit transformation on top
    of the sklean MinMaxScaler. It scales to a range [0+epsilon, 1-epsilon]
    before applying the logit. Setting a small finitie epsilon avoids
    features being mapped to exactly 0 and 1 before the logit is applied.
    If the logit does encounter values beyond (0, 1), it outputs nan for
    these values.
    """

    _parameter_constraints: dict = {
        "epsilon": ["float"],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, epsilon=0, copy=True, clip=False):
        self.epsilon = epsilon
        self.copy = copy
        self.clip = clip
        super().__init__(feature_range=(0+epsilon, 1-epsilon),
                         copy=copy, clip=clip)

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        z = logit(super().transform(X))
        z[np.isinf(z)] = np.nan
        return z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return super().inverse_transform(expit(X))
