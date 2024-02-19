import numpy as np

from numbers import Real
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, _name_estimators, _final_estimator_has

from sklearn.utils.metadata_routing import (_routing_enabled,
                                            _raise_for_params,
                                            process_routing,)
from sklearn.utils.metaestimators import available_if


class LogitScaler(MinMaxScaler):
    """Preprocessing scaler that performs a logit transformation on top
    of the sklean MinMaxScaler. It scales to a range [0+epsilon, 1-epsilon]
    before applying the logit. Setting a small finitie epsilon avoids
    features being mapped to exactly 0 and 1 before the logit is applied.
    If the logit does encounter values beyond (0, 1), it outputs nan for
    these values.
    """

    _parameter_constraints: dict = {
        "epsilon": [Real],
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

    def jacobian_determinant(self, X):
        z = super().transform(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) / np.prod(self.scale_)

    def log_jacobian_determinant(self, X):
        z = super().transform(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) - np.sum(np.log(self.scale_))

    def inverse_jacobian_determinant(self, X):
        z = expit(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) * np.prod(self.scale_)

    def inverse_log_jacobian_determinant(self, X):
        z = expit(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) + np.sum(np.log(self.scale_))


class ExtStandardScaler(StandardScaler):
    """StandardScaler with additional methods for computing the jacobian
    determinant of the transformation.
    """

    def jacobian_determinant(self, X):
        return np.ones((len(X), 1)) / np.prod(self.scale_)

    def log_jacobian_determinant(self, X):
        return -np.sum(np.log(self.scale_)) * np.ones((len(X), 1))

    def inverse_jacobian_determinant(self, X):
        return np.ones((len(X), 1)) * np.prod(self.scale_)

    def inverse_log_jacobian_determinant(self, X):
        return np.sum(np.log(self.scale_)) * np.ones((len(X), 1))


class ExtPipeline(Pipeline):
    """Pipeline with additional methods for computing the jacobian
    determinant of the transformation. Calling predict(_log)_proba
    on the pipeline will coimpute the probabilities with the last
    estimator (as the sklearn pipeline does) but normalizes them
    with the jacobian determinant of the intermediate transformations.
    Also, it has a sample method, which samples from the last estimator
    and inverse_transforms the sample through the intermediate transformations.

    NOTE: the sampling method does not yet support metadata routing.
    """

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        Xt = X
        log_jac_det = np.zeros((len(X), 1))

        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                log_jac_det += transform.log_jacobian_determinant(Xt)
                Xt = transform.transform(Xt)
            raw_log_proba = self.steps[-1][1].predict_log_proba(Xt, **params)
            return raw_log_proba + log_jac_det

        # metadata routing enabled
        routed_params = process_routing(self, "predict_log_proba", **params)
        for _, name, transform in self._iter(with_final=False):
            log_jac_det += transform.log_jacobian_determinant(
                Xt, **routed_params[name].transform)
            Xt = transform.transform(Xt, **routed_params[name].transform)
        raw_log_proba = self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )
        return raw_log_proba + log_jac_det

    def predict_proba(self, X, **params):
        return np.exp(self.predict_log_proba(X, **params))

    def jacobian_determinant(self, X, **params):
        return np.exp(self.log_jacobian_determinant(X, **params))

    def log_jacobian_determinant(self, X, **params):
        Xt = X
        log_jac_det = np.zeros((len(X), 1))

        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=True):
                log_jac_det += transform.log_jacobian_determinant(Xt)
                Xt = transform.transform(Xt)
            return log_jac_det

        # metadata routing enabled
        routed_params = process_routing(self, "transform", **params)
        for _, name, transform in self._iter(with_final=True):
            log_jac_det += transform.log_jacobian_determinant(
                Xt, **routed_params[name].transform)
            Xt = transform.transform(Xt, **routed_params[name].transform)
        return log_jac_det

    def inverse_jacobian_determinant(self, Xt, **params):
        return np.exp(self.log_inverse_jacobian_determinant(Xt, **params))

    def log_inverse_jacobian_determinant(self, Xt, **params):
        _raise_for_params(params, self, "inverse_transform")
        log_jac_det = np.zeros((len(Xt), 1))

        # we don't have to branch here, since params is only non-empty if
        # enable_metadata_routing=True.
        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter()))
        for _, name, transform in reverse_iter:
            log_jac_det += transform.log_inverse_jacobian_determinant(
                Xt, **routed_params[name].inverse_transform)
            Xt = transform.inverse_transform(
                Xt, **routed_params[name].inverse_transform
            )
        return log_jac_det

    @available_if(_final_estimator_has("sample"))
    def sample(self, **params):

        # TODO clean routing-compatible implementation

        Xt = self.steps[-1][1].sample(**params)
        reverse_iter = reversed(list(self._iter())[:-1])
        for _, name, transform in reverse_iter:
            Xt = transform.inverse_transform(Xt)
        return Xt


def make_ext_pipeline(*steps, memory=None, verbose=False):
    """Construct an extended pipeline that normalize proibabilities with
    intermediate jacobians and a sampling method.
    """
    return ExtPipeline(_name_estimators(steps), memory=memory, verbose=verbose)
