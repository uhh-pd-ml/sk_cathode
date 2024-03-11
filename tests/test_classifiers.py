import numpy as np
import pytest

from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier as ClassifierModelNN  # noqa
from sk_cathode.classifier_models.boosted_decision_tree import HGBClassifier as ClassifierModelTree  # noqa


classifier_models = [
    ClassifierModelNN(epochs=1, save_path=None,),
    ClassifierModelTree(max_iters=10, save_path=None,),
]
X_test = np.random.rand(300, 4)
y_test = np.random.randint(0, 2, 300)
X_val = np.random.rand(100, 4)
y_val = np.random.randint(0, 2, 100)


def allow_not_implemented(func):
    def wrapper(classifier_model, *args, **kwargs):
        try:
            func(classifier_model, *args, **kwargs)
        except NotImplementedError:
            assert True
    return wrapper


@pytest.mark.parametrize("classifier_model", classifier_models)
@allow_not_implemented
def test_fit(classifier_model):
    fit_output = classifier_model.fit(
        X_test, y_test, X_val=X_val, y_val=y_val)
    assert isinstance(fit_output, type(classifier_model))


@pytest.mark.parametrize("classifier_model", classifier_models)
@allow_not_implemented
def test_predict(classifier_model):
    predict_output = classifier_model.predict(X_test)
    assert predict_output.shape == (X_test.shape[0], 1)
