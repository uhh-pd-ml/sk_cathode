import numpy as np
import pytest

from sk_cathode.generative_models.conditional_normalizing_flow_torch import ConditionalNormalizingFlow as GenerativeModelTorch  # noqa
from sk_cathode.generative_models.conditional_normalizing_flow_pyro import ConditionalNormalizingFlow as GenerativeModelPyro  # noqa
from sk_cathode.generative_models.conditional_normalizing_flow_nflows import ConditionalNormalizingFlow as GenerativeModelNflows  # noqa
from sk_cathode.generative_models.conditional_flow_matching import ConditionalFlowMatching as GenerativeModelCFM  # noqa


generative_models = [
    GenerativeModelTorch(epochs=1, save_path=None,),
    GenerativeModelPyro(epochs=1, save_path=None,),
    GenerativeModelNflows(epochs=1, save_path=None,),
    GenerativeModelCFM(epochs=1, save_path=None,),
]
X_test = np.random.rand(300, 4)
m_test = np.random.rand(300, 1)
X_val = np.random.rand(100, 4)
m_val = np.random.rand(100, 1)


def allow_not_implemented(func):
    def wrapper(generative_model, *args, **kwargs):
        try:
            func(generative_model, *args, **kwargs)
        except NotImplementedError:
            assert True
    return wrapper


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_fit(generative_model):
    fit_output = generative_model.fit(
        X_test, m_test, X_val=X_val, m_val=m_val)
    assert isinstance(fit_output, type(generative_model))


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_fit_transform(generative_model):
    transform_output = generative_model.fit_transform(
        X_test, m=m_test, X_val=X_val, m_val=m_val)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_transform(generative_model):
    transform_output = generative_model.transform(X_test, m=m_test)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_inverse_transform(generative_model):
    transform_output = generative_model.inverse_transform(X_test, m=m_test)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_log_jacobian_determinant(generative_model):
    jacobian_output = generative_model.log_jacobian_determinant(
        X_test, m=m_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_jacobian_determinant(generative_model):
    jacobian_output = generative_model.jacobian_determinant(
        X_test, m=m_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_inverse_log_jacobian_determinant(generative_model):
    jacobian_output = generative_model.inverse_log_jacobian_determinant(
        X_test, m=m_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_inverse_jacobian_determinant(generative_model):
    jacobian_output = generative_model.inverse_jacobian_determinant(
        X_test, m=m_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_sample(generative_model):
    sample_output = generative_model.sample(n_samples=len(m_test), m=m_test)
    assert sample_output.shape == X_test.shape


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_predict_log_proba(generative_model):
    proba_output = generative_model.predict_log_proba(X_test, m_test)
    assert proba_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_predict_proba(generative_model):
    proba_output = generative_model.predict_proba(X_test, m_test)
    assert proba_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_score_samples(generative_model):
    score_output = generative_model.score_samples(X_test, m_test)
    assert score_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("generative_model", generative_models)
@allow_not_implemented
def test_score(generative_model):
    score_output = generative_model.score(X_test, m_test)
    assert isinstance(score_output, float)


@pytest.mark.parametrize("generative_model", generative_models)
def test_print(generative_model):
    print(generative_model)
    assert True
