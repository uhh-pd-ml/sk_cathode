import numpy as np
import pytest

from sk_cathode.utils.preprocessing import (LogitScaler,
                                            ExtStandardScaler)


scalers = [
    LogitScaler(epsilon=0.1),
    ExtStandardScaler(),
]
X_test = np.random.rand(300, 4)


def allow_not_implemented(func):
    def wrapper(scaler, *args, **kwargs):
        try:
            func(scaler, *args, **kwargs)
        except NotImplementedError:
            assert True
    return wrapper


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_fit(scaler):
    fit_output = scaler.fit(
        X_test)
    assert isinstance(fit_output, type(scaler))


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_fit_transform(scaler):
    transform_output = scaler.fit_transform(
        X_test)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_transform(scaler):
    transform_output = scaler.transform(X_test)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_inverse_transform(scaler):
    transform_output = scaler.inverse_transform(X_test)
    assert transform_output.shape == X_test.shape


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_log_jacobian_determinant(scaler):
    jacobian_output = scaler.log_jacobian_determinant(
        X_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_jacobian_determinant(scaler):
    jacobian_output = scaler.jacobian_determinant(
        X_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_inverse_log_jacobian_determinant(scaler):
    jacobian_output = scaler.inverse_log_jacobian_determinant(
        X_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)


@pytest.mark.parametrize("scaler", scalers)
@allow_not_implemented
def test_inverse_jacobian_determinant(scaler):
    jacobian_output = scaler.inverse_jacobian_determinant(
        X_test)
    assert jacobian_output.shape == (X_test.shape[0], 1)
