import pytest
import numpy as np
from functools import partial
from scipy.optimize import approx_fprime


from alphacsc.utils import get_D
from alphacsc.utils import construct_X_multi
from alphacsc.utils import check_random_state
from alphacsc.utils.whitening import whitening
from alphacsc.loss_and_gradient import gradient_d
from alphacsc.loss_and_gradient import gradient_zi
from alphacsc.utils.shape_manipulation import get_valid_shape
from alphacsc.loss_and_gradient import compute_X_and_objective_multi


def _gradient_zi(X, z, D, loss, loss_params, flatten=False):
    return gradient_zi(X[0], z[0], D, loss=loss, flatten=flatten,
                       loss_params=loss_params)


def _construct_X(X, z, D, loss, loss_params):
    return construct_X_multi(z, D, n_channels=X.shape[1])


def _objective(X, z, D, loss, loss_params):
    return compute_X_and_objective_multi(X, z, D, feasible_evaluation=False,
                                         loss=loss,
                                         loss_params=loss_params)


def _gradient_d(X, z, D, loss, loss_params, flatten=False):
    return gradient_d(D, X, z, loss=loss, flatten=flatten,
                      loss_params=loss_params)


def gradient_checker(func, grad, shape, args=(), kwargs={}, n_checks=10,
                     rtol=1e-5, grad_name='gradient', debug=False,
                     random_seed=None):
    """Check that the gradient correctly approximate the finite difference
    """

    rng = check_random_state(random_seed)

    msg = ("Computed {} did not match gradient computed with finite "
           "difference. Relative error is {{}}".format(grad_name))

    func = partial(func, **kwargs)

    def test_grad(z0):
        grad_approx = approx_fprime(z0, func, 1e-8, *args)
        grad_compute = grad(z0, *args, **kwargs)
        error = np.sum((grad_approx - grad_compute) ** 2)
        error /= np.sum(grad_approx ** 2)
        error = np.sqrt(error)
        try:
            assert error < rtol, msg.format(error)
        except AssertionError:
            if debug:
                import matplotlib.pyplot as plt
                plt.plot(grad_approx)
                plt.plot(grad_compute)
                plt.show()
            raise

    z0 = np.zeros(shape)
    test_grad(z0)

    for _ in range(n_checks):
        z0 = rng.randn(shape)
        test_grad(z0)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('func', [
    _construct_X, _gradient_zi, _objective, _gradient_d])
def test_consistency(loss, func):
    """Check that the result are the same for the full rank D and rank 1 uv.
    """
    n_trials, n_channels, n_times = 5, 3, 100
    n_atoms, n_times_atom = 10, 15

    rng = check_random_state(27)

    loss_params = dict(gamma=.01)

    n_times_valid = n_times - n_times_atom + 1

    z = rng.randn(n_trials, n_atoms, n_times_valid)
    X = rng.randn(n_trials, n_channels, n_times)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    val_D = func(X, z, D, loss, loss_params=loss_params)
    val_uv = func(X, z, uv, loss, loss_params=loss_params)
    assert np.allclose(val_D, val_uv)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
def test_gradients(loss):
    """Check that the gradients have the correct shape.
    """
    n_trials, n_channels, n_times = 5, 3, 100
    n_atoms, n_times_atom = 10, 15

    rng = check_random_state(427)

    n_checks = 5
    if loss == "dtw":
        n_checks = 1

    loss_params = dict(gamma=.01)

    n_times_valid = n_times - n_times_atom + 1

    X = rng.randn(n_trials, n_channels, n_times)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)
    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    # Test gradient D
    assert D.shape == _gradient_d(X, z, D, loss, loss_params=loss_params).shape

    def pobj(ds):
        return _objective(X, z, ds.reshape(n_atoms, n_channels, -1), loss,
                          loss_params=loss_params)

    def grad(ds):
        return _gradient_d(X, z, ds, loss=loss, flatten=True,
                           loss_params=loss_params)

    gradient_checker(pobj, grad, np.prod(D.shape), n_checks=n_checks,
                     grad_name="gradient D for loss '{}'".format(loss),
                     rtol=1e-4)

    # Test gradient z
    assert z[0].shape == _gradient_zi(
        X, z, D, loss, loss_params=loss_params).shape

    def pobj(zs):
        return _objective(X[:1], zs.reshape(1, n_atoms, -1), D, loss,
                          loss_params=loss_params)

    def grad(zs):
        return gradient_zi(X[0], zs, D, loss=loss, flatten=True,
                           loss_params=loss_params)

    gradient_checker(pobj, grad, n_atoms * n_times_valid, n_checks=n_checks,
                     debug=True, grad_name="gradient z for loss '{}'"
                     .format(loss), rtol=1e-4)


def test_gradients_2d():
    """Check that the gradients have the correct shape.
    """
    n_checks = 5

    n_trials, n_channels, sig_shape = 4, 2, (12, 12)
    n_atoms, atom_support = 3, (3, 3)
    valid_shape = get_valid_shape(sig_shape, atom_support)

    rng = check_random_state(427)

    X = rng.randn(n_trials, n_channels, *sig_shape)
    z = rng.randn(n_trials, n_atoms, *valid_shape)

    D = rng.randn(n_atoms, n_channels, *atom_support)
    D_shape = D.shape

    # Test gradient D
    assert D_shape == _gradient_d(X, z, D, 'l2', loss_params={}).shape

    def pobj(ds):
        return _objective(X, z, ds.reshape(D_shape), 'l2',
                          loss_params={})

    def grad(ds):
        return _gradient_d(X, z, ds, loss='l2', flatten=True,
                           loss_params={})

    gradient_checker(pobj, grad, np.prod(D.shape), n_checks=n_checks,
                     grad_name="gradient D for loss '{}'".format('l2'),
                     rtol=1e-4)

    # Test gradient z

    z0_shape = z[0].shape
    assert z0_shape == _gradient_zi(X, z, D, 'l2', loss_params={}).shape

    def pobj(zs):
        return _objective(X[:1], zs.reshape((1, *z0_shape)), D, 'l2',
                          loss_params={})

    def grad(zs):
        return gradient_zi(X[0], zs, D, loss='l2', flatten=True,
                           loss_params={})

    gradient_checker(pobj, grad, np.prod(z[0].shape), n_checks=n_checks,
                     debug=True, grad_name="gradient z for loss '{}'"
                     .format('l2'), rtol=1e-4)
