import pytest
import numpy as np
from scipy import optimize

from alphacsc.utils import construct_X_multi
from alphacsc.utils import check_random_state
from alphacsc.utils.whitening import whitening
from alphacsc.loss_and_gradient import compute_objective
from alphacsc.loss_and_gradient import gradient_d, gradient_uv
from alphacsc.update_d_multi import update_uv, prox_uv, _get_d_update_constants


from alphacsc.utils.shape_manipulation import get_valid_shape

DEBUG = False

LOSSES = ['l2']  # , 'dtw', 'whitening']


@pytest.mark.parametrize('loss', LOSSES)
def test_gradient_d(loss):
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    # Constant for the DTW loss
    loss_params = dict(gamma=1, sakoe_chiba_band=n_times_atom // 2)

    rng = check_random_state(724)
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))
    d = rng.normal(size=(n_atoms, n_channels, n_times_atom)).ravel()

    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=10)

    def func(d0):
        D0 = d0.reshape(n_atoms, n_channels, n_times_atom)
        X_hat = construct_X_multi(z, D=D0)
        return compute_objective(X, X_hat, loss=loss, loss_params=loss_params)

    def grad(d0):
        return gradient_d(D=d0, X=X, z=z, loss=loss, loss_params=loss_params,
                          flatten=True)

    error = optimize.check_grad(func, grad, d, epsilon=2e-8)
    grad_d = grad(d)
    n_grad = np.sqrt(np.dot(grad_d, grad_d))
    try:
        assert error < 1e-5 * n_grad, "Gradient is false: {:.4e}".format(error)
    except AssertionError:
        if DEBUG:
            grad_approx = optimize.approx_fprime(d, func, 2e-8)

            import matplotlib.pyplot as plt
            plt.semilogy(abs(grad_approx - grad_d))
            plt.figure()
            plt.plot(grad_approx, label="approx")
            plt.plot(grad_d, '--', label="grad")
            plt.legend()
            plt.show()
        raise


@pytest.mark.parametrize('loss', LOSSES)
def test_gradient_uv(loss):
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3
    loss_params = dict(gamma=1, sakoe_chiba_band=n_times_atom // 2)

    rng = check_random_state(1)
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))
    uv = rng.normal(size=(n_atoms, n_channels + n_times_atom)).ravel()

    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=10)

    def func(uv0):
        uv0 = uv0.reshape(n_atoms, n_channels + n_times_atom)
        X_hat = construct_X_multi(z, D=uv0, n_channels=n_channels)
        return compute_objective(X, X_hat, loss=loss, loss_params=loss_params)

    def grad(uv0):
        return gradient_uv(uv=uv0, X=X, z=z, flatten=True, loss=loss,
                           loss_params=loss_params)

    error = optimize.check_grad(func, grad, uv.ravel(), epsilon=2e-8)
    grad_uv = grad(uv)
    n_grad = np.sqrt(np.dot(grad_uv, grad_uv))
    try:
        assert error < 1e-5 * n_grad, "Gradient is false: {:.4e}".format(error)
    except AssertionError:

        if DEBUG:
            grad_approx = optimize.approx_fprime(uv, func, 2e-8)

            import matplotlib.pyplot as plt
            plt.semilogy(abs(grad_approx - grad_uv))
            plt.figure()
            plt.plot(grad_approx, label="approx")
            plt.plot(grad_uv, '--', label="grad")
            plt.legend()
            plt.show()
        raise

    if loss == 'l2':
        constants = _get_d_update_constants(X, z)
        msg = "Wrong value for zt*X"
        assert np.allclose(
            gradient_uv(0 * uv, X=X, z=z, flatten=True),
            gradient_uv(0 * uv, constants=constants, flatten=True)), msg
        msg = "Wrong value for zt*z"
        assert np.allclose(
            gradient_uv(uv, X=X, z=z, flatten=True),
            gradient_uv(uv, constants=constants, flatten=True)), msg


@pytest.mark.parametrize('solver_d, uv_constraint', [
    ('joint', 'joint'), ('alternate', 'separate')
])
def test_update_uv(solver_d, uv_constraint):
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    rng = check_random_state(3)
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))
    uv0 = rng.normal(size=(n_atoms, n_channels + n_times_atom))
    uv1 = rng.normal(size=(n_atoms, n_channels + n_times_atom))

    uv0 = prox_uv(uv0)
    uv1 = prox_uv(uv1)

    X = construct_X_multi(z, D=uv0, n_channels=n_channels)

    def objective(uv):
        X_hat = construct_X_multi(z, D=uv, n_channels=n_channels)
        return compute_objective(X, X_hat, loss='l2')

    # Ensure that the known optimal point is stable
    uv = update_uv(X, z, uv0, max_iter=1000, verbose=0)
    cost = objective(uv)

    assert np.isclose(cost, 0), "optimal point not stable"
    assert np.allclose(uv, uv0), "optimal point not stable"

    # Ensure that the update is going down from a random initialization
    cost0 = objective(uv1)
    uv, pobj = update_uv(X, z, uv1, debug=True, max_iter=5000, verbose=10,
                         solver_d=solver_d, momentum=False, eps=1e-10,
                         uv_constraint=uv_constraint)
    cost1 = objective(uv)

    msg = "Learning is not going down"
    try:
        assert cost1 < cost0, msg
        # assert np.isclose(cost1, 0, atol=1e-7)
    except AssertionError:
        import matplotlib.pyplot as plt
        pobj = np.array(pobj)
        plt.semilogy(pobj)
        plt.title(msg)
        plt.show()
        raise


def test_constants_d():
    """Test that _shifted_objective_uv compute the right thing"""
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    rng = check_random_state(7)
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))

    from alphacsc.update_d_multi import _get_d_update_constants
    constants = _get_d_update_constants(X, z)

    ztX = np.sum([[[np.convolve(zik[::-1], xip, mode='valid') for xip in xi]
                   for zik in zi] for zi, xi in zip(z, X)], axis=0)

    assert np.allclose(ztX, constants['ztX'])

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    axes = ([0, 2], [0, 2])

    for t in range(n_times_atom):
        if t == 0:
            ztz[:, :, t0] += np.tensordot(z, z, axes=axes)
        else:
            tmp = np.tensordot(z[:, :, :-t], z[:, :, t:], axes=axes)
            ztz[:, :, t0 + t] += tmp
            tmp = np.tensordot(z[:, :, t:], z[:, :, :-t], axes=axes)
            ztz[:, :, t0 - t] += tmp

    assert np.allclose(ztz, constants['ztz'])
