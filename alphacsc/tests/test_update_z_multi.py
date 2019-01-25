import pytest
import numpy as np
from scipy import sparse

from alphacsc import cython_code
from alphacsc.utils import construct_X_multi
from alphacsc.utils import check_random_state
from alphacsc.utils.whitening import whitening
from alphacsc.update_z_multi import update_z_multi
from alphacsc.update_z_multi import compute_DtD, _coordinate_descent_idx
from alphacsc.loss_and_gradient import compute_X_and_objective_multi

from alphacsc.utils.compute_constants import compute_ztz, compute_ztX


LOSSES = ['l2']  # , 'dtw', 'whitening']


@pytest.mark.parametrize('loss', LOSSES)
@pytest.mark.parametrize('solver', ['l-bfgs', 'ista', 'fista'])
def test_update_z_multi_decrease_cost_function(loss, solver):
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0
    loss_params = dict(gamma=1, sakoe_chiba_band=n_times_atom // 2)

    rng = check_random_state(0)
    X = rng.randn(n_trials, n_channels, n_times)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=10)

    loss_0 = compute_X_and_objective_multi(X=X, z_hat=z, D_hat=uv, reg=reg,
                                           feasible_evaluation=False,
                                           loss=loss, loss_params=loss_params)

    z_hat, ztz, ztX, *_ = update_z_multi(X, uv, reg, z0=z, solver=solver,
                                         loss=loss, loss_params=loss_params,
                                         return_ztz=True)

    loss_1 = compute_X_and_objective_multi(X=X, z_hat=z_hat, D_hat=uv,
                                           reg=reg, feasible_evaluation=False,
                                           loss=loss, loss_params=loss_params)
    assert loss_1 < loss_0

    if loss == 'l2':
        assert np.allclose(ztz, compute_ztz(z_hat, (n_times_atom,)))
        assert np.allclose(ztX, compute_ztX(z_hat, X))


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
def test_support_least_square(solver_z):
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0.1
    solver_kwargs = {'factr': 1e7}

    rng = check_random_state(2)
    X = rng.randn(n_trials, n_channels, n_times)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    # The initial loss should be high
    loss_0 = compute_X_and_objective_multi(X, z_hat=z, D_hat=uv, reg=0,
                                           feasible_evaluation=False)

    # The loss after updating z should be lower
    z_hat, *_ = update_z_multi(X, uv, reg, z0=z, solver=solver_z,
                               solver_kwargs=solver_kwargs)
    loss_1 = compute_X_and_objective_multi(X, z_hat=z_hat, D_hat=uv, reg=0,
                                           feasible_evaluation=False)
    assert loss_1 < loss_0

    # Here we recompute z on the support of z_hat, with reg=0
    z_hat_2, *_ = update_z_multi(X, uv, reg=0, z0=z_hat, solver=solver_z,
                                 solver_kwargs=solver_kwargs,
                                 freeze_support=True)
    loss_2 = compute_X_and_objective_multi(X, z_hat_2, uv, 0,
                                           feasible_evaluation=False)
    assert loss_2 <= loss_1 or np.isclose(loss_1, loss_2)

    # Here we recompute z with reg=0, but with no support restriction
    z_hat_3, *_ = update_z_multi(X, uv, reg=0, z0=z_hat_2,
                                 solver=solver_z,
                                 solver_kwargs=solver_kwargs,
                                 freeze_support=True)
    loss_3 = compute_X_and_objective_multi(X, z_hat_3, uv, 0,
                                           feasible_evaluation=False)
    assert loss_3 <= loss_2 or np.isclose(loss_3, loss_2)


@pytest.mark.parametrize('use_sparse_lil', [True, False])
def test_cd(use_sparse_lil):
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 1

    rng = check_random_state(4)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    if use_sparse_lil:
        density = .1
        z = [sparse.random(n_atoms, n_times_valid, format='lil',
                           density=density, random_state=0)
             for _ in range(n_trials)]
        z_gen = [sparse.random(n_atoms, n_times_valid, format='lil',
                               density=density, random_state=0)
                 for _ in range(n_trials)]
        z0 = z[0]
    else:
        z = abs(rng.randn(n_trials, n_atoms, n_times_valid))
        z_gen = abs(rng.randn(n_trials, n_atoms, n_times_valid))
        z[z < 1] = 0
        z_gen[z_gen < 1] = 0
        z0 = z[0]

    X = construct_X_multi(z_gen, D=uv, n_channels=n_channels)

    loss_0 = compute_X_and_objective_multi(X=X, z_hat=z_gen, D_hat=uv, reg=reg,
                                           loss='l2',
                                           feasible_evaluation=False)

    constants = {}
    constants['DtD'] = compute_DtD(uv, n_channels)

    # Ensure that the initialization is good, by using a nearly optimal point
    # and verifying that the cost does not goes up.
    z_hat, ztz, ztX, *_ = update_z_multi(X, D=uv, reg=reg, z0=z_gen,
                                         solver="lgcd",
                                         solver_kwargs={
                                             'max_iter': 5, 'tol': 1e-5
                                         },
                                         return_ztz=True)
    if use_sparse_lil and cython_code._CYTHON_AVAILABLE:
        from alphacsc.cython_code import _fast_compute_ztz, _fast_compute_ztX
        assert np.allclose(ztz, _fast_compute_ztz(z_hat, n_times_atom))
        assert np.allclose(ztX, _fast_compute_ztX(z_hat, X))

    else:
        assert np.allclose(ztz, compute_ztz(z_hat, (n_times_atom,)))
        assert np.allclose(ztX, compute_ztX(z_hat, X))

    loss_1 = compute_X_and_objective_multi(X=X, z_hat=z_hat, D_hat=uv,
                                           reg=reg, loss='l2',
                                           feasible_evaluation=False)
    assert loss_1 <= loss_0, "Bad initialization in greedy CD."

    z_hat, pobj, times = _coordinate_descent_idx(X[0], uv, constants, reg,
                                                 debug=True, timing=True,
                                                 z0=z0, max_iter=10000)

    try:
        assert all([p1 >= p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), "oups"
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.plot(pobj)
        plt.show()
        raise


def test_update_z_multi_2d():
    n_trials, n_channels, sig_shape = 2, 3, (100, 100)
    n_atoms, atom_support = 4, (10, 10)
    valid_shape = (91, 91)
    reg = 0

    solver_kwargs = dict()

    rng = check_random_state(0)
    X = rng.randn(n_trials, n_channels, *sig_shape)
    D = rng.randn(n_atoms, n_channels, *atom_support)
    z = rng.randn(n_trials, n_atoms, *valid_shape)

    loss_0 = compute_X_and_objective_multi(X=X, z_hat=z, D_hat=D, reg=reg,
                                           feasible_evaluation=False,
                                           loss='l2')

    z_hat, ztz, ztX, cost = update_z_multi(X, D, reg, z0=z, solver="dicod",
                                           loss='l2', return_ztz=True,
                                           solver_kwargs=solver_kwargs)

    loss_1 = compute_X_and_objective_multi(X=X, z_hat=z_hat, D_hat=D,
                                           reg=reg, feasible_evaluation=False,
                                           loss='l2')
    if cost is not None:
        assert np.isclose(loss_1, cost)
    assert loss_1 < loss_0
