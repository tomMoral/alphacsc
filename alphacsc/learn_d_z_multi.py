"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

from __future__ import print_function
import time
import sys

import numpy as np
from scipy import sparse

from .utils import check_random_state
from .utils.lil import is_list_of_lil
from .utils.whitening import whitening
from .cython_code import _assert_cython
from .update_z_multi import update_z_multi
from .utils.profile_this import profile_this  # noqa
from .utils.dictionary import get_lambda_max
from .update_d_multi import update_uv, update_d
from .init_dict import init_dictionary, get_max_error_dict
from .loss_and_gradient import compute_X_and_objective_multi


def learn_d_z_multi(X, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                    loss='l2', loss_params=dict(gamma=.1, sakoe_chiba_band=10,
                                                ordar=10),
                    rank1=True, uv_constraint='separate', eps=1e-10,
                    algorithm='batch', algorithm_params=dict(),
                    detrending=None, detrending_params=dict(),
                    solver_z='l-bfgs', solver_z_kwargs=dict(),
                    solver_d='alternate_adaptive', solver_d_kwargs=dict(),
                    D_init=None, D_init_params=dict(),
                    stopping_pobj=None, use_sparse_z=False, lmbd_max='fixed',
                    verbose=10, callback=None, random_state=None, name="DL",
                    raise_on_increase=True):
    """Learn atoms and activations using Convolutional Sparse Coding.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    reg : float
        The regularization parameter
    n_iter : int
        The number of coordinate-descent iterations.
    n_jobs : int
        The number of parallel jobs.
    loss : 'l2' | 'dtw'
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
    loss_params : dict
        Parameters of the loss
    rank1 : boolean
        If set to True, learn rank 1 dictionary atoms.
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    eps : float
        Stopping criterion. If the cost descent after a uv and a z update is
        smaller than eps, return.
    algorithm : 'batch' | 'greedy' | 'online'
        Dictionary learning algorithm.
    algorithm_params : dict
        Parameters for the global algorithm used to learn the dictionary:
        alpha : float
            Forgetting factor for online learning. If set to 0, the learning is
            stochastic and each D-step is independent from the previous steps.
            When set to 1, each the previous values z_hat - computed with
            different dictionary - have the same weight as the current one.
            This factor should be large enough to ensure convergence but to
            large factor can lead to sub-optimal minima.
        batch_selection : 'random' | 'cyclic'
            The batch selection strategy for online learning. The batch are
            either selected randomly among all samples (without replacement) or
            in a cyclic way.
        batch_size : int in [1, n_trials]
            Size of the batch used in online learning. Increasing it
            regularizes the dictionary learning as there is less variance for
            the successive estimates. But it also increases the computational
            cost as more coding signals z_hat must be estimate at each
            iteration.
    solver_z : str
        The solver to use for the z update. Options are
        'l-bfgs' (default) | "lgcd"
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' | 'alternate_adaptive' (default) | 'joint' | 'l-bfgs'
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    D_init : str or array, shape (n_atoms, n_channels + n_times_atoms) or
                           shape (n_atoms, n_channels, n_times_atom)
        The initial atoms or an initialization scheme in {'kmeans' | 'ssa' |
        'chunks' | 'random'}.
    D_init_params : dict
        Dictionnary of parameters for the kmeans init method.
    use_sparse_z : boolean
        Use sparse lil_matrices to store the activations.
    lmbd_max : 'fixed' | 'scaled' | 'per_atom' | 'shared'
        If not fixed, adapt the regularization rate as a ratio of lambda_max.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.
    random_state : int | None
        The random state.
    raise_on_increase : boolean
        Raise an error if the objective function increase

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    z_hat : array, shape (n_trials, n_atoms, n_times_valid)
        The sparse activation matrix.
    """

    assert lmbd_max in ['fixed', 'scaled', 'per_atom', 'shared'], (
        "lmbd_max should be in {'fixed', 'scaled', 'per_atom', 'shared'}, "
        "not '{}'".format(lmbd_max)
    )

    n_trials, n_channels, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    # initialization
    start = time.time()
    rng = check_random_state(random_state)

    D_hat = init_dictionary(X, n_atoms, n_times_atom, D_init=D_init,
                            rank1=rank1, uv_constraint=uv_constraint,
                            D_init_params=D_init_params, random_state=rng)
    b_hat_0 = rng.randn(n_atoms * (n_channels + n_times_atom))
    init_duration = time.time() - start

    if use_sparse_z:
        _assert_cython()
        z_hat = [sparse.lil_matrix((n_atoms, n_times_valid))
                 for _ in range(n_trials)]
    else:
        z_hat = np.zeros((n_trials, n_atoms, n_times_valid))

    z_kwargs = dict(verbose=verbose)
    z_kwargs.update(solver_z_kwargs)

    # Compute the coefficients to whiten X. TODO: add timing
    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=loss_params['ordar'])

    _lmbd_max = get_lambda_max(X, D_hat).max()
    print("[CDL] Max value for lambda: {}".format(_lmbd_max))
    if lmbd_max == "scaled":
        reg = reg * _lmbd_max

    def compute_z_func(X, z_hat, D_hat, reg=None):
        return update_z_multi(X, D_hat, reg=reg, z0=z_hat,
                              solver=solver_z, solver_kwargs=z_kwargs,
                              loss=loss, loss_params=loss_params,
                              n_jobs=n_jobs)

    def obj_func(X, z_hat, D_hat, reg=None, return_X_hat=False):
        return compute_X_and_objective_multi(X, z_hat, D_hat,
                                             reg=reg, loss=loss,
                                             loss_params=loss_params,
                                             uv_constraint=uv_constraint,
                                             feasible_evaluation=True,
                                             return_X_hat=return_X_hat)

    d_kwargs = dict(verbose=verbose, eps=1e-8)
    d_kwargs.update(solver_d_kwargs)
    if algorithm == "stochastic":
        # The typical stochastic algorithm samples one signal, compute the
        # associated value z and then perform one step of gradient descent
        # for D.
        d_kwargs["max_iter"] = 1

    def compute_d_func(X, z_hat, D_hat, constants):
        if rank1:
            return update_uv(X, z_hat, uv_hat0=D_hat, constants=constants,
                             b_hat_0=b_hat_0, solver_d=solver_d,
                             uv_constraint=uv_constraint, loss=loss,
                             loss_params=loss_params, **d_kwargs)
        else:
            return update_d(X, z_hat, D_hat0=D_hat, constants=constants,
                            b_hat_0=b_hat_0, solver_d=solver_d,
                            uv_constraint=uv_constraint, loss=loss,
                            loss_params=loss_params, **d_kwargs)

    end_iter_func = get_iteration_func(eps, stopping_pobj, callback, lmbd_max,
                                       name, verbose, raise_on_increase)

    if algorithm == 'batch':
        pobj, times, D_hat, z_hat = _batch_learn(
            X, D_hat, z_hat, compute_z_func, compute_d_func,
            obj_func, end_iter_func, n_iter=n_iter,
            verbose=verbose, random_state=random_state,
            reg=reg, lmbd_max=lmbd_max, name=name, **algorithm_params
        )
    elif algorithm == "greedy":
        raise NotImplementedError(
            "Algorithm greedy is not implemented yet.")
    elif algorithm == "online":
        pobj, times, D_hat, z_hat = _online_learn(
            X, D_hat, z_hat, compute_z_func, compute_d_func, obj_func,
            end_iter_func, n_iter=n_iter, verbose=verbose,
            random_state=random_state, reg=reg,
            lmbd_max=lmbd_max, name=name, **algorithm_params
        )
    elif algorithm == "stochastic":
        # For stochastic learning, set forgetting factor alpha of the
        # online algorithm to 0, making each step independent of previous
        # steps and set D-update max_iter to a low value (typically 1).
        pobj, times, D_hat, z_hat = _online_learn(
            X, D_hat, z_hat, compute_z_func, compute_d_func, obj_func,
            end_iter_func, n_iter=n_iter, verbose=verbose,
            random_state=random_state, reg=reg,
            lmbd_max=lmbd_max, name=name, **algorithm_params
        )
    else:
        raise NotImplementedError(
            "Algorithm '{}' is not implemented to learn dictionary atoms."
            .format(algorithm))

    # recompute z_hat with no regularization and keeping the support fixed
    t_start = time.time()
    z_hat, _, _ = update_z_multi(
        X, D_hat, reg=0, z0=z_hat, n_jobs=n_jobs, solver=solver_z,
        solver_kwargs=solver_z_kwargs, freeze_support=True, loss=loss,
        loss_params=loss_params)
    if verbose > 1:
        print("[{}] Compute the final z_hat with support freeze in {:.2f}s"
              .format(name, time.time() - t_start))

    times[0] += init_duration

    return pobj, times, D_hat, z_hat


def _batch_learn(X, D_hat, z_hat, compute_z_func, compute_d_func,
                 obj_func, end_iter_func, n_iter=100,
                 lmbd_max='fixed', reg=None, verbose=0,
                 random_state=None, name="batch"):

    reg_ = reg

    # Initialize constants dictionary
    constants = {}
    constants['n_channels'] = X.shape[1]
    constants['XtX'] = np.dot(X.ravel(), X.ravel())

    # monitor cost function
    times = [0]
    pobj = [obj_func(X, z_hat, D_hat, reg=reg_)]

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            msg = '.' if (ii % 50 != 0) else 'M_%d/%d ' % (ii, n_iter)
            print(msg, end='')
            sys.stdout.flush()
        if verbose > 1:
            print('[{}] CD iterations {} / {}'.format(name, ii, n_iter))

        if lmbd_max not in ['fixed', 'scaled']:
            reg_ = reg * get_lambda_max(X, D_hat)
            if lmbd_max == 'shared':
                reg_ = reg_.max()

        if verbose > 5:
            print('[{}] lambda = {:.3e}'.format(name, np.mean(reg_)))

        # Compute z update
        start = time.time()
        z_hat, constants['ztz'], constants['ztX'] = compute_z_func(
            X, z_hat, D_hat, reg=reg_)

        # monitor cost function
        times.append(time.time() - start)
        cost, X_hat = obj_func(X, z_hat, D_hat, reg=reg_, return_X_hat=True)
        pobj.append(cost)

        if is_list_of_lil(z_hat):
            z_nnz = np.array([[len(d) for d in z.data] for z in z_hat]
                             ).sum(axis=0)
            z_size = len(z_hat) * np.prod(z_hat[0].shape)
        else:
            z_nnz = np.sum(z_hat != 0, axis=(0, 2))
            z_size = z_hat.size

        if verbose > 5:
            print("[{}] sparsity: {:.3e}".format(
                name, z_nnz.sum() / z_size))
            print('[{}] Objective (z) : {:.3e}'.format(name, pobj[-1]))

        if np.all(z_nnz == 0):
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        # Compute D update
        start = time.time()
        D_hat = compute_d_func(X, z_hat, D_hat, constants)

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(obj_func(X, z_hat, D_hat, reg=reg_))

        null_atom_indices = np.where(z_nnz == 0)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            D_hat[k0] = get_max_error_dict(X, z_hat, D_hat)[0]
            if verbose > 1:
                print('[{}] Resampled atom {}'.format(name, k0))

        if verbose > 5:
            print('[{}] Objective (d) : {:.3e}'.format(name, pobj[-1]))

        if end_iter_func(X, z_hat, D_hat, pobj, ii):
            break

    return pobj, times, D_hat, z_hat


def _online_learn(X, D_hat, z_hat, compute_z_func, compute_d_func,
                  obj_func, end_iter_func, n_iter=100, verbose=0,
                  random_state=None, lmbd_max='fixed', reg=None,
                  alpha=.8, batch_selection='random', batch_size=1,
                  name="online"):

    reg_ = reg

    # Initialize constants dictionary
    constants = {}
    n_trials, n_channels = X.shape[:2]
    if D_hat.ndim == 2:
        n_atoms, n_times_atom = D_hat.shape
        n_times_atom -= n_channels
    else:
        n_atoms, _, n_times_atom = D_hat.shape
    constants['n_channels'] = n_channels
    constants['XtX'] = np.dot(X.ravel(), X.ravel())
    constants['ztz'] = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
    constants['ztX'] = np.zeros((n_atoms, n_channels, n_times_atom))

    # monitor cost function
    times = [0]
    pobj = [obj_func(X, z_hat, D_hat, reg=reg_)]

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            msg = '.' if (ii % 50 != 0) else 'M_%d/%d ' % (ii, n_iter)
            print(msg, end='')
            sys.stdout.flush()
        if verbose > 1:
            print('[{}] CD iterations {} / {}'.format(name, ii, n_iter))

        if lmbd_max not in ['fixed', 'scaled']:
            reg_ = reg * get_lambda_max(X, D_hat)
            if lmbd_max == 'shared':
                reg_ = reg_.max()

        if verbose > 5:
            print('[{}] lambda = {:.3e}'.format(name, np.mean(reg_)))

        # Compute z update
        start = time.time()
        if batch_selection == 'random':
            i0 = np.random.choice(n_trials, batch_size, replace=False)
        elif batch_selection == 'cyclic':
            i_slice = (ii * batch_size) % n_trials
            i0 = slice(i_slice, i_slice + batch_size)
        else:
            raise NotImplementedError(
                "the '{}' batch_selection strategy for the online learning is "
                "not implemented.".format(batch_selection))
        z_hat[i0], ztz, ztX = compute_z_func(X[i0], z_hat[i0], D_hat, reg=reg_)

        constants['ztz'] = alpha * constants['ztz'] + ztz
        constants['ztX'] = alpha * constants['ztX'] + ztX

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(obj_func(X, z_hat, D_hat, reg=reg_))

        if is_list_of_lil(z_hat):
            z_nnz = np.array([[len(d) for d in z.data] for z in z_hat]
                             ).sum(axis=0)
            z_size = len(z_hat) * np.prod(z_hat[0].shape)
        else:
            z_nnz = np.sum(z_hat != 0, axis=(0, 2))
            z_size = z_hat.size

        if verbose > 5:
            print("[{}] sparsity: {:.3e}".format(
                name, z_nnz.sum() / z_size))
            print('[{}] Objective (z) : {:.3e}'.format(name, pobj[-1]))

        if np.all(z_nnz == 0):
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        # Compute D update
        start = time.time()
        D_hat = compute_d_func(X, z_hat, D_hat, constants)

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(obj_func(X, z_hat, D_hat, reg=reg_))

        if verbose > 5:
            print('[{}] Objective (d) : {:.3e}'.format(name, pobj[-1]))

        if end_iter_func(X, z_hat, D_hat, pobj, ii):
            break

    return pobj, times, D_hat, z_hat


def get_iteration_func(eps, stopping_pobj, callback, lmbd_max, name, verbose,
                       raise_on_increase):
    def end_iteration(X, z_hat, D_hat, pobj, iteration):
        if callable(callback):
            callback(X, D_hat, z_hat, pobj)

        # Only check that the cost is always going down when the regularization
        # parameter is fixed.
        dz = pobj[-3] - pobj[-2]
        du = pobj[-2] - pobj[-1]
        if ((dz < eps or du < eps) and lmbd_max == 'fixed'):
            if dz < 0 and raise_on_increase:
                raise RuntimeError(
                    "The z update have increased the objective value by %s."
                    % dz)
            if du < -1e-10 and dz > 1e-12 and raise_on_increase:
                raise RuntimeError(
                    "The d update have increased the objective value by %s."
                    "(dz=%s)" % (du, dz))
            if dz < eps and du < eps:
                print("[{}] Converged after {} iteration, dz, du "
                      "={:.3e}, {:.3e}".format(name, iteration, dz, du))
                return True

        if stopping_pobj is not None and pobj[-1] < stopping_pobj:
            return True
        return False

    return end_iteration
