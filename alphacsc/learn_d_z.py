"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

from __future__ import print_function
import time
import sys

import numpy as np
from scipy import linalg
from joblib import Parallel

from .utils import construct_X, check_random_state
from .update_z import update_z
from .update_d import update_d_block


def objective(X, X_hat, z_hat, reg, sample_weights=None):
    residual = X - X_hat
    if sample_weights is not None:
        residual *= np.sqrt(sample_weights)
    obj = 0.5 * linalg.norm(residual, 'fro') ** 2 + reg * z_hat.sum()
    return obj


def compute_X_and_objective(X, z_hat, d_hat, reg, sample_weights=None,
                            feasible_evaluation=True):
    X_hat = construct_X(z_hat, d_hat)

    if feasible_evaluation:
        z_hat = z_hat.copy()
        d_hat = d_hat.copy()
        # project to unit norm
        d_norm = np.linalg.norm(d_hat, axis=1)
        mask = d_norm >= 1
        d_hat[mask] /= d_norm[mask][:, None]
        # update z in the opposite way
        z_hat[mask] *= d_norm[mask][:, None, None]

    return objective(X, X_hat, z_hat, reg, sample_weights)


def learn_d_z(X, n_atoms, n_times_atom, func_d=update_d_block, reg=0.1,
              n_iter=60, random_state=None, n_jobs=1, solver_z='l-bfgs',
              solver_d_kwargs=dict(), solver_z_kwargs=dict(), ds_init=None,
              sample_weights=None, verbose=10, callback=None,
              stopping_pobj=None):
    """Learn atoms and activations using Convolutional Sparse Coding.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    func_d : callable
        The function to update the atoms.
    reg : float
        The regularization parameter
    n_iter : int
        The number of coordinate-descent iterations.
    random_state : int | None
        The random state.
    n_jobs : int
        The number of parallel jobs.
    solver_z : str
        The solver to use for the z update. Options are
        'l-bfgs' (default) | 'ista' | 'fista'
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z
    ds_init : array, shape (n_atoms, n_times_atom)
        The initialization for the atoms.
    sample_weights : array, shape (n_trials, n_times)
        The weights in the alphaCSC problem. Should be None
        when using vanilla CSC.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    d_hat : array, shape (n_atoms, n_times)
        The estimated atoms.
    z_hat : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The sparse activation matrix.
    """
    n_trials, n_times = X.shape

    rng = check_random_state(random_state)

    if ds_init is None:
        d_hat = rng.randn(n_atoms, n_times_atom)
    else:
        d_hat = ds_init.copy()
    d_norm = np.linalg.norm(d_hat, axis=1)
    d_hat /= d_norm[:, None]

    pobj = list()
    times = list()

    if 'ista' in solver_z:
        b_hat_0 = rng.randn(n_atoms * (n_times - n_times_atom + 1))
    else:
        b_hat_0 = None

    lambd0 = None
    z_hat = np.zeros((n_atoms, n_trials, n_times - n_times_atom + 1))

    pobj.append(compute_X_and_objective(X, z_hat, d_hat, reg, sample_weights))
    times.append(0.)
    with Parallel(n_jobs=n_jobs) as parallel:
        for ii in range(n_iter):  # outer loop of coordinate descent
            if verbose == 1:
                msg = '.' if (ii % 50 != 0) else 'V_%d/%d ' % (ii, n_iter)
                print(msg, end='')
                sys.stdout.flush()
            if verbose > 1:
                print('Coordinate descent loop %d / %d [n_jobs=%d]' %
                      (ii, n_iter, n_jobs))

            start = time.time()
            z_hat = update_z(X, d_hat, reg, z0=z_hat,
                             parallel=parallel, solver=solver_z,
                             b_hat_0=b_hat_0, solver_kwargs=solver_z_kwargs,
                             sample_weights=sample_weights)
            times.append(time.time() - start)

            # monitor cost function
            pobj.append(
                compute_X_and_objective(X, z_hat, d_hat, reg, sample_weights))
            if verbose > 1:
                print('[seed %s] Objective (z_hat) : %0.8f' %
                      (random_state, pobj[-1]))

            if len(z_hat.nonzero()[0]) == 0:
                import warnings
                warnings.warn("Regularization parameter `reg` is too large "
                              "and all the activations are zero. No atoms has"
                              " been learned.", UserWarning)
                break

            start = time.time()
            d_hat, lambd0 = func_d(X, z_hat, n_times_atom, lambd0=lambd0,
                                   ds_init=d_hat, verbose=verbose,
                                   solver_kwargs=solver_d_kwargs,
                                   sample_weights=sample_weights)
            times.append(time.time() - start)

            # monitor cost function
            pobj.append(
                compute_X_and_objective(X, z_hat, d_hat, reg, sample_weights))
            if verbose > 1:
                print('[seed %s] Objective (d) %0.8f' % (random_state,
                                                         pobj[-1]))

            if callable(callback):
                callback(X, d_hat, z_hat, reg)

            if stopping_pobj is not None and pobj[-1] < stopping_pobj:
                break

    return pobj, times, d_hat, z_hat
