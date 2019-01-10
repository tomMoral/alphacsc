# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import time

import numpy as np
from scipy import optimize, sparse
from joblib import Parallel, delayed


from . import cython_code
from .utils.optim import fista
from .utils import check_random_state
from .utils import check_1d_convolution
from .loss_and_gradient import gradient_zi
from .utils.lil import is_list_of_lil, is_lil
from .utils.prox_operator import soft_thresholding
from .utils.coordinate_descent import _coordinate_descent_idx
from .utils.compute_constants import compute_DtD, compute_ztz, compute_ztX


def update_z_multi(X, D, reg, z0=None, solver='l-bfgs', solver_kwargs=dict(),
                   loss='l2', loss_params=dict(), freeze_support=False,
                   z_positive=True, return_ztz=False, timing=False, n_jobs=1,
                   random_state=None, debug=False):
    """Update z using L-BFGS with positivity constraints

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data array
    D : array, shape (n_atoms, n_channels + n_times_atom)
        The dictionary used to encode the signal X. Can be either in the form
        f a full rank dictionary D (n_atoms, n_channels, n_times_atom) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + n_times_atom).
    reg : float
        The regularization constant
    z0 : None | array, shape (n_trials, n_atoms, *valid_shape) |
         list of sparse lil_matrices, shape (n_atoms, *valid_shape)
        Init for z (can be used for warm restart).
    solver : 'l-bfgs' | "lgcd"
        The solver to use.
    solver_kwargs : dict
        Parameters for the solver
    loss : 'l2' | 'dtw' | 'whitening'
        The data fit loss, either classical l2 norm or the soft-DTW loss.
    loss_params : dict
        Parameters of the loss
    freeze_support : boolean
        If True, the support of z0 is frozen.
    z_positive : boolean
        If True, constrain the coefficients in z to be positive.
    return_ztz : boolean
        If True, returns the constants ztz and ztX, used to compute D-updates.
    timing : boolean
        If True, returns the cost function value at each iteration and the
        time taken by each iteration for each signal.
    n_jobs : int
        The number of parallel jobs.
    random_state : None or int or RandomState
        random_state to make randomized experiments determinist. If None, no
        random_state is given. If it is an integer, it will be used to seed a
        RandomState.
    debug : bool
        If True, check the gradients.

    Returns
    -------
    z : array, shape (n_trials, n_atoms, *valid_shape)
        The true codes.
    """
    n_trials, n_channels, *sig_shape = X.shape
    if D.ndim == 2:
        check_1d_convolution(sig_shape)
        n_atoms, n_channels_n_times_atom = D.shape
        atom_support = (n_channels_n_times_atom - n_channels,)
    else:
        n_atoms, n_channels, *atom_support = D.shape
    valid_shape = tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_shape, atom_support)])

    rng = check_random_state(random_state)
    parallel_seeds = [rng.randint(2**32) for _ in range(n_trials)]

    # now estimate the codes
    delayed_update_z = delayed(_update_z_multi_idx)

    if z0 is None:
        z0 = np.zeros((n_trials, n_atoms, *valid_shape))

    results = Parallel(n_jobs=n_jobs)(
        delayed_update_z(X[i], D, reg, z0[i], debug, solver, solver_kwargs,
                         freeze_support, loss, loss_params=loss_params,
                         z_positive=z_positive, return_ztz=return_ztz,
                         timing=timing, random_state=seed)
        for i, seed in enumerate(parallel_seeds))

    # Post process the results to get separate objects
    z_hats, pobj, times = [], [], []
    if loss == 'l2' and return_ztz:
        ztz_shape = tuple([2 * size_atom_ax - 1
                           for size_atom_ax in atom_support])
        ztz = np.zeros((n_atoms, n_atoms, *ztz_shape))
        ztX = np.zeros((n_atoms, n_channels, *atom_support))
    else:
        ztz, ztX = None, None
    for z_hat, ztz_i, ztX_i, pobj_i, times_i in results:
        z_hats.append(z_hat), pobj.append(pobj_i), times.append(times_i)
        if loss == 'l2' and return_ztz:
            ztz += ztz_i
            ztX += ztX_i

    # If z_hat is a ndarray, stack and reorder the columns
    if not is_list_of_lil(z0):
        z_hats = np.array(z_hats).reshape(n_trials, n_atoms, *valid_shape)

    return z_hats, ztz, ztX


class BoundGenerator(object):
    def __init__(self, length):
        self.length = length
        self.current_index = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == self.length:
            raise StopIteration()
        self.current_index += 1
        return (0, np.inf)


def _update_z_multi_idx(X_i, D, reg, z0_i, debug, solver='l-bfgs',
                        solver_kwargs=dict(), freeze_support=False, loss='l2',
                        loss_params=dict(), z_positive=True, return_ztz=False,
                        timing=False, random_state=None):
    t_start = time.time()
    n_channels, *sig_shape = X_i.shape
    if D.ndim == 2:
        check_1d_convolution(sig_shape)
        n_atoms, n_channels_n_times_atom = D.shape
        atom_support = (n_channels_n_times_atom - n_channels,)
    else:
        n_atoms, n_channels, *atom_support = D.shape

    valid_shape = tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_shape, atom_support)])

    assert not (freeze_support and z0_i is None), 'Impossible !'
    if freeze_support and solver == "dicod":
        solver = "lgcd"

    if is_lil(z0_i) and solver != "lgcd":
        raise NotImplementedError()

    rng = check_random_state(random_state)

    constants = {}
    if solver == "lgcd":
        constants['DtD'] = compute_DtD(D=D, n_channels=n_channels)
    init_timing = time.time() - t_start

    def func_and_grad(zi):
        return gradient_zi(Xi=X_i, zi=zi, D=D, constants=constants,
                           reg=reg, return_func=True, flatten=True,
                           loss=loss, loss_params=loss_params)

    if z0_i is None:
        z0_i = np.zeros(n_atoms, *valid_shape)

    times, pobj = None, None
    if timing:
        times = [init_timing]
        pobj = [func_and_grad(z0_i)[0]]
        t_start = [time.time()]

    if solver == 'l-bfgs':
        msg = "solver 'l-bfgs' can only be used with positive z"
        assert z_positive is True, msg
        z0_i = z0_i.ravel()
        if freeze_support:
            bounds = [(0, 0) if z == 0 else (0, None) for z in z0_i]
        else:
            bounds = BoundGenerator(n_atoms * np.prod(valid_shape))
        if timing:
            def callback(xk):
                times.append(time.time() - t_start[0])
                pobj.append(func_and_grad(xk)[0])
                # use a reference to have access inside this function
                t_start[0] = time.time()
        else:
            callback = None
        factr = solver_kwargs.get('factr', 1e15)  # default value
        maxiter = solver_kwargs.get('maxiter', 15000)  # default value
        z_hat, f, d = optimize.fmin_l_bfgs_b(
            func_and_grad, x0=z0_i, fprime=None, args=(), approx_grad=False,
            bounds=bounds, factr=factr, maxiter=maxiter, callback=callback)

    elif solver in ("ista", "fista"):
        # Default args
        fista_kwargs = dict(
            max_iter=100, eps=None, verbose=0, scipy_line_search=False,
            momentum=(solver == "fista")
        )
        fista_kwargs.update(solver_kwargs)

        def objective(z_hat):
            return func_and_grad(z_hat)[0]

        def grad(z_hat):
            return func_and_grad(z_hat)[1]

        def prox(z_hat, step_size=0):
            return soft_thresholding(z_hat, step_size * reg,
                                     positive=z_positive)

        output = fista(objective, grad, prox, step_size=None, x0=z0_i,
                       adaptive_step_size=True, timing=timing,
                       name="Update z", **fista_kwargs)
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat, pobj = output

    elif solver == "lgcd":
        if not sparse.isspmatrix_lil(z0_i):
            z0_i = z0_i.reshape(n_atoms, *valid_shape)

        # Default values
        tol = solver_kwargs.get('tol', 1e-1)
        n_seg = solver_kwargs.get('n_seg', 'auto')
        max_iter = solver_kwargs.get('max_iter', 1e15)
        strategy = solver_kwargs.get('strategy', 'greedy')
        output = _coordinate_descent_idx(
            X_i, D, constants, reg=reg, z0=z0_i, max_iter=max_iter, tol=tol,
            strategy=strategy, n_seg=n_seg, z_positive=z_positive,
            freeze_support=freeze_support, timing=timing, random_state=rng,
            name="Update z")
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat = output

    elif solver == "dicod":
        try:
            from dicod_python.dicod import dicod
        except ImportError:
            raise NotImplementedError("You need to install the dicod package "
                                      "to be able to use z_solver='dicod'.")

        assert loss == 'l2', "DICOD is only implemented for the l2 loss"

        tol = solver_kwargs.get('tol', 1e-1)
        n_seg = solver_kwargs.get('n_seg', 'auto')
        n_jobs = solver_kwargs.get('n_jobs', 1)
        hostfile = solver_kwargs.get('hostfile', '')
        max_iter = int(solver_kwargs.get('max_iter', 1e9))
        strategy = solver_kwargs.get('strategy', 'greedy')
        z_hat, ztz, ztX, pobj = dicod(
            X_i, D, reg=reg, z0=z0_i, n_seg=n_seg, strategy=strategy,
            n_jobs=n_jobs, hostfile=hostfile, tol=tol, max_iter=max_iter,
            z_positive=z_positive, return_ztz=return_ztz, timing=timing,
            random_state=random_state, verbose=1
        )
    else:
        raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                         " 'l-bfgs', or 'lgcd'." % solver)

    if not is_lil(z_hat):
        z_hat = z_hat.reshape(n_atoms, *valid_shape)

    if loss == 'l2' and return_ztz:
        if solver != 'dicod':
            if not is_lil(z_hat):
                ztz = compute_ztz(z_hat[None], atom_support)
                ztX = compute_ztX(z_hat[None], X_i[None])
            else:
                cython_code._assert_cython()
                ztz = cython_code._fast_compute_ztz([z_hat], atom_support[0])
                ztX = cython_code._fast_compute_ztX([z_hat], X_i[None])
    else:
        ztz, ztX = None, None

    return z_hat, ztz, ztX, pobj, times
