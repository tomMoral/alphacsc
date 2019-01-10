"""Convolutional utilities for dictionary learning"""

# Authors: Thomas Moreau <thomas.moreau@inria.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numba
import numpy as np
from scipy import signal

from .. import cython_code
from . import check_1d_convolution
from .lil import get_z_shape, is_lil, is_list_of_lil


def construct_X(z, ds):
    """
    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times_valid)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atoms

    Returns
    -------
    X : array, shape (n_trials, n_times)
    """
    assert z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve(z[:, i], ds)
    return X


def construct_X_multi(z, D=None, n_channels=None):
    """
    Parameters
    ----------
    z : array, shape (n_trials, n_atoms, *valid_shape)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, *atom_support) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels

    Returns
    -------
    X : array, shape (n_trials, n_channels, *sig_shape)
    """
    n_trials, n_atoms, *valid_shape = get_z_shape(z)
    assert n_atoms == D.shape[0]
    if D.ndim == 2:
        check_1d_convolution(valid_shape)
        atom_support = (D.shape[1] - n_channels,)
    else:
        _, n_channels, *atom_support = D.shape
    sig_shape = tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_support)])

    X = np.zeros((n_trials, n_channels, *sig_shape))
    for i in range(n_trials):
        X[i] = _choose_convolve_multi(z[i], D=D, n_channels=n_channels)
    return X


def _sparse_convolve(z_i, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, *atom_support = ds.shape
    n_atoms, *valid_shape = z_i.shape
    sig_shape = tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_support)])
    Xi = np.zeros(sig_shape)
    for zik, dk in zip(z_i, ds):
        for nnz in zip(*zik.nonzero()):
            X_slice = tuple([slice(v, v + size_atom_ax)
                             for v, size_atom_ax in zip(nnz, atom_support)])
            Xi[X_slice] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi(z_i, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_channels, *atom_support = ds.shape
    n_atoms, *valid_shape = z_i.shape
    sig_shape = tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_support)])
    Xi = np.zeros(shape=(n_channels, *sig_shape))
    for zik, dk in zip(z_i, ds):
        for nnz in zip(*zik.nonzero()):
            X_slice = (Ellipsis,) + tuple([
                slice(v, v + size_atom_ax)
                for v, size_atom_ax in zip(nnz, atom_support)])
            Xi[X_slice] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi_uv(z_i, uv, n_channels):
    """Same as _dense_convolve, but use the sparsity of zi."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, *valid_shape = z_i.shape
    n_atoms, *atom_support = v.shape
    sig_shape = tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_support)])
    Xi = np.zeros(shape=(n_channels, *sig_shape))
    for zik, uk, vk in zip(z_i, u, v):
        zik_vk = np.zeros(*sig_shape)
        for nnz in zip(*zik.nonzero()):
            X_slice = tuple([slice(v, v + size_atom_ax)
                             for v, size_atom_ax in zip(nnz, atom_support)])
            zik_vk[X_slice] += zik[nnz] * vk

        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _dense_convolve(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    if z_i.ndim == 2:
        return sum([np.convolve(zik, dk) for zik, dk in zip(z_i, ds)], 0)
    else:
        return sum([signal.convolve(zik, dk) for zik, dk in zip(z_i, ds)], 0)


def _dense_convolve_multi(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    if z_i.ndim == 2:
        return np.sum([[np.convolve(zik, dkp) for dkp in dk]
                       for zik, dk in zip(z_i, ds)], 0)
    else:
        return np.sum([[signal.convolve(zik, dkp) for dkp in dk]
                       for zik, dk in zip(z_i, ds)], 0)


def _dense_convolve_multi_uv(z_i, uv, n_channels):
    """Convolve z_i[k] and uv[k] for each atom k, and return the sum."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = z_i.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros((n_channels, n_times))
    for zik, uk, vk in zip(z_i, u, v):
        if zik.ndim == 1:
            zik_vk = np.convolve(zik, vk)
        else:
            zik_vk = signal.convolve(zik, vk)
        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _choose_convolve(z_i, ds):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert z_i.shape[0] == ds.shape[0]

    if np.sum(z_i != 0) < 0.01 * z_i.size:
        return _sparse_convolve(z_i, ds)
    else:
        return _dense_convolve(z_i, ds)


def _choose_convolve_multi(z_i, D=None, n_channels=None):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels
    """
    assert z_i.shape[0] == D.shape[0]

    if is_lil(z_i):
        cython_code._assert_cython()
        if D.ndim == 2:
            return cython_code._fast_sparse_convolve_multi_uv(
                z_i, D, n_channels, compute_D=True)
        else:
            return cython_code._fast_sparse_convolve_multi(z_i, D)

    elif np.sum(z_i != 0) < 0.01 * z_i.size:
        if D.ndim == 2:
            return _sparse_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _sparse_convolve_multi(z_i, D)

    else:
        if D.ndim == 2:
            return _dense_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _dense_convolve_multi(z_i, D)


@numba.jit((numba.float64[:, :, :], numba.float64[:, :]), cache=True,
           nopython=True)
def numpy_convolve_uv(ztz, uv):
    """Compute the multivariate (valid) convolution of ztz and D

    Parameters
    ----------
    ztz: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    uv: array, shape = (n_atoms, n_channels + n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    assert uv.ndim == 2
    n_times_atom = (ztz.shape[2] + 1) // 2
    n_atoms = ztz.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:][:, ::-1]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for t in range(n_times_atom):
                G[k0, :, t] += (
                    np.sum(ztz[k0, k1, t:t + n_times_atom] * v[k1]) * u[k1, :])

    return G


def tensordot_convolve(ztz, D):
    """Compute the multivariate (valid) convolution of ztz and D

    Parameters
    ----------
    ztz: array, shape = (n_atoms, n_atoms, *(2 * atom_support - 1))
        Activations
    D: array, shape = (n_atoms, n_channels, atom_support)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, *atom_support)
        Gradient
    """
    n_atoms, n_channels, *atom_support = D.shape

    n_time_support = np.prod(atom_support)

    G = np.zeros(D.shape)
    axis_sum = list(range(2, D.ndim))
    D_revert = np.flip(D, axis=axis_sum)
    for t in range(n_time_support):
        pt = np.unravel_index(t, atom_support)
        ztz_slice = tuple(
            [Ellipsis] +
            [slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
        G[(Ellipsis, *pt)] = np.tensordot(
            ztz[ztz_slice], D_revert, axes=([1] + axis_sum, [0] + axis_sum))
    return G


def sort_atoms_by_explained_variances(D_hat, z_hat, n_channels):
    n_atoms = D_hat.shape[0]
    assert z_hat.shape[1] == n_atoms
    variances = np.zeros(n_atoms)
    for kk in range(n_atoms):
        variances[kk] = construct_X_multi(z_hat[:, kk:kk + 1],
                                          D_hat[kk:kk + 1],
                                          n_channels=n_channels).var()
    order = np.argsort(variances)[::-1]
    z_hat = z_hat[:, order, :]
    D_hat = D_hat[order, ...]
    return D_hat, z_hat


def dense_transpose_convolve_z(residual, z):
    """Convolve residual[i] with the transpose for each atom k, and return the sum

    Parameters
    ----------
    residual : array, shape (n_trials, n_channels, n_times)
    z : array, shape (n_trials, n_atoms, n_times_valid)

    Return
    ------
    grad_D : array, shape (n_atoms, n_channels, n_times_atom)

    """
    if is_list_of_lil(z):
        raise NotImplementedError()

    if z.ndim == 3:
        correlation_op = np.correlate
    else:
        correlation_op = signal.correlate

    return np.sum([[[correlation_op(res_ip, z_ik, mode='valid')  # n_times_atom
                     for res_ip in res_i]                        # n_channels
                    for z_ik in z_i]                             # n_atoms
                   for z_i, res_i in zip(z, residual)], axis=0)  # n_trials


def dense_transpose_convolve_d(residual_i, D=None, n_channels=None):
    """Convolve residual[i] with the transpose for each atom k

    Parameters
    ----------
    residual_i : array, shape (n_channels, n_times)
    D : array, shape (n_atoms, n_channels, n_times_atom) or
               shape (n_atoms, n_channels + n_times_atom)

    Return
    ------
    grad_zi : array, shape (n_atoms, n_times_valid)

    """

    if D.ndim == 2:
        # multiply by the spatial filter u
        uR_i = np.dot(D[:, :n_channels], residual_i)

        # Now do the dot product with the transpose of D (D.T) which is
        # the conv by the reversed filter (keeping valid mode)
        return np.array([
            np.correlate(uR_ik, v_k, 'valid')
            for (uR_ik, v_k) in zip(uR_i, D[:, n_channels:])
        ])
    else:
        if D.ndim == 3:
            correlation_op = np.correlate
        else:
            correlation_op = signal.correlate
        return np.sum([[
            correlation_op(res_ip, d_kp, mode='valid')  # n_times_valid
            for res_ip, d_kp in zip(residual_i, d_k)]   # n_channels
            for d_k in D], axis=1)                      # n_atoms
