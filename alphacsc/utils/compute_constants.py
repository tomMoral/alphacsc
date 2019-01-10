import numba
import numpy as np

from .shape_manipulation import get_valid_shape


def compute_DtD(D, n_channels=None):
    """Compute the DtD matrix
    """
    if D.ndim == 2:
        return _compute_DtD_uv(D, n_channels)

    return _compute_DtD_D(D)


@numba.jit((numba.float64[:, :], numba.int64), nopython=True, cache=True)
def _compute_DtD_uv(uv, n_channels):  # pragma: no cover
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_times_atom = uv.shape
    n_times_atom -= n_channels

    u = uv[:, :n_channels]

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(uv[k0, n_channels:],
                                            uv[k, n_channels:])
                else:
                    DtD[k0, k, t0 + t] = np.dot(uv[k0, n_channels:-t],
                                                uv[k, n_channels + t:])
                    DtD[k0, k, t0 - t] = np.dot(uv[k0, n_channels + t:],
                                                uv[k, n_channels:-t])
    DtD *= np.dot(u, u.T).reshape(n_atoms, n_atoms, 1)
    return DtD


@numba.jit((numba.float64[:, :, :],), nopython=True, cache=True)
def _compute_DtD_D(D):  # pragma: no cover
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_channels, n_times_atom = D.shape

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(D[k0].ravel(), D[k].ravel())
                else:
                    DtD[k0, k, t0 + t] = np.dot(D[k0, :, :-t].ravel(),
                                                D[k, :, t:].ravel())
                    DtD[k0, k, t0 - t] = np.dot(D[k0, :, t:].ravel(),
                                                D[k, :, :-t].ravel())
    return DtD


def compute_ztz(z, atom_support):
    if z.ndim == 3:
        return _compute_ztz_numba(z, atom_support[0])

    n_trials, n_atoms, *valid_shape = z.shape
    ztz_support = tuple(2 * np.array(atom_support) - 1)
    ztz_shape = (n_atoms, n_atoms) + ztz_support

    padding_shape = [(size_atom_ax - 1, size_atom_ax - 1)
                     for size_atom_ax in atom_support]

    padding_shape = np.asarray([(0, 0), (0, 0)] + padding_shape, dtype='i')
    z_pad = np.pad(z, padding_shape, mode='constant')
    ztz = np.empty(ztz_shape)
    for i in range(ztz.size):
        i0 = k0, k1, *pt = np.unravel_index(i, ztz.shape)
        zk1_slice = tuple([slice(None), k1] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, valid_shape)])
        ztz[i0] = np.dot(z[:, k0].ravel(), z_pad[zk1_slice].ravel())
    return ztz


@numba.jit((numba.float64[:, :, :], numba.int64), nopython=True,
           cache=True)
def _compute_ztz_numba(z, n_times_atom):  # pragma: no cover
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_trials, n_atoms, n_times_valid = z.shape

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for i in range(n_trials):
        for k0 in range(n_atoms):
            for k in range(n_atoms):
                for t in range(n_times_atom):
                    if t == 0:
                        ztz[k0, k, t0] += (z[i, k0] * z[i, k]).sum()
                    else:
                        ztz[k0, k, t0 + t] += (
                            z[i, k0, :-t] * z[i, k, t:]).sum()
                        ztz[k0, k, t0 - t] += (
                            z[i, k0, t:] * z[i, k, :-t]).sum()
    return ztz


def compute_ztX(z, X):
    """
    z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ztX.shape = n_atoms, n_channels, n_times_atom
    """
    n_trials, n_atoms, *valid_shape = z.shape
    _, n_channels, *sig_shape = X.shape
    atom_shape = get_valid_shape(sig_shape, valid_shape)

    ztX = np.zeros((n_atoms, n_channels, *atom_shape))
    for pt in zip(*z.nonzero()):
        n, k, *pt = pt
        patch_slice = tuple([n, slice(None)] + [
            slice(v, v+size_ax) for v, size_ax in zip(pt, atom_shape)
        ])
        ztX[k, :, :] += z[(n, k , *pt)] * X[patch_slice]

    return ztX
