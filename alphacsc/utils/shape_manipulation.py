import numpy as np


def get_valid_shape(sig_shape, atom_shape):
    """Get the valid shape from sig_shape and atom_shape.
    """
    return tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_shape, atom_shape)
    ])


def get_full_shape(valid_shape, atom_shape):
    """Get the full shape from valid_shape and atom_shape.
    """
    return tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_shape, atom_shape)
    ])


def check_shape(atom_shape):
    if isinstance(atom_shape, int):
        return (atom_shape,)
    elif np.iterable(atom_shape):
        return tuple(atom_shape)
    else:
        raise ValueError("atom_shape should either be an integer or a tuple")