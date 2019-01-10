import numpy as np


def soft_thresholding(x, mu, positive=True):
    if positive:
        return np.maximum(x - mu, 0)
    else:
        return np.sign(x) * np.maximum(abs(x) - mu, 0)


def projection_l2_ball(x, inplace=False, return_norm=False):
    sum_axis = tuple(range(1, x.ndim))
    if x.ndim == 3:
        norm_x = np.maximum(1, np.linalg.norm(x, axis=sum_axis, keepdims=True))
    else:
        norm_x = np.sqrt(np.sum(x * x, axis=sum_axis, keepdims=True))

    if inplace:
        x /= norm_x
    else:
        x = x / norm_x

    if return_norm:
        return x, norm_x.squeeze()
    else:
        return x
