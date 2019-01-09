import numpy as np


def soft_thresholding(x, mu, positive=True):
    if positive:
        return np.maximum(x - mu, 0)
    else:
        return np.sign(x) * np.maximum(abs(x) - mu, 0)
