import os
import matplotlib.pyplot as plt

MANDRIL = "images/standard_images/mandril_color.tif"


def load_data(image_path=MANDRIL):
    """Load an image and correctly set its axis for DICOD_CDL.

    Parameters
    ----------
    random_state : int | None
        State to seed the random number generator

    Return
    ------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Simulated 10Hz sinusoidal signals with 10Hz mu-wave between 2sec and
        [3, 3.5]sec. some random phases are applied to the different part of
        the signal.
    info : dict
        Contains the topomap 'u' associated to each component of the signal.
    """
    data_dir = os.environ.get("DATA_DIR", "../../data")
    mandril = os.path.join(data_dir, MANDRIL)
    X = plt.imread(mandril) / 255
    X = X.swapaxes(0, 2)
    return X, None


def plot_image(X, ax=None):
    if ax is None:
        ax = plt.subplot(111)

    ax.imshow(X.swapaxes(0, 2))
    ax.axis('off')

if __name__ == "__main__":

    X, info = load_data()
    plot_image(X)
    plt.tight_layout()
    plt.show()
