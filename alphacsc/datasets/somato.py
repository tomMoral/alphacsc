import os
import mne
import numpy as np
from joblib import Memory
from scipy.signal import tukey

mem = Memory(location='.', verbose=0)


@mem.cache(ignore=['n_jobs'])
def load_data(n_splits=10, sfreq=None, epoch=None, filter_params=[2., None],
              n_jobs=1):
    """Load and prepare the somato dataset for multiCSC


    Parameters
    ----------
    n_splits : int
        Split the signal in n_split signals of same length before returning it.
        If epoch is provided, the signal is instead splitted according to the
        epochs and this option is not followed.
    sfreq : float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    filter_params : tuple, (default: [2., None])
        Frequency cut for a band pass filter applied to the signals. The
        default is a high-pass filter with frequency cut at 2Hz.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.

    Return
    ------
    X : ndarray, shape (n_splits, n_channels, n_times)
        Signals loaded from HCP
    info : mne.Info
        Info related to each signals.
    """
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
    raw.filter(*filter_params, n_jobs=n_jobs)
    event_id = 1

    t_min = -2
    if epoch:
        t_min, t_max = epoch
        baseline = (None, 0)
        events = mne.find_events(raw, stim_channel='STI 014')

        picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                               stim=False)
        epochs = mne.Epochs(raw, events, event_id, t_min, t_max, picks=picks,
                            baseline=baseline, reject=dict(
                                grad=4000e-13, eog=350e-6), preload=True)
        epochs.pick_types(meg='grad', eog=False)
        if sfreq is not None:
            epochs.resample(sfreq, npad='auto')
        X = epochs.get_data()
        info = epochs.info

    else:
        raw.pick_types(meg='grad', eog=False, stim=True)
        if sfreq is not None:
            raw.resample(sfreq, npad='auto', n_jobs=n_jobs)

        events = mne.find_events(raw, stim_channel='STI 014')
        raw.pick_types(meg='grad', stim=False)
        events[:, 0] -= raw.first_samp

        X = raw.get_data()
        n_channels, n_times = X.shape
        n_times = n_times // n_splits
        X = X[:, :n_splits * n_times]
        X = X.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)
        info = raw.info

    # XXX: causes problems when saving EvokedArray
    info['t_min'] = t_min
    info['event_id'] = event_id
    info['events'] = events

    # define n_channels, n_splits, n_times
    n_splits, n_channels, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X, info
