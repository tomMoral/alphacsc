
import os
import mne
import numpy as np
from joblib import Memory
from scipy.signal import tukey


mem = Memory(cachedir='.', verbose=0)


@mem.cache(ignore=['n_jobs'])
def load_data(sfreq=None, epoch=True, n_jobs=1, filt=[2., None], n_trials=10,
              return_epochs=False):
    """Load and prepare the somato dataset for multiCSC


    Parameters
    ----------
    sfreq: float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : boolean
        If set to True, extract epochs from the raw data. Else, use the raw
        signal, divided in 10 chunks.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.
    return_epochs : boolean
        If True, return epochs instead of X and info
    """
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
    raw.filter(*filt, n_jobs=n_jobs)
    events = mne.find_events(raw, stim_channel='STI 014')
    event_id, t_min, t_max = 1, -2., 4.

    if epoch:

        baseline = (None, 0)
        picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                               stim=False)

        epochs = mne.Epochs(raw, events, event_id, t_min, t_max,
                            picks=picks, baseline=baseline, reject=dict(
                                grad=4000e-13, eog=350e-6), preload=True)
        epochs.pick_types(meg='grad', eog=False)
        if sfreq is not None:
            epochs.resample(sfreq, npad='auto')
        X = epochs.get_data()
        info = epochs.info
        if return_epochs:
            return epochs

    else:
        raw.pick_types(meg='grad', eog=False)
        if sfreq is not None:
            raw.resample(sfreq, npad='auto', n_jobs=n_jobs)
        X = raw.get_data()
        T = X.shape[-1]
        n_times = T // n_trials
        X = np.array([X[:, i * n_times:(i + 1) * n_times]
                      for i in range(n_trials)])
        info = raw.info
        if return_epochs:
            raise ValueError('return_epochs=True is not allowed with '
                             'epochs=False')

    events[:, 0] -= raw.first_samp

    # XXX: causes problems when saving EvokedArray
    # info['t_min'] = t_min
    # info['event_id'] = event_id
    # info['events'] = events

    # define n_channels, n_trials, n_times
    n_trials, n_channels, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X, info
