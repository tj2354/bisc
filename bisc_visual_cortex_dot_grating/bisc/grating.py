from typing import Optional
from pathlib import Path
import numpy as np

from jarvis.utils import tqdm

from . import rcParams
from .data import get_stim, get_valid_idxs, get_triggered_average
from .cache import cached

CACHE_PATH = Path(__file__).parent.parent/rcParams.cache_path


@cached(CACHE_PATH)
def grating_triggered_average(
    session: dict,
    *,
    trial_idxs: Optional[set[int]] = None,
    **kwargs,
):
    r"""Returns grating triggered average responses.

    Args
    ----
    session:
        A session key, e.g. from the list returned by `get_sessions`.
    trial_idxs:
        Set of trial indices to be averaged. If ``None``, use all valid trials.
    kwargs:
        Keyword arguments for `get_triggered_average`.

    """
    if trial_idxs is None:
        trial_idxs, _ = get_valid_idxs(session)
    else:
        trial_idxs = list(trial_idxs)
    onsets, offsets, stim_params = get_stim(session)
    orientations = stim_params['orientations']
    num_gratings = orientations.shape[1]

    grating_duration = (offsets-onsets)[trial_idxs].mean()/num_gratings
    grating_duration = round(grating_duration/5)*5
    thetas = sorted(list(np.unique(orientations)))

    triggers = np.empty(len(thetas), dtype=object)
    for i in range(len(thetas)):
        triggers[i] = set()
    for trial_idx in trial_idxs:
        for grating_idx in range(num_gratings):
            i = thetas.index(orientations[trial_idx, grating_idx])
            triggers[i].add((trial_idx, grating_idx*grating_duration))

    taus = None
    responses_mean = np.empty(triggers.shape, dtype=object)
    responses_std = np.empty(triggers.shape, dtype=object)

    rng = np.random.default_rng()
    for i in tqdm(rng.permutation(len(thetas)), unit='orien', leave=False):
        try:
            _taus, responses_mean[i], responses_std[i] = get_triggered_average(
                session, triggers=triggers[i], **kwargs,
            )
            assert np.unique(np.diff(_taus)).size==1
        except:
            print('session: {}, orientation: {}, kwargs:\n{}'.format(session, thetas[i], kwargs))
            raise
        if taus is None:
            taus = _taus
        else:
            assert np.all(_taus==taus)
    responses_mean = np.stack([*responses_mean])
    responses_std = np.stack([*responses_std])
    counts = np.array([len(t) for t in triggers])
    return thetas, taus, responses_mean, responses_std, counts
