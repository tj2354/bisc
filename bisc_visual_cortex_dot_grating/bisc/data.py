from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import morlet
import yaml, random, h5py
import datajoint as dj
from typing import Optional, Union

from jarvis.utils import tqdm
from . import rcParams
from .alias import Array
from .cache import cached

STIMULUS_PATH = Path(__file__).parent.parent/rcParams.stimulus_path
RESPONSE_PATH = Path(__file__).parent.parent/rcParams.response_path
CACHE_PATH = Path(__file__).parent.parent/rcParams.cache_path

with open(Path(__file__).parent.parent/'scripts/sessions.yaml', 'r') as f:
    METAS = yaml.safe_load(f)


def get_acq():
    acq = dj.create_virtual_module('acq', 'acq')
    return acq


def fetch_sessions(
    date: str, animal: str = 'Paul', min_trials: int = 100,
) -> list[dict]:
    r"""Returns session keys.

    Args
    ----
    date:
        A string with format 'yyyy-mm-dd', for the day of experiment.
    animal:
        Name of the animal.
    min_trials:
        Minimum number of correct trials.

    Returns
    -------
    sessions:
        A list of session keys.

    """
    acq = get_acq()
    sessions = (
        acq.Sessions*acq.Stimulation*acq.Ephys
        & f'session_path like "%{animal}/{date}%"'
        & f'correct_trials>={min_trials}'
    ).fetch('KEY')
    return sessions


def fetch_offset_and_slope(session: dict) -> tuple[float, float]:
    r"""Fetches sync parameters."""
    acq = get_acq()
    offset, slope = (acq.NPTimeToTimestamps & session).fetch1('ephys_offset', 'ephys_slope')
    return offset, slope


def fetch_spath_str(session: dict) -> str:
    r"""Fetches stimulus file name."""
    acq = get_acq()
    folder_name = (acq.Stimulation&session).fetch1('stim_path').split('/')[-1]
    exp_type = (acq.Stimulation&session).fetch1('exp_type')
    spath = f'{folder_name}/{exp_type}Synched.mat'
    return spath


def fetch_epash_str(session: dict) -> str:
    r"""Fetches recording file name."""
    acq = get_acq()
    epath = (acq.Ephys & session).fetch1('ephys_path')
    epath = '/'.join(epath.split('/')[3:])
    return str(epath)


def get_session(session_id: str) -> dict:
    r"""Returns session key given the 8-digit ID."""
    for meta in METAS:
        if '{:08d}'.format(int(meta['session']['session_start_time']%1e8))==session_id:
            return meta['session']


def get_offset_and_slope(session: dict) -> tuple[float, float]:
    for meta in METAS:
        if meta['session']['session_start_time']==session['session_start_time']:
            offset = meta['offset']
            slope = meta['slope-1']+1
            return offset, slope


def get_spath(session: dict) -> Path:
    for meta in METAS:
        if meta['session']['session_start_time']==session['session_start_time']:
            return STIMULUS_PATH/meta['spath']


def get_epath(session: dict) -> Path:
    for meta in METAS:
        if meta['session']['session_start_time']==session['session_start_time']:
            return RESPONSE_PATH/meta['epath']


def get_stim(session: dict) -> tuple[Array, Array, dict[str, Union[int, Array]]]:
    r"""Returns stimulus information.

    Args
    ----
    session:
        Session key.

    Returns
    -------
    onsets, offsets: (num_trials,)
        Stimulus onset and offset of correct trials.
    stim_params:
        A dict containing arrays about stimulus details, the first dimension of
        each array should be trials, i.e. aligned with `onsets` and `offsets`.

    """
    stim = loadmat(get_spath(session))['stim'][0, 0]
    exp_type = stim['params'][0, 0]['constants'][0, 0]['expType'][0]
    stim_params = {'exp_type': exp_type}
    # correct trials defined by behavior
    trials = stim['params'][0, 0]['trials'][0]
    mask = np.array([trial['validTrial'][0, 0] for trial in trials], dtype=bool)
    trials = trials[mask]
    events = stim['events'][0, mask]
    stim_params.update({
        'num_total_trials': len(mask),
        'num_correct_trials': len(trials),
    })
    # get stimulus onsets and offsets from event timings
    onset_id = (stim['eventTypes'][:, 0]=='showStimulus').nonzero()[0][0]+1
    offset_id = (stim['eventTypes'][:, 0]=='endStimulus').nonzero()[0][0]+1
    onsets, offsets = [], []
    for event in events:
        idxs, = (event['types'][0]==onset_id).nonzero()
        assert idxs.size==1, "Multiple 'showStimulus' detected"
        onsets.append(event['times'][0, idxs[0]])
        idxs, = (event['types'][0]==offset_id).nonzero()
        assert idxs.size==1, "Multiple 'endStimulus' detected"
        offsets.append(event['times'][0, idxs[0]])
    onsets, offsets = np.array(onsets), np.array(offsets)
    # additional stimulus parameters for different experiment type
    if exp_type=='DotMappingExperiment':
        stim_params['dot_locations'] = np.stack([
            np.concatenate(trial['dotLocations'][0], axis=1).T for trial in trials
        ]) # (num_trials, num_dots, 2)
        stim_params['dot_colors'] = np.stack([
            np.concatenate(trial['dotColors'][0])[:, 0] for trial in trials
        ]) # (num_trials, num_dots)
    if exp_type=='MultDimExperiment':
        conditions = np.concatenate([ # grating conditions
            trial['conditions'] for trial in trials
        ])
        stim_params['orientations'] = np.array([
            stim['params'][0, 0]['conditions'][0, c-1]['orientation'][0, 0]
            for c in conditions.ravel()
        ]).reshape(conditions.shape) # (num_trials, num_gratings)
        stim_params['spatial_freqs'] = np.array([
            stim['params'][0, 0]['conditions'][0, c-1]['spatialFreq'][0, 0]
            for c in conditions.ravel()
        ]).reshape(conditions.shape) # (num_trials, num_gratings)
        stim_params['speeds'] = np.array([
            stim['params'][0, 0]['conditions'][0, c-1]['speed'][0, 0]
            for c in conditions.ravel()
        ]).reshape(conditions.shape) # (num_trials, num_gratings)
        stim_params['phases'] = np.array([
            stim['params'][0, 0]['conditions'][0, c-1]['initialPhase'][0, 0]
            for c in conditions.ravel()
        ]).reshape(conditions.shape) # (num_trials, num_gratings)
    if exp_type=='CenterSurroundExperiment':
        stim_params['img_ids'] = np.concatenate([
            trial['image_idx_used'].T for trial in trials
        ]).astype(int)
        stim_params['flags'] = np.concatenate([
            trial['trial_type']=='Train' for trial in trials
        ])
    return onsets, offsets, stim_params


def get_num_trials(session: dict) -> int:
    r"""Returns number of trials for analysis."""
    *_, stim_params = get_stim(session)
    num_trials: int = stim_params['num_correct_trials']
    return num_trials


def _get_ephys_file(session: dict) -> h5py.File:
    r"""Returns file handle to responses.

    BISC recordings are saved as 'electrophysiology' responses, in the format
    of a family of HDF5 files.

    Returns
    -------
    f:
        File handle for all h5 files, preferably used with 'with' statement.

    """
    f = h5py.File(get_epath(session), 'r', driver='family', memb_size=2**31)
    return f


def get_fs(session: dict) -> float:
    r"""Returns sampling frequency in Hz."""
    with _get_ephys_file(session) as f:
        fs = f.attrs['Fs'][0] # sampling frequency (Hz)
    return fs


def get_num_channels(session: dict) -> int:
    r"""Returns number of channels."""
    with _get_ephys_file(session) as f:
        num_channels = f['data'].shape[0]-1 # last row is syncing signal
    return num_channels


def get_raw_responses(
    session: dict, tic: float, toc: float, channel_idxs: Optional[list[int]] = None,
) -> Array:
    r"""Returns a segment of raw responses.

    Args
    ----
    session:
        Session key.
    tic, toc:
        Start and end of the time period of interest.
    channel_idxs:
        Indices of channels of interest, if not provided, return all channels.

    Returns
    -------
    response: (num_channels, num_samples)
        Raw response of the segment.

    """
    num_channels = get_num_channels(session)
    if channel_idxs is None:
        channel_idxs = list(range(num_channels))
    else:
        assert np.all([0<=c<num_channels for c in channel_idxs]), (
            f"Channel index must range in [0, {num_channels})"
        )
    fs = get_fs(session)
    offset, slope = get_offset_and_slope(session)
    tic_idx = int((tic-offset)/slope*fs/1000)
    toc_idx = int((toc-offset)/slope*fs/1000)
    try:
        with _get_ephys_file(session) as f:
            responses = f['data'][channel_idxs, tic_idx:toc_idx].astype(float)
    except:
        responses = np.zeros((len(channel_idxs), toc_idx-tic_idx))
    return responses


@cached(CACHE_PATH)
def get_valid_idxs(session: dict) -> tuple[Array, Array]:
    r"""Returns indices of valid trials and channels.

    Given a channel, response of a trial is considered as valid when the values
    are not too close to extreme values. A channel is considered valid when most
    of its responses are valid, and a trial is considered valid when all
    channels are valid.

    Args
    ----
    session:
        Session key.

    Returns
    -------
    trial_idxs: (num_trials,), int
        Indices of valid trials.
    channel_idxs: (num_channels,) int
        Indices of valid channels.

    """
    num_trials = get_num_trials(session)
    num_channels = get_num_channels(session)
    is_valid = np.full((num_trials, num_channels), fill_value=False, dtype=bool)

    margin = 8 # valid response value margin
    onsets, offsets, _ = get_stim(session)
    for trial_idx in tqdm(range(num_trials), unit='trial', desc='Check valid', leave=False):
        tic, toc = onsets[trial_idx], offsets[trial_idx]
        responses = get_raw_responses(session, tic, toc)
        is_valid[trial_idx] = ((responses>margin)&(responses<2**10-margin)).mean(axis=1)>0.999
    scores = np.mean(is_valid, axis=0)
    channel_idxs, = (scores>0.99*scores.max()).nonzero() # valid channels
    mask_0 = np.all(is_valid[:, channel_idxs], axis=1) # trials with all valid channels
    durations = offsets-onsets
    mask_1 = np.abs(durations-durations.mean())<3*durations.std() # trials with normal durations
    trial_idxs, = (mask_0&mask_1).nonzero()
    return trial_idxs, channel_idxs


@cached(CACHE_PATH)
def get_whitening_matrix(session: dict, freq=None) -> Array:
    r"""Returns whitening matrix for all valid channels."""
    num_channels = get_num_channels(session)
    trial_idxs, channel_idxs = get_valid_idxs(session)
    onsets, offsets, _ = get_stim(session)
    r_m1 = np.zeros((len(channel_idxs),)) # first-order moment
    r_m2 = np.zeros((len(channel_idxs), len(channel_idxs))) # second-order moment
    count = 0
    for trial_idx in tqdm(trial_idxs, desc='Gather stats', unit='trial', leave=False):
        if freq is None:
            responses = get_raw_responses(session, onsets[trial_idx], offsets[trial_idx], channel_idxs)
            responses = (responses-512)/1024
        else:
            _, responses = get_trial_responses(session, trial_idx=trial_idx, transform={'type': 'morlet', 'freq': freq})
        r_m1 += responses.sum(axis=1)
        r_m2 += np.matmul(responses, responses.T)
        count += responses.shape[1]
    r_mean = r_m1/count
    r_cov = r_m2/count-r_mean[:, None]*r_mean[None]
    u, s, _ = np.linalg.svd(r_cov, hermitian=True)
    _W = np.matmul(np.matmul(u, np.diag(s**-0.5)), u.T) # partial whitening
    W = np.eye(num_channels)
    for i, c_i in enumerate(channel_idxs):
        for j, c_j in enumerate(channel_idxs):
            W[c_i, c_j] = _W[i, j]
    return W


@cached(CACHE_PATH)
def get_trial_responses(
    session: dict,
    *,
    trial_idx: int, dt: float = 0.5,
    pre_trial: float = 800., post_trial: float = 400.,
    transform: Optional[dict] = None,
):
    r"""Returns processed responses of all channels for one trial.

    Args
    ----
    session:
        Session key.
    trial_idx:
        Index of a trial in the trials returned by `get_stim`.
    dt:
        Temporal resolution of returned responses, in milliseconds.
    pre_trial, post_trial:
        Time before and after one trial to be included, in milliseconds.
    transform:
        The transformation imposed on the raw responses.

    Returns
    -------
    taus: (num_taus,)
        Time axis of responses, relative to trial onset and in milliseconds.
    responses: (num_channels, num_taus)
        Transformed responses of all channels.

    """
    if transform is None:
        transform = {'type': 'remove_mean'}
    onsets, offsets, _ = get_stim(session)
    tic, toc = onsets[trial_idx]-pre_trial, offsets[trial_idx]+post_trial
    fs = get_fs(session)
    if transform.get('whitening', False):
        W = get_whitening_matrix(session, freq=transform.get('freq', None))
    else:
        num_channels = get_num_channels(session)
        W = np.eye(num_channels)
    if transform['type']=='remove_mean':
        responses = get_raw_responses(session, tic, toc)
        responses = np.matmul(W, responses)
        stamps = np.arange(responses.shape[1])/fs*1000+tic
        r_mean = responses[:, (stamps>onsets[trial_idx])&(stamps<offsets[trial_idx])].mean(axis=1)
        responses -= r_mean[:, None]
    if transform['type']=='morlet':
        freq = transform.get('freq') # center frequency (Hz)
        n_cycles = transform.get('n_cycles', 7)
        scaling = transform.get('scaling', 0.5)
        down = transform.get(
            'down', max(int(np.floor(fs/(max(2.5*freq, 1000/dt)))), 1),
        )
        # pad and down-sample before convolution
        padding = n_cycles/freq*500 # padding to avoid edge effect (ms)
        responses = get_raw_responses(session, tic-padding, toc+padding)
        responses = responses[:, :(responses.shape[1]//down)*down]
        responses = responses.reshape(responses.shape[0], -1, down).mean(axis=2)
        stamps = (np.arange(responses.shape[1])+0.5)*(1000/fs*down)+tic-padding
        responses = np.matmul(W, responses)
        # convolve with Morlet wavelet
        wavelet_size = int(padding/1000*fs/down)
        wavelet = morlet(2*wavelet_size+1, w=n_cycles, s=scaling)*freq
        responses = np.stack([
            np.abs(np.convolve(response, wavelet, 'valid')) for response in responses
        ])
        stamps = stamps[wavelet_size:-wavelet_size]
    stamps -= onsets[trial_idx]
    taus = np.arange(np.ceil(stamps[0]/dt), np.ceil(stamps[-1]/dt))*dt
    responses = np.stack([
        np.interp(taus, stamps, response) for response in responses
    ])
    return taus, responses


def align_responses(
    taus_list: list[Array], responses_list: list[Array],
) -> tuple[Array, Array]:
    r"""Aligns segments of responses with similar time axis.

    Args
    ----
    taus_list:
        A list of time stamps array, mostly similar to each other. Each element
        is a 1D array, and is assumed to use the same sampling frequency.
    responses_list:
        A list of responses, with the last dimension matching the corresponding
        time stamps in `taus_list`.

    Returns
    -------
    taus: (num_stamps,)
        The shared time stamp, which is the intersection of individual intervals.
    responses: (num_samples, *, num_stamps)
        Stacked responses, with the first dimension for samples, and last
        dimension for time.

    """
    dts = np.array([np.diff(taus).mean() for taus in taus_list])
    assert np.std(dts)<1e-5, "Sampling rate inconsistent (mean {}, std {})".format(np.mean(dts), np.std(dts))
    dt = dts.mean()
    tic = max([int(taus.min()/dt) for taus in taus_list])
    toc = min([int(taus.max()/dt) for taus in taus_list])
    responses = []
    for i, taus in enumerate(taus_list):
        _taus = (taus/dt).astype(int)
        responses.append(responses_list[i][..., (_taus>=tic)&(_taus<=toc)])
    try:
        responses = np.stack(responses)
    except:
        for i in range(len(responses)):
            print(responses[i].shape)
        raise
    taus = np.arange(tic, toc+1)*dt
    return taus, responses


@cached(CACHE_PATH)
def get_trial_average(
    session: dict,
    *,
    trial_idxs: Optional[set[int]] = None,
    **kwargs,
) -> tuple[Array, Array, Array]:
    r"""Returns trial average responses.

    Responses of each trial are fetched, and aligned to a shared time axis
    relative to trial onset.

    Args
    ----
    session:
        Session key.
    trial_idxs:
        Set of trial indices to be averaged. If ``None``, use all valid trials.
    kwargs:
        Keyword arguments for `get_trial_responses`.

    Returns
    -------
    taus: (num_stamps,)
        Time stamps relative to trial onset for all trials, in milliseconds.
    responses_mean, responses_std: (num_channels, num_stamps)
        Mean and standard deviation by averaging over all trials.

    """
    if trial_idxs is None:
        trial_idxs, _ = get_valid_idxs(session)
    else:
        trial_idxs = list(trial_idxs)
    random.shuffle(trial_idxs)
    taus_list, responses_list = [], []
    for trial_idx in tqdm(trial_idxs, unit='trial', leave=False):
        _taus, _responses = get_trial_responses(
            session, trial_idx=trial_idx, **kwargs,
        )
        taus_list.append(_taus)
        responses_list.append(_responses)
    taus, responses = align_responses(taus_list, responses_list)
    responses_mean = responses.mean(axis=0)
    responses_std = responses.std(axis=0)
    return taus, responses_mean, responses_std


@cached(CACHE_PATH)
def get_triggered_average(
    session: dict,
    *,
    triggers: set[tuple[int, float]],
    tau_min: float = -50., tau_max: float = 250.,
    **kwargs,
) -> tuple[Array, Array, Array]:
    r"""Returns triggered average response.

    Args
    ----
    session:
        Session key.
    triggers:
        A set of trigger times, in the format of `(trial_idx, shift)`, meaning
        time `shift` (ms) relative to the onset of trial `trial_idx`.
    kwargs:
        Keyword arguments for `get_trial_responses`.

    Returns
    -------
    taus: (num_stamps,)
        Time stamps relative to triggering event, in milliseconds.
    responses_mean, responses_std: (num_channels, num_stamps)
        Mean and standard deviation by averaging over events.

    """
    groups = {}
    for trial_idx, shift in triggers:
        if trial_idx in groups:
            groups[trial_idx].append(shift)
        else:
            groups[trial_idx] = [shift]
    taus_list, responses_list = [], []
    for trial_idx, shifts in tqdm(groups.items(), unit='trial', desc='Triggered average', leave=False):
        _taus, _responses = get_trial_responses(session, trial_idx=trial_idx, **kwargs)
        if min(shifts)+tau_min<_taus.min() or max(shifts)+tau_max>_taus.max():
            continue
        for shift in shifts:
            mask = (_taus>=shift+tau_min)&(_taus<=shift+tau_max)
            taus_list.append(_taus[mask]-shift)
            responses_list.append(_responses[:, mask])
    taus, responses = align_responses(taus_list, responses_list)
    responses_mean = responses.mean(axis=0)
    responses_std = responses.std(axis=0)
    return taus, responses_mean, responses_std


@cached(CACHE_PATH)
def get_baseline(
    session: dict,
    *,
    tau_min: float = -300., tau_max: float = 0.,
    **kwargs,
) -> tuple[Array, Array]:
    r"""Returns baseline response and across-trial deviation.

    Args
    ----
    session:
        Session key.
    tau_min, tau_max:
        Time interval to define baseline responses, relative to trial onset.
    kwargs:
        Keyword arguments for `get_trial_responses`.

    Returns
    r_base, r_unit: (num_channels,)
        Averaged baseline response and the across-trial standard deviation for
    all channels.

    """
    trial_idxs, _ = get_valid_idxs(session)
    triggers = set()
    for trial_idx in trial_idxs:
        triggers.add((trial_idx, 0))
    _, responses_mean, responses_std = get_triggered_average(
        session, triggers=triggers, tau_min=tau_min, tau_max=tau_max, **kwargs,
    )
    r_base = responses_mean.mean(axis=1)
    r_unit = np.clip(responses_std.mean(axis=1), 1, None)
    return r_base, r_unit
