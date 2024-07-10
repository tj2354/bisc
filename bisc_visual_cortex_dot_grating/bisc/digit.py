from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.io import loadmat
from typing import Optional

from . import rcParams
from .alias import Array
from .data import get_num_channels, get_valid_idxs, get_stim, get_baseline, get_trial_responses

STIMULUS_PATH = Path(__file__).parent.parent/rcParams.stimulus_path

def get_mnist_images() -> tuple[Array, Array]:
    r"""Returns full MNIST data.

    Returns
    -------
    labels: (70000,)
        Labels of all MNIST data.
    imgs: (70000, 28, 28, 1)
        MNIST images, normalized to [0, 1].

    """
    saved = loadmat(STIMULUS_PATH/'MNISTImages1.mat')
    labels, imgs = [], []
    for i in range(70000):
        labels.append(saved['exp_images'][0, 0][4][i, 0][1][0, 0])
        imgs.append(saved['exp_images'][0, 0][4][i, 0][4])
    labels = np.array(labels).astype(int)-1
    imgs = np.stack(imgs)[..., 0][..., None].astype(float)/255.
    return labels, imgs


def prepare_dsets(
    session: dict, latency: float = 180.,
    seed: Optional[int] = None, split: float = 0.95,
    W: Array = None,
    **kwargs,
):
    trial_idxs, channel_idxs = get_valid_idxs(session)
    onsets, offsets, stim_params = get_stim(session)
    flags = stim_params['flags']
    rng = np.random.default_rng(seed)
    idxs = {'train': [], 'val': [], 'test': []}
    for trial_idx in trial_idxs:
        if flags[trial_idx]:
            if rng.random()<split:
                idxs['train'].append(trial_idx)
            else:
                idxs['val'].append(trial_idx)
        else:
            idxs['test'].append(trial_idx)
    for tag in idxs:
        idxs[tag] = np.array(idxs[tag])

    img_ids = stim_params['img_ids']
    num_imgs = img_ids.shape[1]
    img_duration = (offsets-onsets)[trial_idxs].mean()/num_imgs
    img_duration = round(img_duration/5)*5
    labels, _ = get_mnist_images()

    r_base, r_unit = get_baseline(session, **kwargs)
    num_channels = get_num_channels(session)
    if W is None:
        W = np.eye(num_channels)
    else:
        assert W.shape[1]==num_channels
    invalid_mask = np.full((num_channels,), fill_value=True).astype(bool)
    invalid_mask[channel_idxs] = False
    dsets = {}
    for tag in idxs:
        inputs, targets = [], []
        for trial_idx in idxs[tag]:
            taus, responses = get_trial_responses(session, trial_idx=trial_idx, **kwargs)
            # responses = (responses-r_base[:, None])/r_unit[:, None]
            responses[invalid_mask] = 0.
            for img_idx in range(num_imgs):
                offset = img_idx*img_duration
                inputs.append(np.matmul(W, responses[:, (taus>=offset)&(taus<offset+latency)]))
                targets.append(labels[img_ids[trial_idx, img_idx]])
        dsets[tag] = TensorDataset(
            torch.tensor(np.array(inputs), dtype=torch.float),
            torch.tensor(np.array(targets), dtype=torch.long),
        )
    return dsets
