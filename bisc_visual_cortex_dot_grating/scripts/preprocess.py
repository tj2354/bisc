import os, shutil, argparse, yaml, random
from pathlib import Path

from bisc.data import (
    get_session, get_num_trials, get_trial_responses,
    get_valid_idxs, get_baseline,
)
from bisc.dot import dot_triggered_average
from bisc.grating import grating_triggered_average
from bisc import rcParams
from bisc.cache import Cache

CACHE_PATH = Path(__file__).parent.parent/rcParams.cache_path

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--session-ids', nargs='+', help="path of yaml file of sessions",
)
parser.add_argument(
    '-t', '--transforms', help="path of yaml file of transformations",
)
parser.add_argument(
    '--dt', default=0.5, type=float, help="time resolution of saved response (ms)",
)
parser.add_argument(
    '--pre-trial', default=800., type=float, help="time before trial onset (ms)",
)
parser.add_argument(
    '--post-trial', default=400., type=float, help="time after trial offset (ms)",
)
parser.add_argument(
    '--dot-triggered-average', action='store_true', help="whether to compute dot-triggered average",
)
parser.add_argument(
    '--grating-triggered-average', action='store_true', help="whether to compute grating-triggered average",
)
args = parser.parse_args()

def get_path(filename):
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent.parent/path
    return path


if __name__=='__main__':
    if args.session_ids is None:
        args.session_ids = ['cache/sessions.yaml']
    session_ids = args.session_ids
    if len(session_ids)==1 and session_ids[0].endswith('.yaml'):
        with open(get_path(session_ids[0]), 'r') as f:
            session_ids = yaml.safe_load(f)
    for session_id in session_ids:
        assert len(session_id)==8 and session_id.isdigit()

    if args.transforms is None:
        transforms = [{'type': 'remove_mean'}]+[
            {'type': 'morlet', 'freq': float(freq)} for freq in [4, 8, 16, 32, 64, 128]
        ]
    else:
        with open(get_path(args.transforms), 'r') as f:
            transforms = yaml.safe_load(f)

    random.shuffle(session_ids)
    for session_id in session_ids:
        print(f'Preprocessing {session_id}...')
        cache = Cache(CACHE_PATH/session_id)
        session = get_session(session_id)

        num_trials = get_num_trials(session)
        choices = {
            'session': [session], 'dt': [args.dt],
            'trial_idx': list(range(num_trials)), 'transform': transforms,
            'pre_trial': [args.pre_trial], 'post_trial': [args.post_trial],
        }
        cache.sweep(get_trial_responses, choices, tqdm_kwargs={'desc': 'Preprocess', 'leave': True})

        trial_idxs, channel_idxs = get_valid_idxs(session)
        print('{} valid trials, {} valid channels'.format(len(trial_idxs), len(channel_idxs)))
        choices = {
            'session': [session], 'dt': [args.dt], 'transform': transforms,
        }
        cache.sweep(get_baseline, choices, tqdm_kwargs={'desc': 'Baseline', 'unit': 'trans', 'leave': True})

        if args.dot_triggered_average:
            cache.sweep(dot_triggered_average, choices, tqdm_kwargs={'desc': 'DTA', 'unit': 'trans', 'leave': True})

        if args.grating_triggered_average:
            cache.sweep(grating_triggered_average, choices, tqdm_kwargs={'desc': 'GTA', 'unit': 'trans', 'leave': True})
