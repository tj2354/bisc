import argparse, yaml, random
from pathlib import Path

from bisc import rcParams
from bisc.cache import Cache
from bisc.data import get_session, get_valid_idxs
from bisc.dot import fit_channel_rf

CACHE_PATH = Path(__file__).parent.parent/rcParams.cache_path

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--session-ids', nargs='+', help="path of yaml file of sessions",
)
parser.add_argument(
    '-f', '--freq-spec', default='cache/freqs.yaml', help="path of yaml file of wavelet frequencies",
)
parser.add_argument(
    '--dt', default=0.5, type=float, help="time resolution of saved response (ms)",
)
parser.add_argument(
    '--num-optims', default=200, type=int, help="number of optimization tries per fit",
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
    with open(get_path(args.freq_spec), 'r') as f:
        freqs = yaml.safe_load(f)

    random.shuffle(session_ids)
    for session_id in session_ids:
        print(f'Fitting RFs for session {session_id}...')
        cache = Cache(CACHE_PATH/session_id)
        session = get_session(session_id)
        _, channel_idxs = get_valid_idxs(session)

        choices = {
            'session': [session], 'channel_idx': channel_idxs,
            'type': ['Gauss'], 'num_optims': [args.num_optims], 'dt': [args.dt],
            'transform': [{'type': 'morlet', 'freq': freq} for freq in freqs],
        }
        cache.sweep(fit_channel_rf, choices, tqdm_kwargs={'desc': 'Fit Gauss', 'leave': True})

        choices = {
            'session': [session], 'channel_idx': channel_idxs,
            'type': ['Gabor'], 'num_optims': [args.num_optims], 'dt': [args.dt],
            'transform': [{'type': 'remove_mean'}],
        }
        cache.sweep(fit_channel_rf, choices, tqdm_kwargs={'desc': 'Fit Gabor', 'leave': True})
