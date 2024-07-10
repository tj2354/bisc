import os, argparse, yaml
from datetime import datetime, timedelta
from pathlib import Path

from jarvis.utils import tqdm
from bisc.data import (
    get_acq, fetch_spath_str, fetch_epash_str,
    fetch_offset_and_slope, fetch_sessions,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--start', default='2023-10-02', help="start date of range",
)
parser.add_argument(
    '-e', '--end', default='2023-12-02', help="end date of range",
)
parser.add_argument(
    '-v', '--valid', default=100, type=int, help="required number of valid trials",
)
parser.add_argument(
    '-o', '--overwrite', action='store_true', help="overwrite existing yaml file",
)
args = parser.parse_args()

format = '%Y-%m-%d'
args.start = datetime.strptime(args.start, format)
args.end = datetime.today() if args.end is None else datetime.strptime(args.end, format)
meta_path = Path(__file__).parent/'sessions.yaml'


if __name__=='__main__':
    print('Saving sessions from {} to {} with at least {} valid trials to\n{}'.format(
        datetime.strftime(args.start, format), datetime.strftime(args.end, format),
        args.valid, meta_path,
    ))

    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path, 'r') as f:
            metas = yaml.safe_load(f)
        sessions = [m['session'] for m in metas]
    else:
        sessions = []

    for d in tqdm(range((args.end-args.start).days+1), unit='day'):
        sessions += fetch_sessions(
            datetime.strftime(args.start+timedelta(days=d), format), min_trials=args.valid,
        )
    sessions = set(frozenset(s.items()) for s in sessions)
    sessions = [{k: v for k, v in s} for s in list(sessions)]
    sessions = sorted(sessions, key=lambda s: s['session_start_time'])

    acq = get_acq()
    tag = None
    with open(meta_path, 'w') as f:
        for session in tqdm(sessions, unit='session'):
            try:
                offset, slope = fetch_offset_and_slope(session)
            except:
                continue
            date = datetime.strftime((acq.Sessions & session).fetch1('session_datetime'), format)
            if date!=tag:
                tag = date
                f.write(f'# {tag}\n')
            f.write('- session:\n')
            for key in [
                'subject_id', 'setup', 'session_start_time',
                'ephys_start_time', 'stim_start_time',
            ]:
                f.write('    {}: {}\n'.format(key, session[key]))
            spath = fetch_spath_str(session)
            f.write(f'  spath: {spath}\n')
            epath = fetch_epash_str(session)
            f.write(f'  epath: {epath}\n')
            f.write(f'  offset: {offset}\n')
            f.write(f'  slope-1: {slope-1}\n')
