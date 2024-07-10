import os, shutil, argparse
from datetime import datetime
from pathlib import Path

from jarvis.utils import tqdm
from bisc import acq
from bisc.data import fetch_sessions, fetch_epash_str

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--date', default='2023-10-02', help="date of sessions to copy",
)
parser.add_argument(
    '-r', '--response-dir', default='/mnt/v/Paul', type=Path, help="response directory",
)
parser.add_argument(
    '-s', '--stimulus-dir', default='/mnt/y/stimulation/Paul', type=Path, help="stimulus directory",
)
parser.add_argument(
    '-t', '--target-dir', default='/mnt/d/BISC_2023', type=Path, help="target directory",
)
args = parser.parse_args()

format = '%Y-%m-%d'
args.date = datetime.strptime(args.date, format)


if __name__=='__main__':
    # copy stimulus files
    sessions = fetch_sessions(datetime.strftime(args.date, format))
    for session in sessions:
        folder_name = args.stimulus_dir/(acq.Stimulation&session).fetch1('stim_path').split('/')[-1]
        exp_type = (acq.Stimulation&session).fetch1('exp_type')
        src = folder_name/f'{exp_type}Synched.mat'
        dst = args.target_dir/'stimulus'/'/'.join(src.parts[-2:])
        os.makedirs(dst.parent, exist_ok=True)
        shutil.copyfile(src, dst)
    # copy response files
    src_files, dst_files = [], []
    for session in sessions:
        src = Path(fetch_epash_str(session))
        src = args.response_dir/'/'.join(src.parts[-3:-1])
        dst = args.target_dir/'response'/'/'.join(src.parts[-2:])
        for f in os.listdir(src):
            if (
                f.startswith('BiscElectrophysiology') and f.endswith('.h5')
                and (not os.path.exists(dst/f) or os.path.getsize(dst/f)<os.path.getsize(src/f))
            ):
                src_files.append(src/f)
                dst_files.append(dst/f)
    for src, dst in tqdm(list(zip(src_files, dst_files)), unit='file'):
        os.makedirs(dst.parent, exist_ok=True)
        shutil.copyfile(src, dst)
