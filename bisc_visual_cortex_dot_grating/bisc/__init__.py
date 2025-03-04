from pathlib import Path
import datajoint as dj
import yaml
from jarvis import Config

with open(Path(__file__).parent/'VERSION.txt', 'r') as f:
    __version__ = f.readline().split('"')[1]
try:
    with open(Path(__file__).parent/'dj_credential.yaml', 'r') as f:
        dj.config.update(yaml.safe_load(f))
except:
    pass
with open(Path(__file__).parent/'rcParams.yaml', 'r') as f:
    rcParams = Config(yaml.safe_load(f))
