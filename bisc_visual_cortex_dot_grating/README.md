# Data analysis for BISC recordings

Spatial-temporal receptive fields of each BISC channel

![](cache/48834689_raw_data.gif)

## Requirements

`datajoint`, `h5py`, [`jarvis>=0.7`](https://github.com/lizhe07/jarvis/releases/tag/v0.7.0)

## Installation

It is recommended to install the `bisc` package in editable mode by running the script at current
folder.
```bash
pip install -e .
```

## Data preparation
By default, the raw stimulus data are stored in `'stimulus'` folder, the raw response data are
stored in `'response'` folder and all intermediate results are stored in `'cache'` folder, all
relative to the repository root folder. They can either be changed in `bisc/rcParams.yaml` or at the
first time `bisc` module is imported like below
```python
from bisc import rcParams

rcParams['stimulus_path'] = '/mnt/d/BISC_2023/stimulus'
rcParams['response_path'] = '/mnt/d/BISC_2023/response'
```

`'stimulus'` folder contains subfolders named as `'YYYY-MM-DD_HH-MM-SS'`, each corresponding to one
experiment session and should have a `.mat` file with `'Synched'` at the end of file name.

`'response'` folder contains subfolders named as `'YYYY-MM-DD_HH-MM-SS'` for each session as well,
but usually the exact seconds are different from the stimulus folder of the same session. Each
subfolder contains one subfolder of a similar name, inside of which are a series of `.h5` files
named as `'BiscElectrophysiology%d.h5'`.

Details of folder names of all sessions are listed in `scripts/sessions.yaml`. The last 8 digits of
`session_start_time` of each session is defined as the session ID, and are used to refer the session
in most of the codes.

`'cache'` folder contains subfolders named by session IDs. Each subfolder contains the intermediate
results of various analysis functions and is managed by `bisc.cache.Cache` class. The codes will
automatically creates a new folder of each session if it does not exist already.

## Figures in manuscript

Notebooks used to generate some figures in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.05.17.594333v1)
are saved in `'notebooks'` folder.

Figure 5B is created with `plot-grating-stimuli.ipynb`. Figure 5CE, 6B and S11 are created with
`filtered-response.ipynb`. Figure 5F-K and S13 are created with `grating-session.ipynb`. Figure 5L-N
and S14 are created with `orientation-classification.ipynb`. Figure S15 is created with
`orientation-regression.ipynb`.

Figure 6C-E are created with `receptive-field-analysis.ipynb`. Figure 6F is created with
`cortical-scaling.ipynb`. Figure 6G is created with `plot-dense-rfs-filtered.ipynb` using
data in `cache/RF.fits_dense.zip`.

Figure S16 is created with `mixture-fit.ipynb` and `plot-dense-rfs-raw.ipynb`.

Some notebooks need to have raw stimulus and response data to run, which is too large to be shared
directly. We have saved those data on Globus storage and will share access when requested.
- `filtered-response.ipynb`
- `orientation-classification.ipynb`
- `orientation-regression.ipynb`
- `mixture-fit.ipynb`
- `plot-dense-rfs-raw.ipynb`
