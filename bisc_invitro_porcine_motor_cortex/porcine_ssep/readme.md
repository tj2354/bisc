## Running the Scripts

### Required Modifications

Before running the scripts, please modify the following variables in `utils_ssep_global.py` to point to the path to your data directory:
`DATA_DIR`

To generate movie with script `5p4_16x16_ani.ipynb`, please download FFMPegWriter from this [link](https://ffmpeg.org/download.html).
One way to use the downloaded file is to copy `ffmpeg.exe` to the `./scripts` directory

### Optional Modifications
Modify the following variables in `5p1_waterfall_plot.ipynb` and `5p2_16x16_temporal_plot.ipynb` to generate plots corresopnding to the specific location of peripheral stimulation:
```
# 0: Median Nerve
# 1: Snout Lateral
# 2: Snout Superior
# 3: Snout Medial
# 4: Snout Inferior
stim_site = 3 # pick from 0, 1, 2, 3, 4. 
```

### Running the Scripts
Scripts with extension `.ipynb` should be executed in the order of their file names (numbered from 5p1 to 9) to reproduce the plots and movies relevant to the BISC porcine SSEP recordings

Expected run time for each script is less than a minute

Please note that t-SNE result can be stochastic

### File Descriptions

#### Each script loads SSEP recording and does the following:
- `5p1_waterfall_plot`: plot SSEPs on a shared time axis (Fig. 3C)
- `5p2_16x16_tempoeral_plot`: plot SSEPs by their channel location (Fig. 3B)
- `5p3_16x16_spatial_plot`: generate spatial map of normalized SSEPs (Fig. 3D)
- `5p4_16x16_movie_plot`: generate movie of the SSEPs (Video S1)
- `7_zscore`: perform z-scoring on SSEPs
- `9_tsne_and_lda`: perform t-SNE clustering and build LDA classifier (Fig. 3E and F)

#### Scripts in `./scripts/preprocessing` are used for data pre-processing and are not meant to be executed

#### Scripts in `./scripts/read_nwb` are independent from rest of the analysis. It is a bonus script that provides an example of how to parse and plot raw BISC recordings in the provided NWB format (`/porcine_ssep/sample_nwb_data`). Instruction on how to run this independent script is provided within itself: `bisc_sample_nwb_recording_plot.ipynb` 