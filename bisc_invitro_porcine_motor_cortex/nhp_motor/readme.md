## Running the Scripts

### Required Modifications
Before running the scripts, please modify the following variables in `utils_motor_global.py` to point to the path to your data directory:
`DATA_DIR`

To generate movie with script `1_spatiotemporal_video.ipynb`, please download FFMPegWriter from this [link](https://ffmpeg.org/download.html).
One way to use the downloaded file is to copy `ffmpeg.exe` to the `./scripts` directory

### Running the Scripts
Scripts with extension `.ipynb` should be executed in the order of their file names (numbered from 1 to 4) to reproduce the plots and movies relevant to the BISC NHP motor cortex recordings

Expected run time for generating movie from `1_spatiotemporal_video.ipynb` is around 5 minutes.
Expected run times for all other scripts are less than a minute

### File Descriptions

#### Each script loads motor data (pre-processed motor feature and ephys recordings) and does the following:
- `1_spatiotemporal_video`: generate video showing frame-by-frame LMP, beta, and high gamma band recordings aligned to the wrist velocity (Video S2)
- `2_opt_model`: finds the optimal PLS model for continuous motor feature decoding, sweeping across the hyperparameter. It saves the result of predicted and observed features from the optimal model and plots the spectral contributions (Fig. 4F)
- `3_decoder_and_band_plot`: generate plots of an example time segments of "predicted vs. observed motor feature" and "z-scored multi-channel LMP and high gamma band recordings" (Fig. 4E,G,H)
- `4_example_channel`: generate plots of an example channel waveforms and spectra, before and after hemodynamic artifact removal

#### Scripts in `./scripts/full_pipeline` are used for data pre-processing and are not meant to be executed