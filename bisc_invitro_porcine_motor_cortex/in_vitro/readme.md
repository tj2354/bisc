## Running the Scripts

### Required Modification
Before running the scripts, please modify the following variables in `utils_invitro_global.py` to point to the path to your data directory:
`ELEC_DATA_DIR`, `REC_DATA_DIR`, `STIM_DATA_DIR`

### Optional Modifications
Modify the following variables in `2_input_noise_psd_plot.ipynb` to load measurement data corresponding to the specific chip configuration:
```
VGA_GAIN       = 0 # choose from 0, 1, 2, 3.  0: lowest gain. 3: highest gain
VBIAS_PR       = 6 # choose from 5, 6, 7. 5: lower cut-off. 7: higher-cut-off
```

Modify the following variables in `3_transfer_function_plot.ipynb` to load measurement data corresponding to the specific chip configuration:
```
""" choose which configs to plot: HPF, 256 vs 1024 """
VBIAS_PR       = 6 # choose from 5, 6, 7. 5: lower cut-off. 7: higher-cut-off
EN_STATIC_ELEC = 0 # choose from 0, 1. 0: 1024, 1: 256
```
```
""" choose which configs to plot: VGA_GAIN, 256 vs 1024 """
VGA_GAIN       = 2 # choose from 0, 1, 2, 3.  0: lowest gain. 3: highest gain
EN_STATIC_ELEC = 1 # choose from 0, 1. 0: 1024, 1: 256
```

### Running the Scripts
Scripts with extension `.ipynb` can be executed in any order to reproduce plots relevant to the BISC in vitro characterizations

Expected run time for each script is less than a minute

### File Descriptions

#### Each script loads and plots data on the following measurements
- `1_impedance_plot`: electrochemical spectroscopy (EIS) of the titanium nitride electrode sample (Fig. 2A)
- `2_input_noise_psd_plot`: input noise power spectra for different recording configurations (Fig. 2E)
- `3_transfer_function_plot`: overall device transfer function for different recording configurations (Fig. 2B,D)
- `4_histogram_plot`: noise and gain distribution of channels over the entire array (Fig. 2C)
- `5_stim_pulse_amplitude_pot`: stimulation current waveforms across different amplitude configurations (Fig. 2G)
- `6_stim_pulse_width_plot`: stimulation current waveforms across different pulse width configurations (Fig. 2H)