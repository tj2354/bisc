### Re-define PATHs according to your file system
### BEGINNING OF PATH DEFINITIONS
# processed data directories

DATA_DIR            = '../data'

BAND_MU_SIGMA_DIR   = f'{DATA_DIR}/channel_band_mu_sigma'
MODEL_INPUT_DIR     = f'{DATA_DIR}/model_input'
MODEL_OUTPUT_DIR     = f'{DATA_DIR}/model_output'

MOTION_DIR          = f'{MODEL_INPUT_DIR}/motion'
SPECT_DATA_DIR      = f'{DATA_DIR}/spectrogram'
CH_DATA_DIR         = f'{DATA_DIR}/channel_example'
### END OF PATH DEFINITIONS

GOOD_SESSIONS = [6, 7, 9, 10, 11, 12, 13]
# session 3 is too short. some destructive event happened during (or before) session 17, affecting all later sessions
BAD_SESSIONS  = [3, 17, 19, 20] 
ALL_SESSIONS  = [3, 6, 7, 9, 10, 11, 12, 13, 17, 19, 20] 
SESSION_KEYS  = ['006A', '006B', '006C', '007A', '007B', '009', '010', '011', '012', '013']

# global variables

# BISC ADC dynamic range
FS_ADC = 3 # (V)
MAX_ADC_CODE = 2**10-1 # (bits)

# define saturation range
LOW_CUT, HIGH_CUT = 2, MAX_ADC_CODE-2

# recording specs
NCH = 256               # number of recordring channels
FS, TS = 33900, 1/33900 # (S/s) sampling rate per channel
USE_TETRODE = False
VGA_GAIN = 0            # (bits) back-end vga gain configuration
GAIN = 82*4*1.5**(VGA_GAIN + 1) # (V/V) nominal system gain

# 1. Impute and Truncate
# criteria: for extracting recording segment that corresponds to good behavior
T_SETTLE = 1.5    # (s). initial settling time (for BISC recording channels to stabilize after exponential decay..)
T_BPF_PAD = 3     # (s). padding to deal with BPF artifacts
T_SCALO_PAD = 1.2 # (s). padding to account for scalogram creation (1 + 0.2 sec)
# criteria: good channels have non-saturated data points greater than GOODCH_CUT
GOODCH_CUT = 0.99

# 2. BPF and Downsample
BPF_LOW = 0.3  # (Hz). Stavisky 2015 used 0.3Hz
BPF_HIGH = 300 # (Hz)
LPF_FREQ = 300 # (Hz)
N_BPF = 4 # filter order

RS = 16 # 16: 2.12 kS/s
RS1, RS2 = 4, 4
assert RS == RS1*RS2

# 3. Hemodynamics (HD) Removal. HB: heartbeat
# band-pass filter params
HB_BPF_LOW = 3   # (Hz). 1st harmonic
HB_BPF_HIGH = 12 # (Hz). cut-off greater than 3rd harmonic
HB_BPF_N = 4 # filter order
# SVD params
T_PC = 10 # (s). recording will be partitioned to T_PC long segments. Last trailing segment will be rounded up
N_PC = 15 # number of PCs to calculate (rank of SVD)
N_PC_REMOVE = 5 # number of PCs to remove
N_PC_PLOT = 10 # number of PCs to plot
# PC multi-taper half-bandwidth
W_MT_PC = 0.5 # (Hz). frequency smoothing: [f0 - W , f0 + W]
HB_FREQ_LOW, HB_FREQ_HIGH = 3, 3.7 # (Hz) # dominant tone of hemodynamic

# 5. Scalogram and LMP extraction
T_SCALO = 200e-3     # (s). scalogram window size
T_STEP_SCALO = 5e-3  # (s). window sliding step size
W_MT_SCALO = 10      # (Hz). multi-taper frequency smoothing: [f0 - W , f0 + W]
T_LMP = 50e-3        # (s). LMP window size (boxcar sampling)
# 4/12/2024: Clipping is no longer used. deprecated
# LMP_VCLIP = 300e-6   # channel waveform will be clipped at +/-LMP_VCLIP for LMP calculation
# band definitions (partitioned!) 
BETA_FREQ0, BETA_FREQ1 = 10, 30 # (Hz)
LGA_FREQ0, LGA_FREQ1 = 30, 70   # (Hz)
HGA_FREQ0, HGA_FREQ1 = 70, 190  # (Hz)
SCALO_MAX_FREQ = 200 # (Hz)

# 6. Model input matrix params. The final model input matrix is further truncated and downsampled (in time domain)
# time lag parameters
FULL_TAU_START = -1.0
FULL_TAU_END = 1.0

# "band data" have been decimated from the 2.11kS/s data by a factor of 2
T_DF_MATRIX = 2 # decimation factor that has been applied to time-frequency recording data (LMP, beta, lga, hga)

# 7. Final model decoder input params
TAU_DF = 5 # decimation factor for final model input time resolution
TAU_START = -0.5
TAU_END = 0.5
T_DF_MOTION = 20 # decimation factor that will be applied to motor feature (for decoder)

# model parameters
OPT_HPARAM = 5 # optimal number of PLS component
N_SPLIT = 5 # kfold CV
DIMENSION = 'y'

X_OPT_HPARAM = 1
Y_OPT_HPARAM = 5 # 1e4 if using Ridge
Z_OPT_HPARAM = 5