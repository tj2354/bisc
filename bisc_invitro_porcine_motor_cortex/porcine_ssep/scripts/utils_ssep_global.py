# raw data
RAW_DATA_DIR = './recording_raw'
LOG_FNAME = 'record_log_20231017_cleaned.csv' # log. contains peripheral stim params & chip config
STIM_RATE = 2.79 # (Hz) peripheral stimulation rate
STIM_PULSE_WIDTH = 0.3e-3 # (s) peripheral stimulatino pulse width
T_SETTLE = 1 # (s) recording channel settling time. typically used for truncating data [0:T_SETTLE]

DEAD_THRESH = 2.5
NOISY_THRESH = 3

# Delay between "peripheral stim" and "sychronization pulse" from stimulator and BISC relay station
T_SYNC_DELAY = 55e-3 # (s)

# processed data
# DATA_DIR = './recording_preprocessed'
DATA_DIR = '../data'

# BISC ADC dynamic range
FS_ADC = 3 # (V)
MAX_ADC_CODE = 2**10-1 # (bits)

# define saturation range
LOW_CUT, HIGH_CUT = 2, MAX_ADC_CODE-2

# recording specs
NCH = 256               # number of recordring channels
FS, TS = 33900, 1/33900 # (S/s) sampling rate per channel
VGA_GAIN = 0            # (bits) back-end vga gain configuration
GAIN = 82*4*1.5**(VGA_GAIN + 1) # (V/V) nominal system gain

# stimulation sites
N_SITES = 5
STIM_LABELS = ['Median', 'Snout Lateral', 'Snout Superior', 'Snout Medial', 'Snout Inferior']

# criteria for determining "unstable" channels. used for CAR channel selection
STABLE_PROP_THRESH = 1.0
# number of channels to use for CAR
N_CAR_CH = 10

# Full window
FULL_T0, FULL_T1 = -250e-3, 250e-3
# SSEP window
SEP_T0, SEP_T1 = 5e-3, 45e-3
# Baseline window
BASELINE_T0, BASELINE_T1 = -50e-3, -5e-3
# Artifact window (remove sync artifact)
ARTIFACT_T0, ARTIFACT_T1 = 49e-3, 101e-3
N_PC = 30 # number of artifact PCs (i.e. number of singular values) to compute (ok as long as N_PC >= N_REMOVE)
N_REMOVE = 15 # number of PCs to remove
N_STITCH = 5 # number of data points to use to "stitch" the segment after artifact removal
# Downsampled window
DS_T0, DS_T1 = -101e-3, 101e-3


# Downsampling factor
RS1, RS2 = 4, 4 # scipy.decimate will be called twice, nested
RS = 16
assert RS1*RS2 == RS

# Multi-taper Params
MT_LEN_WIN = 100e-3 # (s) window length
MT_LEN_WINSTEP = 5e-3 # (s) window sliding step
MT_W = 15 # (Hz) half-bandwidth
# MT_W = 20 gives 3 tapers, but spectrogram appears smudged

