"""
Collection of functions for processing motion data
"""
import numpy as np
import h5py

##########
RIG_SYNC_HIGH = 5 # (V). Logic high of the psuedorandom sync pulse

# interpolation params
T_SHORT = 0.5 # (s) linearly interpolate gaps smaller than T_SHORT
T_LONG = 2.0 # (s) for gaps smaller than T_LONG, use cubic. otherwise, use linear interpolation
T_WIN = 200e-3 # (s) size of moving average window
T_SPLINE = 1.0 # (s) length (from each side of missing segment) to use for applying CubicSpline

# MOTION_IDX_DICT: these indices define the time segments where the animal was performing 
# "good behavior". they have been manually identified by looking at the video.
# they have been modified (by TJ) from v3p0, which was provided by KEW
MOTION_IDX_DICT = { 
    # my index (python) , KEW's index (MATLAB)
    "003A": (2406, 4875), # (2407, 4875)
    "003B": (5174, 6741), # (5175, 6741)
    "006A": (1322, 1865), # (1323, 1865)
    "006B": (6616, 7129), # (6617, 7129)
    "006C": (7242, 7761), # (7243, 7761)
    "007A": (2273, 4474 - 30), # (2274, 4474)
    "007B": (4673, 6824), # (4674, 6824)
    "009" : (7079, -1),   # (7080, 9421)
    "010" : (0, -1),      # Entire session
    "011" : (3797 + 700, -1),   # (3798, 8562)
    "012" : (5249 + 150, -1),   # (5250, end)
    "013" : (2291, -1),   # (2292: 9172)
    "017A": (1499 + 30 , 3900), # (1500, 3900)
    "017B": (5299, 7800 + 30), # (5300, 7800)
    "019" : (1999, -1),   # (2000, end)
    "020" : (0, 3011)     # (1, 3011)
}

MOTION_IDX_DICT_V3P0 = { 
    # my index (python) , KEW's index (MATLAB)
    "003A": (2406, 4875), # (2407, 4875)
    "003B": (5174, 6741), # (5175, 6741)
    "006A": (1322, 1865), # (1323, 1865)
    "006B": (6616, 7129), # (6617, 7129)
    "006C": (7242, 7761), # (7243, 7761)
    "007A": (2273, 4474), # (2274, 4474)
    "007B": (4673, 6824), # (4674, 6824)
    "009" : (7079, -1),   # (7080, 9421)
    "010" : (0, -1),      # Entire session
    "011" : (3797, -1),   # (3798, 8562)
    "012" : (5249, -1),   # (5250, end)
    "013" : (2291, -1),   # (2292: 9172)
    "017A": (1499, 3900), # (1500, 3900)
    "017B": (5299, 7800), # (5300, 7800)
    "019" : (1999, -1),   # (2000, end)
    "020" : (0, 3011)     # (1, 3011)
}

def read_sync_h5(load_dir, h5name):
    """
    Parse h5 file that contains timestamp info of the psuedorandom pulses used for synchronization

    sample_interval: ns
    sample_rate: Hz
    received: Nx2 array. column 0: timestamp in ns, column 1: cumulative number of data points collected
    data: raw sample data (sync low: 0V, high: 5V)

    for example,
    received[0,:] = (1032310632898270, 16): at 1032310632898270 ns data[0:16] was collected
    received[1,:] = (1032310649862101, 33): at 1032310649862101 ns data[16:33] was collected, etc.

    And the timestamp refers to the time of the last sample, so data[15] corresponds to 1032310632898270, data[14] is 1032310632898270 - "Sample Interval", data[13] is 1032310632898270 - 2*"Sample Interval", etc.
    """
    try:
        with h5py.File(f'{load_dir}/{h5name}.h5', 'r') as file:
            group = file['/analog/Node 3']
            attributes = group.attrs
            sample_interval = attributes['Sample Interval'][0]
            sample_rate = attributes['Sample Rate'][0]
            received = file['/analog/Node 3/received'][:]
            data = file['/analog/Node 3/data'][:]
    except:
        print('could not read file')
        return None, None, None, None
    return sample_interval, sample_rate, received, data

def correlate_custom(a, v, i0=0, i1=np.inf):
    """
    a: First 1-D input array.
    v: Second 1-D input array
    i0, i1: indices that define range of correlation to compute

    out: 1-D array containing cross-correlation of 'a' and 'v'.
    """
    M = len(a)
    N = len(v)
    out_size = M + N - 1

    out = np.zeros(out_size, dtype=np.float64)

    for k in range(out_size):
        if k < i0 or k > i1:
            continue
        start_idx_a = max(0, k - N + 1)  # Start index for 'a'
        end_idx_a = min(M, k + 1)        # End index for 'a'
        start_idx_v = max(0, N - k - 1)  # Start index for 'v'
        end_idx_v = min(N, M + N - k - 1)  # End index for 'v'

        out[k] = np.sum(a[start_idx_a:end_idx_a] * v[start_idx_v:end_idx_v])

    return out

##########
def interpolate_nans(arr, N):
    """
    Given an array arr, interpolate up to N continuous NaN elements
    """
    nan_idxs = np.where(np.isnan(arr))[0]  # Get indices of NaN elements
    new_arr = np.copy(arr)
    
    for idx in nan_idxs:
        start = max(0, idx - N)  # Define the start index for checking non-NaN values
        end = min(len(arr), idx + N + 1)  # Define the end index for checking non-NaN values
        
        # Check if there are non-NaN values in the range
        if np.any(~np.isnan(arr[start:idx])) and np.any(~np.isnan(arr[idx+1:end])):

            nn_idx0 = np.where(~np.isnan(arr[start:idx]))[0][-1] + start
            nn_idx1 = np.where(~np.isnan(arr[idx+1:end]))[0][0] + (idx + 1)

            if nn_idx1 - nn_idx0 > N + 1:
                continue
            
            v0 = arr[nn_idx0]
            v1 = arr[nn_idx1]

            interp_val = v0 + (v1 - v0)/(nn_idx1 - nn_idx0)*(idx - nn_idx0)

            # print(nn_idx0, idx, nn_idx1)
            # print(arr[nn_idx0], interp_val, arr[nn_idx1])
            new_arr[idx] = interp_val
        # break
    
    return new_arr

def moving_average(pos, window_size, nan_tol=0, target_idxs=None):
    """
    pos: time series
    window_size: moving average window = 2*window_size
    nan_tol: tolerance on # of invalid data points
    target_idxs: apply windowing only on this target
    
    for target_idxs in pos, look at left hand, and right hand side segments
    apply smoothing only if LHS and RHS have invalid points leq nan_tol
    """

    smoothed_pos = np.copy(pos)
    if target_idxs is None:
        target_idxs = np.arange(0, len(pos))
    
    for idx in target_idxs:
        start = max(0, idx - window_size + 1)
        end = min(len(pos), idx + window_size)
        
        full_seg = pos[start:end]
        lhs_seg  = pos[start:idx]
        rhs_seg  = pos[idx+1:end]

        lhs_nan_idxs = np.isnan(lhs_seg)
        rhs_nan_idxs = np.isnan(rhs_seg)

        lhs_nan_count = np.count_nonzero(lhs_nan_idxs)
        rhs_nan_count = np.count_nonzero(rhs_nan_idxs)

        if lhs_nan_count > nan_tol and rhs_nan_count > nan_tol:
            continue
        if lhs_nan_count > nan_tol and len(rhs_seg) == 0:
            continue
        if rhs_nan_count > nan_tol and len(lhs_seg) == 0:
            continue
        
        smoothed_pos[idx] = np.nanmean(full_seg)
    
    return smoothed_pos

def get_nanseg_idxs(pos):
    """
    pos: time series data
    identify segments where all elements are np.nan, and return the indices
    corresponding to start and end of those segments
    """
    nan_idxs = np.where(np.isnan(pos))[0]
    if len(nan_idxs) == 0:
        return np.array([]), np.array([])

    nanseg_start_idxs = [] # segment start
    nanseg_end_idxs = [] # segment end

    nanseg_start_idxs.append(nan_idxs[0])

    for idx in range(1, len(nan_idxs)-1):
        if nan_idxs[idx] - nan_idxs[idx-1] > 1: # start
            nanseg_start_idxs.append(nan_idxs[idx])
        if nan_idxs[idx+1] - nan_idxs[idx] > 1: # end
            nanseg_end_idxs.append(nan_idxs[idx])

    nanseg_end_idxs.append(nan_idxs[-1])

    nanseg_start_idxs = np.array(nanseg_start_idxs)
    nanseg_end_idxs = np.array(nanseg_end_idxs)

    assert len(nanseg_start_idxs) == len(nanseg_end_idxs)
    assert np.alltrue(nanseg_end_idxs - nanseg_start_idxs)

    return nanseg_start_idxs, nanseg_end_idxs

from scipy.interpolate import interp1d, CubicSpline

def spline_nans(t, pos, nanseg_start_idxs, nanseg_end_idxs, T_spline, T_limit):
    """
    t: time
    pos: time series
    nanseg_start_idxs, nsegseg_end_idxs: start and end indices of np.nan segments
    T_spline: length of data (on both side of nanseg) to use for applying CubicSpline
    T_limit: if the np.nan segment is greater than T_limit, apply linear interpolation
    if less than equal to T_limit, apply Cubic interpolation
    """
    if len(nanseg_start_idxs) == 0 and len(nanseg_end_idxs) == 0:
        return pos

    splined_pos = np.copy(pos)

    for idx0, idx1 in zip(nanseg_start_idxs, nanseg_end_idxs):

        assert np.isnan(pos[idx0]) and np.isnan(pos[idx1])

        if idx0 == 0: # nanseg is left extrema
            vfill = pos[idx1+1] # backward fill
            splined_pos[idx0:idx1+1] = np.full_like(splined_pos[idx0:idx1+1], vfill)
            continue

        if (idx1 + 1) == len(t): # nanseg is right extrema
            vfill = pos[idx0-1] # forward fill
            splined_pos[idx0:idx1+1] = np.full_like(splined_pos[idx0:idx1+1], vfill)
            continue

        T_nanseg = t[idx1] - t[idx0]
        assert ~np.isnan(pos[idx0-1]) and ~np.isnan(pos[idx1+1])
        if T_nanseg > T_limit: # linear interpolation
            x0, x1 = t[idx0-1], t[idx1 + 1]
            y0, y1 = pos[idx0-1], pos[idx1 + 1]

            f_interp = interp1d([x0, x1], [y0, y1])
            splined_pos[idx0:idx1+ 1] = f_interp(t[idx0:idx1+1])
        else: # cubic interpolation
            lhs_t0 = max(t[0], t[idx0] - T_spline)
            rhs_t1 = min(t[-1], t[idx1] + T_spline)
            # print(lhs_t0, t[idx0-1])
            # print(t[idx1+1], rhs_t1)

            lhs_idx0, lhs_idx1 = np.where(t >= lhs_t0)[0][0], idx0-1
            rhs_idx0, rhs_idx1 = idx1 + 1, np.where(t <= rhs_t1)[0][-1]
            # print(lhs_idx0, lhs_idx1)
            # print(rhs_idx0, rhs_idx1)

            lhs_t   =   t[lhs_idx0:lhs_idx1+1]
            rhs_t   =   t[rhs_idx0:rhs_idx1+1]
            lhs_pos = pos[lhs_idx0:lhs_idx1+1]
            rhs_pos = pos[rhs_idx0:rhs_idx1+1]

            # print(lhs_t[0], lhs_t[-1])
            # print(rhs_t[0], rhs_t[-1])

            lhs_t   =   lhs_t[~np.isnan(lhs_pos)]
            rhs_t   =   rhs_t[~np.isnan(rhs_pos)]
            lhs_pos = lhs_pos[~np.isnan(lhs_pos)]
            rhs_pos = rhs_pos[~np.isnan(rhs_pos)]

            # print(lhs_t)
            # print(rhs_t)
            # print(np.concatenate(lhs_t, rhs_t).shape)
            f_interp = CubicSpline(np.concatenate((lhs_t, rhs_t)),
                                   np.concatenate((lhs_pos, rhs_pos)))
            splined_pos[idx0:idx1+ 1] = f_interp(t[idx0:idx1+1])

    # do not spline exceeding the raw data range
    vmax = np.nanmax(pos)
    vmin = np.nanmin(pos)
    splined_pos[splined_pos > vmax] = vmax
    splined_pos[splined_pos < vmin] = vmin

    return splined_pos