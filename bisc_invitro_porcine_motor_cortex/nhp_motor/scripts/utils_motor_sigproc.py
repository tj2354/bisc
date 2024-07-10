import numpy as np

def interpolate_good_channels(rec_data, good_channels):
    """
    Given array rec_data, interpolate all NaN points of good channels
    """
    interp_data = np.copy(rec_data[:, good_channels])

    for interp_ch_data, gch in zip(interp_data.T, good_channels):
        raw_ch_data = rec_data[:, gch]

        # forward/backward fill edges
        if np.isnan(raw_ch_data[0]):
            nn_idx = np.where(~np.isnan(raw_ch_data))[0][0] # index of first non-NaN element
            interp_ch_data[:nn_idx] = raw_ch_data[nn_idx]
        if np.isnan(raw_ch_data[-1]):
            nn_idx = np.where(~np.isnan(raw_ch_data))[0][-1] # index of last non-NaN element
            interp_ch_data[nn_idx+1:] = raw_ch_data[nn_idx]
        
        # compute non-nan indices after filling the edges
        ch_data = np.copy(interp_ch_data)
        nn_idxs = np.where(~np.isnan(ch_data))[0] # non-nan indices

        for idx in np.where(np.isnan(ch_data))[0]:
            ii = np.searchsorted(nn_idxs, idx)

            nn_idx0 = nn_idxs[ii-1] # index of last non-nan data preceding idx
            nn_idx1 = nn_idxs[ii]  # index of first non-nan data following idx

            v0 = ch_data[nn_idx0]
            v1 = ch_data[nn_idx1]

            # linear interpolation
            interp_val = v0 + (v1 - v0)/(nn_idx1 - nn_idx0)*(idx - nn_idx0)
            interp_ch_data[idx] = interp_val
    
    return interp_data

def get_pc(pc_idx, u, s, v):
    """
    u, s, v: matrix components of (truncated) SVD. (left unitary, singular value, right unitary)
    pc_idx: index of singular value to preserve

    returns inverse SVD using just one singular value specified by pc_idx. other values set to 0.

    I think the operation can be described as rank-N approximation where N = pc_idx
    But I'm more inclined to call it as extracting PC that correspond to N'th singular value
    """
    s_padded = np.zeros((u.shape[1], v.shape[0]))
    s_padded[pc_idx, pc_idx] = s[pc_idx]

    return u@s_padded@v

def get_mt_ch_psd(ch_data, dpss_tapers, wt):
    """
    ch_data: (T,) dimension time-domain channel data
    dpss_tapers: discrete prolate spherodal sequences (Slepian multi-taper)
    wt: weight to apply on dpss_tapers
    returns multi-taper channel PSD
    """
    tapered_signal = np.multiply(ch_data, dpss_tapers)

    n = len(ch_data)
    half_n = int(np.ceil(n/2))

    fft_mag = np.fft.fft(tapered_signal, n, axis=1)
    fft_mag = 2*fft_mag[:,:half_n]/np.sqrt(n)  # Normalize by window size.
    fft_mag[:,0] /= 2

    fft_power = np.real(fft_mag)**2 + np.imag(fft_mag)**2
    fft_power_aggregate = fft_power.T@wt # summation across tapers

    return fft_power_aggregate

def get_ch_psd(ch_data):
    """
    ch_data: (T,) dimension channel data
    returns FFT (V^2/Hz)
    """
    n = len(ch_data)
    half_n = int(np.ceil(n/2))

    fft_mag = np.fft.fft(ch_data)
    fft_mag = 2*fft_mag[:half_n]/np.sqrt(n)  # Normalize by window size.
    fft_mag[0] /= 2

    fft_power = np.real(fft_mag)**2 + np.imag(fft_mag)**2

    return fft_power

def compute_overall_std_dev(sigmas, ns):
    """
    mus: channel recording std dev. per session. np.nan if saturated
    ns: length of session

    returns std dev. of the aggregate sample

    example:
    n1, n2, n3 = 1000, 3000, 500
    x1 = np.random.normal(loc=0, scale=2, size=n1)
    x2 = np.random.normal(loc=0, scale=10, size=n2)
    x3 = np.random.normal(loc=0, scale=5, size=n3)
    y = np.concatenate((x1, x2, x3))

    sig1 = np.std(x1)
    sig2 = np.std(x2)
    sig3 = np.std(x3)
    sigy = np.std(y)

    sigy, compute_overall_std_dev([sig1, sig2, sig3], [n1, n2, n3])
    """
    if np.alltrue(np.isnan(sigmas)):
        return np.nan

    numel, denom = 0, 0
    for sigma, n in zip(sigmas, ns):
        if np.isnan(sigma):
            continue
        
        numel += (n-1)*sigma**2
        denom += (n-1)
    
    return np.sqrt(numel/denom)

def compute_overall_mean(mus, ns):
    """
    mus: channel recording mean. per session. np.nan if saturated
    ns: length of session
    """

    if np.alltrue(np.isnan(mus)):
        return np.nan

    numel, denom = 0, 0
    for mu, n in zip(mus, ns):
        if np.isnan(mu):
            continue
        
        numel += n*mu
        denom += n
    
    return (numel/denom)

def normalize_band(good_chs, band, mus, sigmas, sel_zscore=False):
    """
    good_chs: non-saturated channels
    band:     single band of a spectrogram. dimensions: ch * time, (ch: good_chs)
    spect_mu: spectrogram channel means across all sessions. dimension: 256
    spect_mu: spectrogram channel std devs across all sessions. dimension: 256

    returns normalized version of "band"
    """
    norm_band = np.zeros_like(band)
    assert band.shape[0] == len(good_chs)

    for idx, (ch, ch_data) in enumerate(zip(good_chs, band)):
        if sel_zscore:
            norm_band[idx,:] = (ch_data-mus[ch])/sigmas[ch]
        else:
            norm_band[idx,:] = ch_data/sigmas[ch]

    return norm_band

def normalize_spect(good_chs, spect, mus, sigmas, sel_zscore=False):
    norm_spect = np.zeros_like(spect)

    for f_idx in range(spect.shape[2]):
        norm_spect[:,:,f_idx] = normalize_band(good_chs, spect[:,:,f_idx],
                                                mus[:,f_idx], sigmas[:,f_idx], sel_zscore)
    
    return norm_spect

# def normalize_spectrogram(spect):
#     norm_spect = np.zeros_like(spect)
#     for ch, ch_data in enumerate(spect):
#         for fidx, freq_data in enumerate(ch_data.T):
#             sigma = np.std(freq_data)
#             assert sigma > 0
#             norm_spect[ch,:,fidx] = freq_data/sigma
    
#     return norm_spect

# def zscore_spectrogram(spect):
#     zspect = np.zeros_like(spect)

#     for ch, ch_data in enumerate(spect):
#         for fidx, freq_data in enumerate(ch_data.T):
#             mu = np.mean(freq_data)
#             sigma = np.std(freq_data)
#             assert sigma > 0
#             zspect[ch,:,fidx] = (freq_data - mu)/sigma
    
#     return zspect