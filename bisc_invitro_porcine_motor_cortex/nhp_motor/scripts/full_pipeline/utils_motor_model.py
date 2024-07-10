import numpy as np

def make_full_t_model_input(arr, t_idx0, t_idx1, t_idx_incr, f_idxs=None):
    """
    arr: input recording data (dimensions: either "channel x time x frequency" or "channel x time")

    arr truncated and downsampled in time domain.
    - truncation window defined by t_idx0, t_idx1.
    - downsamping factor defined by t_idx_incr

    returns (channel*lag*freq)*time tensor, used for decoder model input
    """
    X = []

    if not f_idxs is None: # spectrogram. DEPRECATED
        # There used to be a time when spectrogram (as opposed to bands)
        # was used to generate model input, but not anymore.
        # If using spectrogram, all frequency bins used would carry the same weights,
        # as opposed to using bands where for example, bins integrated over 70 - 190 Hz (HGA)
        # carry the same weight as bins integrated over 10 - 30 Hz (beta)
        for idx in range(t_idx0, arr.shape[1]-t_idx1, t_idx_incr):
            idx0 = idx - t_idx0
            idx1 = idx + t_idx1
            X.append(arr[:,idx0:idx1+1:t_idx_incr, f_idxs])
    else: # lmp, beta/gamma bands
        for idx in range(t_idx0, arr.shape[1]-t_idx1, t_idx_incr):
            idx0 = idx - t_idx0
            idx1 = idx + t_idx1
            X.append(arr[:,idx0:idx1+1:t_idx_incr])

    return np.array(X)

from utils_motor_global import SESSION_KEYS
from utils_motor_global import FULL_TAU_START, FULL_TAU_END, T_DF_MATRIX

def prepare_model_data(root_matrix_dir, root_motion_dir,
                        tau_start, tau_end, tau_df, t_df_motion,
                        sel_include_lfs, sel_channels, M1_ch_idxs, S1_ch_idxs):
    """
    given the specified input params:
    - root_matrix_dir: directory that contains per-session matrix data
    - root_motion_dir: directory that contains motor features
    - tau_start: (s). lag start
    - tau_end:   (s). lag end
    - tau_df: lag downsampling factor
    - t_df_motion: motor feature downsampling factor
    - sel_include_lfs: boolean flag on whether to include low frequency signal in the matrix
    - sel_channels: ternary flag on which channels to include in the matrix

    returns the following, merged across all sessions:
    X: input for the decoder model
    wrist_* : motor feature to decode against
    
    taus: range of time lags
    """

    """ Step 0. Define time lag and its dependent params """
    # get dt by loading a sample spect_t
    key = "006A"
    load_dir = f'{root_matrix_dir}/{key}'
    spect_t   = np.load(f'{load_dir}/spect_t_{key}.npy')
    dt = (spect_t[1] - spect_t[0])
    tau_idx0 = round(-FULL_TAU_START/dt)
    tau_idx1 = round(FULL_TAU_END/dt)
    win_tau = np.arange(-tau_idx0, tau_idx1+1, T_DF_MATRIX)*dt
    
    # shift by start_idx s.t. t=0 is included in the scalogram
    start_idx = ((len(win_tau) - 1)//2) % tau_df
    win_tau = win_tau[start_idx::tau_df] # this window will be further truncated below
    taus = win_tau[np.logical_and(tau_start < win_tau, win_tau < tau_end)]

    """ Step 1. Load Data """
    lmps, lfss, betas, lgas, hgas = [], [], [], [], []
    wrist_pos_xs, wrist_pos_ys, wrist_pos_zs = [], [], []
    wrist_vel_xs, wrist_vel_ys, wrist_vel_zs = [], [], []
    session_ids = []

    for idx, key in enumerate(SESSION_KEYS):
        """ load matrix data """
        load_dir = f'{root_matrix_dir}/{key}'
        # print(f'Loading session {key}..')

        # time*channel*lags
        session_lmp  = np.load(f'{load_dir}/zs_lmp_matrix_{key}.npy')
        session_lfs  = np.load(f'{load_dir}/norm_lfs_matrix_{key}.npy')
        session_beta = np.load(f'{load_dir}/norm_beta_matrix_{key}.npy')
        session_lga  = np.load(f'{load_dir}/norm_lga_matrix_{key}.npy')
        session_hga  = np.load(f'{load_dir}/norm_hga_matrix_{key}.npy')
    
        # downsample t and tau
        RS = int(t_df_motion/T_DF_MATRIX) # type casting only.
        assert RS == t_df_motion/T_DF_MATRIX

        session_lmp  = session_lmp [::RS,:,start_idx::tau_df]
        session_lfs  = session_lfs [::RS,:,start_idx::tau_df]
        session_beta = session_beta[::RS,:,start_idx::tau_df]
        session_lga  = session_lga [::RS,:,start_idx::tau_df]
        session_hga  = session_hga [::RS,:,start_idx::tau_df]
    
        # truncate tau

        session_lmp  = session_lmp [:,:,np.logical_and(tau_start < win_tau, win_tau < tau_end)]
        session_beta = session_beta[:,:,np.logical_and(tau_start < win_tau, win_tau < tau_end)]
        session_lga  = session_lga [:,:,np.logical_and(tau_start < win_tau, win_tau < tau_end)]
        session_hga  = session_hga [:,:,np.logical_and(tau_start < win_tau, win_tau < tau_end)]
        
        """ load motion data """
        motion_dir = f'{root_motion_dir}/{key}'
        wrist_pos_x = np.load(f'{motion_dir}/norm_wrist_pos_x_{key}.npy')
        wrist_pos_y = np.load(f'{motion_dir}/norm_wrist_pos_y_{key}.npy')
        wrist_pos_z = np.load(f'{motion_dir}/norm_wrist_pos_z_{key}.npy')
        wrist_vel_x = np.load(f'{motion_dir}/norm_wrist_vel_x_{key}.npy')
        wrist_vel_y = np.load(f'{motion_dir}/norm_wrist_vel_y_{key}.npy')
        wrist_vel_z = np.load(f'{motion_dir}/norm_wrist_vel_z_{key}.npy')
    
        # downsample 
        wrist_pos_x = wrist_pos_x[::t_df_motion]
        wrist_pos_y = wrist_pos_y[::t_df_motion]
        wrist_pos_z = wrist_pos_z[::t_df_motion]
        wrist_vel_x = wrist_vel_x[::t_df_motion]
        wrist_vel_y = wrist_vel_y[::t_df_motion]
        wrist_vel_z = wrist_vel_z[::t_df_motion]
    
        """ append to list """
        lmps.append(session_lmp)
        lfss.append(session_lfs)
        betas.append(session_beta)
        lgas.append(session_lga)
        hgas.append(session_hga)

        wrist_pos_xs.append(wrist_pos_x)
        wrist_pos_ys.append(wrist_pos_y)
        wrist_pos_zs.append(wrist_pos_z)
        wrist_vel_xs.append(wrist_vel_x)
        wrist_vel_ys.append(wrist_vel_y)
        wrist_vel_zs.append(wrist_vel_z)

        session_ids.append(np.full_like(wrist_vel_y, idx))

    """ Step 2. Merge Session Data """
    lmps  = np.concatenate(lmps, axis=0)
    lfss  = np.concatenate(lfss, axis=0)
    betas = np.concatenate(betas, axis=0)
    lgas  = np.concatenate(lgas, axis=0)
    hgas  = np.concatenate(hgas, axis=0)

    wrist_pos_xs = np.concatenate(wrist_pos_xs, axis=-1)
    wrist_pos_ys = np.concatenate(wrist_pos_ys, axis=-1)
    wrist_pos_zs = np.concatenate(wrist_pos_zs, axis=-1)
    wrist_vel_xs = np.concatenate(wrist_vel_xs, axis=-1)
    wrist_vel_ys = np.concatenate(wrist_vel_ys, axis=-1)
    wrist_vel_zs = np.concatenate(wrist_vel_zs, axis=-1)

    session_ids = np.concatenate(session_ids, axis=-1)
    
    if sel_channels == 'all':
        pass
    elif sel_channels == 'M1':
        lmps  = lmps [:,M1_ch_idxs,:]
        lfss  = lfss [:,M1_ch_idxs,:]
        betas = betas[:,M1_ch_idxs,:]
        lgas  = lgas [:,M1_ch_idxs,:]
        hgas  = hgas [:,M1_ch_idxs,:]
    elif sel_channels == 'S1':
        lmps  = lmps [:,S1_ch_idxs,:]
        lfss  = lfss [:,S1_ch_idxs,:]
        betas = betas[:,S1_ch_idxs,:]
        lgas  = lgas [:,S1_ch_idxs,:]
        hgas  = hgas [:,S1_ch_idxs,:]
    else:
        raise Exception

    """ Step 3. Combine LMP and Spectrogram, Flatten into X """
    if sel_include_lfs:
        X = np.zeros((lmps.shape[0], lmps.shape[1], lmps.shape[2], 5)) # tensor
        X[:,:,:,1] = lfss
        X[:,:,:,2] = betas
        X[:,:,:,3] = lgas
        X[:,:,:,4] = hgas
    else:
        X = np.zeros((lmps.shape[0], lmps.shape[1], lmps.shape[2], 4)) # tensor
        X[:,:,:,1] = betas
        X[:,:,:,2] = lgas
        X[:,:,:,3] = hgas
    X[:,:,:,0] = lmps

    X = X.reshape(X.shape[0], -1)

    """ return """
    return X, (wrist_pos_xs, wrist_pos_ys, wrist_pos_zs), \
        (wrist_vel_xs, wrist_vel_ys, wrist_vel_zs), session_ids, taus

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def sweep_model(n_splits, X, y, sel_model, hparam_range):
    """
    given the specified input params,
    - n_splits: number of splits for CV
    - X: model input
    - y: observed predictor
    - sel_model: selects which model to use. only supports PLS and Ridge
    - hparam_range: range of hyper params. only supports single hyper param

    builds and evaluates models, sweeping through the hyper parameter space and returns:
    - result_dict: key model performance metrics, evaluated against each hyper param
    """
    result_dict = {}
    for hparam in hparam_range:
        _, _, _, _, (presses, r2s, mses, rs) = build_model(n_splits, X, y, sel_model, hparam)

        result_dict[hparam] = {
            'presses': presses, # predictive error sum of squares
            'r2s': r2s,   # coefficient of determination
            'mses': mses, # mean square error
            'rs': rs      # correlation coefficient
        }

    return result_dict

def build_model(n_splits, X, y, sel_model, hparam):
    # opt_hparam = hparam_range[np.argmin(mse_avgs)]
    if sel_model == 'pls':
        model = PLSRegression(n_components=hparam)
    if sel_model == 'ridge':
        model = Ridge(alpha=hparam)

    train_idxs, test_idxs = [], []
    y_preds = []
    coefs, intercepts = [], []
    presses, r2s, mses, rs =  [], [], [], []

    kf = KFold(n_splits=n_splits)
    for train_idx, test_idx in kf.split(X, y):
        # fit model
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = np.squeeze(model.predict(X_test))

        # append
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
        y_preds.append(y_pred)

        # fetch coefficients
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

        # evaluate
        presses.append(np.sum((y_test - y_pred)**2))
        r2s.append(r2_score(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
        rs.append(np.corrcoef(y_test, y_pred)[0, 1])
    
    coefs = np.squeeze(np.array(coefs))
    intercepts = np.array(intercepts)
    presses = np.array(presses)
    r2s = np.array(r2s)
    mses = np.array(mses)
    rs = np.array(rs)
    return y_preds, test_idxs, coefs, intercepts, (presses, r2s, mses, rs)

from utils_motor_global import T_STEP_SCALO, T_DF_MOTION
def plot_decoded_y(fig, ax, n_splits, y, y_preds, test_idxs, rs, title_str):
    for idx in range(n_splits):
        t = np.arange(0, len(y_preds[idx]))*T_STEP_SCALO*T_DF_MOTION
        ax[idx].plot(t, y[test_idxs[idx]])
        ax[idx].plot(t, y_preds[idx])
        ax[idx].grid(True)
        ax[idx].set_ylabel(f'Split #{idx}\nr={rs[idx]:.2f}')
        ax[idx].set_yticks([])
        ax[idx].set_xlim(t[0], t[-1])

    ax[0].set_title(title_str)
    ax[-1].set_xlabel('Time (sec)')


def compute_and_plot_model_coeff_contributions(fig, ax, coefs, n_splits, chs, taus, band_strs,
                                               q_cut, title_str):
    nch = 256
    # good_chs = common_chs
    coefs = coefs.reshape((n_splits, len(chs), len(taus), len(band_strs)))
    coef_avg = np.mean(coefs, axis=0)

    """ compute contributions """
    # spatial
    w_ch = np.sum(np.sum(np.abs(coef_avg), axis=1), axis=1)
    z = np.zeros((nch, ))
    z[chs] = w_ch/np.sum(w_ch)*100
    z[z==0] = np.nan

    # temporal
    w_t  = np.sum(np.sum(np.abs(coefs), axis=3), axis=1)
    w_t = w_t/np.sum(w_t)*100*n_splits

    w_t_avg = np.mean(w_t, axis=0)
    w_t_stderr = np.std(w_t, axis=0)/np.sqrt(n_splits)

    # spectral
    w_f  = np.sum(np.sum(np.abs(coefs), axis=2), axis=1)
    w_f = w_f/np.sum(w_f)*100*n_splits

    w_f_avg = np.mean(w_f, axis=0)
    w_f_stderr = np.std(w_f, axis=0)/np.sqrt(n_splits)

    vmax = np.nanquantile(z, 1-q_cut)
    vmin = np.nanquantile(z, q_cut)

    """ plot """
    # spatial
    im = ax[0].imshow(z.reshape(16, -1), vmin=vmin, vmax=vmax)
    ax[0].set_title('Spatial')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    cbar = fig.colorbar(im, ax=ax[0], pad=0.05, shrink=1.0, location='left')
    cbar.set_label('Contribution (%)')

    # temporal
    ax[1].errorbar(taus, w_t_avg, yerr=w_t_stderr, fmt='o-', capsize=6) # markersize.

    ax[1].set_title('Temporal')
    ax[1].set_xlabel('Time Lag (s)')
    ax[1].set_ylabel('Contribution (%)')

    # spectral
    ax[2].bar(band_strs, w_f_avg)
    ax[2].errorbar(band_strs, w_f_avg, yerr=w_f_stderr, fmt='o', capsize=6, color='k')
    ax[2].tick_params(axis='x', labelrotation=90)

    ax[2].set_title('Spectral')
    ax[2].set_ylabel('Contribution (%)')
    fig.suptitle(title_str)
