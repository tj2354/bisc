import numpy as np

def draw_CS_boundary(ax):
    """
    draws central sulcus boundary over 16x16 pixel
    """
    # Coordinates of the CS
    line_x = [10, 10.5, 11, 11.25, 11.5, 11.5, 11.5, 11.25,
              11, 10.75, 10.5, 10, 9.25, 8.75, 8.25, 7.5, 6.75]
    line_y = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
              9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]

    # Plot the curvilinear line
    ax.plot(line_x, line_y, color='black', linewidth=4)

CS_COLS =np.array([
    # For each row,
    # pixel with column address less than idx0 are labeled as M1
    # pixel with column address greater or equal to idx1 are labeled as S1
    [10, 11], # [idx0, idx1]
    [11, 12],
    [11, 12],
    [11, 12],
    [12, 12],
    [12, 12],
    [11, 12],
    [11, 12],
    [11, 12],
    [11, 12],
    [10, 11],
    [10, 11],
    [ 9, 10],
    [ 9, 10],
    [ 8, 9 ],
    [ 7, 8 ]
]) 

def plot_spect_channel(ax, t, rec_data, zspect, ch, ch_idx, t0, t1, f0, f1,
                       motion_t, motion_x, motion_y, motion_z, pos_str,
                       key, q_cut = 0.01, motion_yoff=5, cmap='viridis'):
    """
    plot single channel spectrogram
    ax0: time domain motor feature
    ax1: time domain single channel recording
    ax2: single channel spectrogram
    """
    extent = [t0, t1, f1, f0]

    ch_zspect = zspect[ch_idx][:, :]
    vmin = np.quantile(ch_zspect, q_cut)
    vmax = np.quantile(ch_zspect, 1-q_cut)

    ax[0].plot(motion_t, motion_x + motion_yoff)
    ax[0].plot(motion_t, motion_y)
    ax[0].plot(motion_t, motion_z - motion_yoff)
    ax[0].legend(['x', 'y', 'z'])

    ax[1].plot(t, rec_data[:,ch_idx])

    ax[2].imshow(ch_zspect.T, aspect='auto', vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
    ax[2].invert_yaxis()

    ax[0].set_title(f'Session {key}. Channel {ch}')

    ax[0].set_ylabel(f'Wrist {pos_str}')
    ax[1].set_ylabel('Amplitude (a.u.)')
    ax[2].set_ylabel('Frequency (Hz)')

    ax[2].set_xlabel('Time (sec)')

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

def plot_band_power(ax, band_data, good_channels, t0, t1,
                       motion_t, motion_x, motion_y, motion_z,
                       pos_str, title_str, q_cut = 0.01, motion_yoff=5, cmap='bwr'):
    """
    plot band power of the array
    ax0: time domain motor feature
    ax1: band power. (x-axis: time, y-axis: channels)
    """

    extent = [t0, t1, good_channels[-1], good_channels[0]]

    # transpose if needed
    if not band_data.shape[0] == len(good_channels):
        band_data = band_data.T
    assert band_data.shape[0] == len(good_channels)

    vmin = np.quantile(band_data, q_cut)
    vmax = np.quantile(band_data, 1-q_cut)

    if cmap == 'bwr': # white = 0
        abs_vmax = max(-vmin, vmax)
        vmin = -1*abs_vmax
        vmax = abs_vmax

    ax[0].plot(motion_t, motion_x + motion_yoff)
    ax[0].plot(motion_t, motion_y)
    ax[0].plot(motion_t, motion_z - motion_yoff)
    ax[0].legend(['x', 'y', 'z'])

    ax[1].imshow(band_data, aspect='auto', vmin=vmin, vmax=vmax, 
                 extent=extent, cmap=cmap)
    ax[1].invert_yaxis()

    ax[0].set_title(title_str)

    ax[0].set_ylabel(f'Wrist {pos_str}')
    ax[1].set_ylabel('Channels')

    ax[1].set_xlabel('Time (sec)')

    ax[0].grid(True)
    ax[1].grid(True)

def plot_all_bands(ax, zlmp_data, zspect_beta, zspect_lga, zspect_hga, good_channels, t0, t1,
                    motion_t, motion_x, motion_y, motion_z,
                    key, pos_str, q_cut = 0.01, motion_yoff=5):
    """
    plot band powers of the array
    ax0: time domain motor feature
    ax1:high-gamma band power. (x-axis: time, y-axis: channels)
    ax2:low-gamma  band power. (x-axis: time, y-axis: channels)
    ax3:beta       band power. (x-axis: time, y-axis: channels)
    ax4:local motor potential. (x-axis: time, y-axis: channels)
    """

    extent = [t0, t1, good_channels[-1], good_channels[0]]


    for ii in range(5):
        ax[ii].grid(True)

    ax[0].plot(motion_t, motion_x + motion_yoff)
    ax[0].plot(motion_t, motion_y)
    ax[0].plot(motion_t, motion_z - motion_yoff)
    ax[0].set_ylabel(pos_str)
    ax[0].legend(['x', 'y', 'z'], loc=(1.03, 0))

    cmap = 'viridis'

    vmin_hga = np.quantile(zspect_hga, q_cut)
    vmax_hga = np.quantile(zspect_hga, 1-q_cut)
    ax[1].imshow(zspect_hga, aspect='auto', vmin=vmin_hga, vmax=vmax_hga,
                 extent=extent, cmap=cmap)
    ax[1].invert_yaxis()

    vmin_lga = np.quantile(zspect_lga, q_cut)
    vmax_lga = np.quantile(zspect_lga, 1-q_cut)
    ax[2].imshow(zspect_lga, aspect='auto', vmin=vmin_lga, vmax=vmax_lga,
                 extent=extent, cmap=cmap)
    ax[2].invert_yaxis()

    vmin_beta = np.quantile(zspect_beta, q_cut)
    vmax_beta = np.quantile(zspect_beta, 1-q_cut)
    ax[3].imshow(zspect_beta, aspect='auto', vmin=vmin_beta, vmax=vmax_beta,
                 extent=extent, cmap=cmap)
    ax[3].invert_yaxis()

    vmin_lmp = np.quantile(zlmp_data, q_cut)
    vmax_lmp = np.quantile(zlmp_data, 1-q_cut)
    abs_vmax_lmp = max(-vmin_lmp, vmax_lmp)
    ax[4].imshow(zlmp_data, aspect='auto', vmin=-1*abs_vmax_lmp, vmax=abs_vmax_lmp,
                 extent=extent, cmap='bwr')
    ax[4].invert_yaxis()

    ax[0].set_title(f'Session {key}')
    ax[1].set_ylabel('High-γ Band\nChannels')
    ax[2].set_ylabel('Low-γ Band\nChannels')
    ax[3].set_ylabel('β Band\nChannels')
    ax[4].set_ylabel('LMP\nChannels')
    ax[-1].set_xlabel('Time (sec)')