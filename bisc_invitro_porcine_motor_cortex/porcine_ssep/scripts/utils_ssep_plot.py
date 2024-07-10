import numpy as np
from matplotlib.patches import Rectangle
from scipy.signal import detrend

def plot_waterfall(ax, t, ch_means, sort_chs, t0, t1, voff, sel_detrend=True, title_str="",
                  sel_add_scalebar=True, linewidth=0.75, color='k'):

    t_idx = np.where((t > t0) & (t < t1))[0]

    for ii, ch in enumerate(sort_chs):
        offset = voff*ii

        y = ch_means[ch][t_idx]
        if sel_detrend:
            y = detrend(y)

        ax.plot(t[t_idx]/1e-3, y + offset, linewidth=linewidth, color=color)

    ax.set_title(title_str, y=0.97)

    # erase spines
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.set_xticks([])
    ax.set_yticks([])

    # add stim onset indicator
    if sel_add_scalebar:
        w0 = 2
        h0 = voff*10
        x0, y0 = -w0//2, -h0//2
        ax.add_patch(Rectangle((x0, y0), w0, np.abs(h0), facecolor='black'))
        ax.text(x0 + 4, y0 - voff, 'Stim', fontsize='medium')
    
        bbox_props = dict(facecolor='white', edgecolor='none',
                          alpha=0.75)
        # add time scale bar
        w1 = 25
        h1 = voff*1.25
        x1, y1 = 70, -2*voff
        ax.add_patch(Rectangle((x1, y1), w1, h1, facecolor='black'))
        ax.text(x1, y1 - voff*6, f'{w1} ms', fontsize='medium')
    
        # add voltage scale bar
        w2 = w0
        h2 = 200
        x2, y2 = x1, y1
        ax.add_patch(Rectangle((x2, y2), w2, h2, facecolor='black'))
        ax.text(x2 - 42, y2 + voff*8, f'{h2} μV', bbox=bbox_props, fontsize='medium')

##########

def erase_axis_spines(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    
def plot_spatiotemporal(fig, ax, t, data_means, data_stds, disqualified_chs, t0, t1,
                    title_str):
    t0, t1 = 0, 50e-3
    idxs = np.where((t >= t0) & (t <= t1))[0]

    ylim_min = np.nanmin(data_means[:, idxs])
    ylim_max = np.nanmax(data_means[:, idxs])

    fig.suptitle(title_str, y=1.0)
    fig.subplots_adjust(top=0.95)

    # make sure sharex, sharey are set to True
    ax[0,0].set_xlim(t0/1e-3, t1/1e-3)
    ax[0,0].set_ylim((ylim_min*1.01, ylim_max*1.01))

    for r in range(16):
        for c in range(16):
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            erase_axis_spines(ax[r,c])

            ch = r*16 + c
            if ch in disqualified_chs:
                continue

            y_mean = data_means[ch][idxs]
            y_std = data_stds[ch][idxs]
            under_line = (y_mean - y_std)
            over_line = (y_mean + y_std)

            if np.min(y_mean) < ylim_min: ylim_min = np.min(y_mean)
            if np.max(y_mean) > ylim_max: ylim_max = np.max(y_mean)


            ax[r,c].plot(t[idxs]/1e-3, y_mean, linewidth=2.0)
            ax[r,c].fill_between(t[idxs]/1e-3, under_line, over_line, color='gray',
                                 alpha=0.5)

    ylim_full = ylim_max - ylim_min
    xlim_full = (t1 - t0)/1e-3

    x0 = 0
    y0 = ylim_min
    h0 = ylim_full*0.1
    w0 = xlim_full
    ax[0,15].add_patch(Rectangle((x0, y0), w0, h0, facecolor='black'))

    # Add Time Scale Text
    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.0)
    ax[1,15].text(t0/1e-3, np.mean([ylim_min, ylim_max]) - 0.05*ylim_full, 
                   f'{w0:.0f} ms', bbox=bbox_props, fontsize='medium')

    # Add Voltage Scale Bar
    x1, y1 = x0, y0
    h1 = np.round(ylim_full/50)*50

    w1 = xlim_full*0.1
    ax[0,15].add_patch(Rectangle((x1, y1), w1, h1, facecolor='black'))

    # Add Voltage Scale Text
    if h1 > 1000:
        vscale_text = f'{h1*1e-3:.2f}\n mV'
    else:
        vscale_text = f'{h1:.0f}\n μV'
    ax[0,14].text(-30, ylim_min + ylim_full*0.2,
                   vscale_text, bbox=bbox_props, fontsize='medium')

##########

def plot_spatial_cbar(fig, ax):
    # create a blank 16x16
    ax.set_xticks([])
    ax.set_yticks([])
    erase_axis_spines(ax)

    im = ax.imshow(np.full((16, 16), np.nan), vmax=1, vmin=-1, cmap='bwr')
    ax.set_xlim((0, 16))
    ax.set_ylim((0, 16))
    ax.invert_yaxis()

    cax = ax.inset_axes([0.3, 0.6, 0.7, 0.075]) # posx, posy, lenx, leny
    cbar = fig.colorbar(im, cax=cax, ticks=[-1, 1], 
                        orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)

    cbar.set_label('Normalized\n SSEP', fontsize=16, labelpad=3)

def plot_spatial_ruler(ax):
    # draw ruler
    # arrow(x, y, dx, dy)
    hw = 1.0
    ax.arrow(0, 16, 14, 0, head_width=hw, fc='k', overhang=0.5, shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(16.25, 16, -14, 0, head_width=hw, fc='k', overhang=0.5, shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(0, 0, 0, 13.75, head_width=hw, fc='k', overhang=0.5, shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(0, 12.5, 0, -10.25, head_width=hw, fc='k', overhang=0.5, shape='full', clip_on=False, head_starts_at_zero=True)

    ax.text(8, 14, '6.8 mm', ha='center', va='top', fontsize=16)
    ax.text(0.5, 8, '7.4 mm', ha='left', va='center', rotation='vertical', fontsize=16)

def plot_spatial_compass(ax):
    # draw compass
    hw=0.8
    ax.arrow(9, 1, 2, 0, head_width=hw, head_length=hw, fc='k', shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(11, 1, -2, 0, head_width=hw, head_length=hw, fc='k', shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(10, 0, 0, 2, head_width=hw, head_length=hw, fc='k', shape='full', clip_on=False, head_starts_at_zero=True)
    ax.arrow(10, 2, 0, -2, head_width=hw, head_length=hw, fc='k', shape='full', clip_on=False, head_starts_at_zero=True)
    ax.text(6.5, 1.5, 'L')
    ax.text(12.5, 1.5, 'M')
    ax.text(9.25, -1.5, 'R')
    ax.text(9.25, 4.75, 'C')

def plot_spatial_peaks(fig, ax, data_sep_pks, title_strs, ax_rc):
    
    # plot spatial maps
    for ii, sep_pks in enumerate(data_sep_pks):
        r, c = ax_rc[ii]
    
        norm_sep_pks = sep_pks/np.nanmax(np.abs(sep_pks))
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].imshow(norm_sep_pks.reshape(16, -1), vmax=1, vmin=-1, cmap='bwr')
        ax[r, c].set_title(title_strs[ii], fontsize=20)
    
    # reserve one axis for a blank plot
    ax0 = ax[0, 2]
    plot_spatial_cbar(fig, ax0)
    
    # draw scalebar
    plot_spatial_ruler(ax0)
   
    # draw compass
    plot_spatial_compass(ax0)

##########

def plot_cbars(fig, ax, cmaps, vmins, vmaxs):
    ax.set_xticks([])
    ax.set_yticks([])
    erase_axis_spines(ax)

    ax.set_xlim((0, 16))
    ax.set_ylim((0, 16))
    ax.invert_yaxis()

    dy = 0.18
    for idx in range(5):
        im = ax.imshow(np.full((16, 16), np.nan), vmax=1, vmin=vmins[idx], cmap=cmaps[idx])


        cax = ax.inset_axes([0.22, 0.95 - dy*idx, 0.7, 0.05]) # posx, posy, lenx, leny
        cbar = fig.colorbar(im, cax=cax, ticks=[vmins[idx], vmaxs[idx]], orientation='horizontal') 
        cbar.ax.tick_params(labelsize=10)

        if idx == 0: # Add title to the top bar
            cbar.set_label('Norm. ESNR', fontsize=12, labelpad=-40)


def plot_spatial_band_power(fig, ax, band_powers, cmaps, q_cut_min, q_cut_max, vmin, vmax,
                            suptitle_str, title_strs, ax_rc):
    # plot spatial maps
    vmins, vmaxs = [], []
    for idx, band_power in enumerate(band_powers):
        r, c = ax_rc[idx]

        norm_band_power = band_power - np.nanmin(band_power)
        norm_band_power = norm_band_power/np.nanmax(np.abs(norm_band_power))

        if q_cut_min is not None:
            vmin = np.nanquantile(norm_band_power, q_cut_min)
        if q_cut_max is not None:
            vmax = np.nanquantile(norm_band_power, q_cut_max)

        vmins.append(np.round(vmin, 2))
        vmaxs.append(np.round(vmax, 2))

        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].imshow(norm_band_power.reshape(16, -1), vmin = vmin, vmax=1, cmap=cmaps[idx])
        ax[r, c].set_title(title_strs[idx], fontsize=20)
    

    fig.suptitle(suptitle_str)

    # reserve one axis for a blank plot
    ax0 = ax[0, 2]

    # plot cmaps
    plot_cbars(fig, ax0, cmaps, vmins, vmaxs)

    # draw scalebar
    plot_spatial_ruler(ax0)

def plot_combined_map_band_power(fig, ax, band_powers, cmaps, title_str, nthresh=None,
                                 vthresh=None):
    """
    nthresh: plot up to nthresh channels for each stimulation site
    vthresh: pixel will be colored if the maximum stimulation site response exceeds vthresh
    """
    nch = 256
    norm_band_powers = np.copy(band_powers)
    
    # normalize
    for norm_band_power in norm_band_powers:
        norm_band_power -= np.nanmin(norm_band_power)
        norm_band_power /= np.nanmax(norm_band_power)
        # print(np.nanmin(norm_band_power), np.nanmax(norm_band_power))

    # if there is a conflict between stim sites, take the largest
    for ch in range(256):
        y = norm_band_powers[:,ch] # normalized band power of a channel, for all stim sites
        if np.all(np.isnan(y)):
            continue

        # step 1. find index of stim site with largest norm. band power
        idx = np.nanargmax(y) 

        # step 2. invalidate all other stim sites
        norm_band_powers[:idx, ch] = np.nan
        norm_band_powers[idx+1:, ch] = np.nan

    if nthresh is not None: 
        for norm_band_power in norm_band_powers:
            valid_idxs = np.where(~np.isnan(norm_band_power))[0]

            # for each stimulation site, find the top XXX channels and invalidate all others
            top_chs = valid_idxs[np.argsort(norm_band_power[valid_idxs])[-nthresh:]]
            for ch in range(nch):
                if ch not in top_chs:
                    norm_band_power[ch] = np.nan

    # plot
    for ii, norm_band_power in enumerate(norm_band_powers):
        ax.set_xticks([])
        ax.set_yticks([])
        if vthresh is not None:
            norm_band_power[norm_band_power < vthresh] = np.nan
        im = ax.imshow(norm_band_power.reshape(16, -1), cmap=cmaps[ii])

    ax.set_title(title_str)