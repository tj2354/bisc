import random
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from itertools import product
from typing import Union
from scipy.optimize import minimize

from jarvis.utils import tqdm

from . import rcParams
from .alias import Array, Figure, Axes, Artist, Animation
from .data import get_stim, get_valid_idxs, get_triggered_average, get_baseline
from .cache import cached

CACHE_PATH = Path(__file__).parent.parent/rcParams.cache_path


@cached(CACHE_PATH)
def dot_triggered_average(
    session: dict,
    *,
    trial_idxs: Optional[set[int]] = None,
    **kwargs,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""Returns dot triggered average response.

    Args
    ----
    session:
        A session key, e.g. from the list returned by `get_sessions`.
    trial_idxs:
        Set of trial indices to be averaged. If ``None``, use all valid trials.
    kwargs:
        Keyword arguments for `get_triggered_average`.

    Returns
    -------
    xs, ys: (num_ticks,)
        Dot center coordinates, in visual degrees.
    taus: (num_stamps)
        Time stamps relative to dot onset, in milliseconds.
    responses_mean, responses_std: (num_channels, num_stamps, height, width)
        Mean and standard deviation by averaging over all dot occurances.
    counts: (height, width)
        Number of dot occurances, useful to compute standard error of mean.

    """
    if trial_idxs is None:
        trial_idxs, _ = get_valid_idxs(session)
    else:
        trial_idxs = list(trial_idxs)
    onsets, offsets, stim_params = get_stim(session)
    dot_locations = stim_params['dot_locations']
    num_dots = dot_locations.shape[1]

    dot_duration = (offsets-onsets)[trial_idxs].mean()/num_dots
    dot_duration = round(dot_duration/5)*5
    xs = sorted(list(np.unique(dot_locations[..., 0])))
    ys = sorted(list(np.unique(dot_locations[..., 1])))

    triggers = np.empty((len(ys), len(xs)), dtype=object)
    for i in range(len(ys)):
        for j in range(len(xs)):
            triggers[i, j] = set()
    for trial_idx in trial_idxs:
        for dot_idx in range(num_dots):
            i = ys.index(dot_locations[trial_idx, dot_idx, 1])
            j = xs.index(dot_locations[trial_idx, dot_idx, 0])
            triggers[i, j].add((trial_idx, dot_idx*dot_duration))
    xs = (np.array(xs).astype(float)-960)/64
    ys = (np.array(ys).astype(float)-540)/64

    taus = None
    responses_mean = np.empty(triggers.shape, dtype=object)
    responses_std = np.empty(triggers.shape, dtype=object)
    dot_ijs = list(product(range(len(ys)), range(len(xs))))
    random.shuffle(dot_ijs)
    for i, j in tqdm(dot_ijs, unit='dot', desc='RF mapping', leave=True):
        try:
            _taus, responses_mean[i, j], responses_std[i, j] = get_triggered_average(
                session, triggers=triggers[i, j], **kwargs,
            )
            assert np.unique(np.diff(_taus)).size==1
        except:
            print('session: {}, dot: {}, kwargs:\n{}'.format(session, (i, j), kwargs))
            raise
        if taus is None:
            taus = _taus
        else:
            assert np.all(_taus==taus)
    responses_mean = np.stack(responses_mean.ravel(), axis=2).reshape(-1, len(taus), len(ys), len(xs))
    responses_std = np.stack(responses_std.ravel(), axis=2).reshape(-1, len(taus), len(ys), len(xs))
    counts = np.array([len(t) for t in triggers.ravel()]).reshape(len(ys), len(xs))
    return xs, ys, taus, responses_mean, responses_std, counts


def gauss_1d(taus, A, tau0, sigma_t, b=0):
    g = A*np.exp(-0.5*((taus-tau0)/sigma_t)**2)+b
    return g

def gabor_1d(taus, A, tau0, sigma_t, omega_t, phi_t, b=0):
    g = A*np.exp(-0.5*((taus-tau0)/sigma_t)**2)*np.cos(omega_t*(taus-tau0)+phi_t)+b
    return g

def gauss_2d(xs, ys, A, x0, y0, sigma_s, b=0):
    g = A*np.exp(-0.5*(((xs-x0)/sigma_s)**2+((ys-y0)/sigma_s)**2))+b
    return g

def gabor_2d(xs, ys, A, x0, y0, sigma_s, omega_x, omega_y, phi_s, b=0):
    g = A*np.exp(-0.5*(((xs-x0)/sigma_s)**2+((ys-y0)/sigma_s)**2))*np.cos(omega_x*(xs-x0)+omega_y*(ys-y0)+phi_s)+b
    return g

def mixed_gauss(
    taus, xs, ys,
    A, tau0, sigma_t, omega_t, phi, x0, y0, sigma_s, b=0,
):
    _taus, _ys, _xs = np.meshgrid(taus, ys, xs, indexing='ij')
    g = b*np.ones((len(taus), len(ys), len(xs)))
    for i in range(len(A)):
        rf_t = gabor_1d(_taus, 1, tau0[i], sigma_t[i], omega_t[i], phi[i])
        rf_s = gauss_2d(_xs, _ys, 1, x0[i], y0[i], sigma_s[i])
        g += A[i]*rf_t*rf_s
    return g


def get_rf_data(session: dict, channel_idx: int, **kwargs):
    xs, ys, taus, responses_mean, _, _ = dot_triggered_average(session, **kwargs)
    r_base, r_unit = get_baseline(session, **{
        k: v for k, v in kwargs.items() if k not in ['tau_min', 'tau_max']
    })
    rf_data = (responses_mean[channel_idx]-r_base[channel_idx])/r_unit[channel_idx]
    b = rf_data.mean(axis=(1, 2))
    rf_data -= b[:, None, None]

    u, s, vh = np.linalg.svd(rf_data.reshape(len(taus), len(xs)*len(ys)))
    rf_t = u[:, 0]
    rf_s = vh[0].reshape((len(ys), len(xs)))
    c = s[0]
    return b, rf_data, rf_t, rf_s, c


def get_rf_fit(est_t, est_s, c):
    rf_fit = est_t[:, None, None]*est_s[None]*c
    return rf_fit


@cached(CACHE_PATH)
def fit_channel_rf(
    session: dict,
    *,
    channel_idx: int,
    type: float = 'Gauss',
    num_optims: int = 200,
    **kwargs,
) -> tuple[float, Array, Array, dict[str, Array]]:
    r"""Fits receptive field for one channel.

    Args
    ----
    session:
        Session key.
    channel_idx:
        Channel index.
    type:
        Type of fitting function, use 'Gauss' for banded signal power and 'Gabor'
        for raw signal.
    num_optims:
        Number of tries of optimization. Each try randomly starts from a guess
        uniformly drawn from the preset bounds, and the best fit is kept.
    kwargs:
        Keyword arguments for `dot_triggered_average`.

    Returns
    -------
    fvu:
        Fraction of variance unexplained.
    param_t, param_s:
        Best parameters found for temporal and spatial profile respectively. If
        `type` is 'Gauss', tempral profile can be retrieved by
        `gauss_1d(taus, *param_t)` and spatial profile can be retrieved by
        `gauss_2d(xs, ys, *param_s)`. If `type` is 'Gabor', use `gabor_1d` and
        `gabor_2d`.
    info:
        A dictionary containing detailed results of fitting, with following keys:
    - xs, ys, taus:
        1D array of space and time coordinates.
    - b: (len(taus),)
        Space-agnostic temporal baseline of dot-triggered average.
    - rf_data, rf_fit: (len(taus), len(ys), len(xs))
        Receptive field averaged from data, and the best fit of it. `rf_data` is
        the dot-triggered average response relative to baseline response, and
        scaled by across-trial standard deviation of the channel. The baseline
        dynamics `b` is removed. `rf_fit` is a rank-1 approximation computed
        from the separately fitted temporal and spatial profile.
    - rf_t, rf_s: (len(taus),) and (len(ys), len(xs))
        Temporal and spatial profile computed by SVD on `rf_data`.
    - est_t, est_s: (len(taus),) and (len(ys), len(xs))
        Fitted temporal and spatial profile.

    """
    rng = np.random.default_rng(num_optims)
    xs, ys, taus, _, _, _ = dot_triggered_average(session, **kwargs)
    _ys, _xs = np.meshgrid(ys, xs, indexing='ij')
    _, rf_data, rf_t, rf_s, c = get_rf_data(session, channel_idx, **kwargs)

    vlim = np.abs(rf_t).max()
    if type=='Gauss':
        func = gauss_1d
        bounds = (
            (-3*vlim, 3*vlim), (0, 150), (5, 100),
        )
    if type=='Gabor' or type=='Hybrid':
        func = gabor_1d
        bounds = (
            (-3*vlim, 3*vlim), (0, 150), (5, 100),
            (-2*np.pi/40, 2*np.pi/40), (-np.pi, np.pi),
        )
    min_loss, param_t = float('inf'), None
    for _ in tqdm(range(num_optims), unit='try', desc='Temporal', leave=False):
        res = minimize(
            lambda param: ((func(taus, *param)-rf_t)**2).mean(),
            [rng.uniform(l, h) for l, h in bounds], bounds=bounds,
        )
        if res.fun<min_loss:
            min_loss = res.fun
            param_t = res.x
    est_t = func(taus, *param_t)

    vlim = np.abs(rf_s).max()
    if type=='Gauss' or type=='Hybrid':
        func = gauss_2d
        bounds = (
            (-3*vlim, 3*vlim), (-2, 4), (-1, 5), (0.1, 5), (-vlim, vlim),
        )
    if type=='Gabor':
        func = gabor_2d
        bounds = (
            (-3*vlim, 3*vlim), (-2, 4), (-1, 5), (0.1, 5),
            (-2*np.pi/2, 2*np.pi/2), (-2*np.pi/2, 2*np.pi/2), (-np.pi/2, np.pi/2),
            (-vlim, vlim),
        )
    min_loss, param_s = float('inf'), None
    for _ in tqdm(range(num_optims), unit='try', desc='Spatial', leave=False):
        res = minimize(
            lambda param: ((func(_xs, _ys, *param)-rf_s)**2).mean(),
            [rng.uniform(l, h) for l, h in bounds], bounds=bounds,
        )
        if res.fun<min_loss:
            min_loss = res.fun
            param_s = res.x
    est_s = func(_xs, _ys, *param_s)

    rf_fit = get_rf_fit(est_t, est_s, c)
    fvu = ((rf_fit-rf_data)**2).mean()/rf_data.var()
    return fvu, param_t, param_s, est_t, est_s


def plot_rf_heatmap(
    ax: Axes, rf: Array, xs: Array, ys: Array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap = 'coolwarm',
) -> Artist:
    r"""Plots a frame of receptive field.

    Args
    ----
    ax:
        Axis to plot.
    rf: (height, width)
        A frame of receptive field.
    xs, ys:
        Spatial coordinates, in visual degrees.
    vmin, vmax:
        Heatmap value limits.
    cmap:
        Color map.

    """
    dx, dy = xs[1]-xs[0], ys[1]-ys[0]
    extent = [
        xs[0]-dx/2, xs[-1]+dx/2, ys[-1]+dy/2, ys[0]-dy/2,
    ]
    h = ax.imshow(rf, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
    fix_size = 0.3
    ax.plot([-fix_size, fix_size], [0, 0], color='green', linewidth=1)
    ax.plot([0, 0], [-fix_size, fix_size], color='green', linewidth=1)
    ax.axis('off')
    return h


def plot_scale_bar(ax: Axes, bar_len: float = 1., y_offset: float = 1.5, width=0.3) -> None:
    r"""Plots a scale bar."""
    ax.add_artist(Rectangle(
        (0, y_offset-width/2), bar_len, width, edgecolor='none', facecolor='black',
    ))
    ax.text(bar_len+0.5, y_offset, r'${:g}^\degree$'.format(bar_len), va='center')


def plot_rf_snapshots(
    rf: Array, xs: Array, ys: Array, taus: Array,
    subplots_kwargs: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap = 'coolwarm',
) -> tuple[Figure, Axes]:
    r"""Plots snapshots of a receptive field.

    Args
    ----
    rf: (num_stamps, height, width)
        Spatial temporal receptive field.
    xs, ys:
        Spatial coordinates, see `plot_rf_heatmap` for more details.
    taus:
        Time stamps, in milliseconds.
    subplots_kwargs:
        Keyword arguments for `subplots`.
    vmin, vmax:
        Heatmap value limits, if ``None``, use the extrema from all snapshots.
    cmap:
        Color map.

    """
    if subplots_kwargs is None:
        subplots_kwargs = {'nrows': 3, 'ncols': 5, 'figsize': (9, 6)}
    fig, axes = plt.subplots(**subplots_kwargs)
    tau_idxs = ((np.arange(axes.size)+0.5)/axes.size*len(taus)).astype(int)
    if vmin is None:
        vmin = rf[tau_idxs].min()
    if vmax is None:
        vmax = rf[tau_idxs].max()
    for i, ax in enumerate(axes.ravel()):
        h = plot_rf_heatmap(ax, rf[tau_idxs[i]], xs, ys, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title('{:g} ms'.format(taus[tau_idxs[i]]), fontsize='medium')
    plot_scale_bar(axes[0, 0])
    plt.colorbar(h, fraction=0.05, shrink=0.95, ax=axes, label='Response (A.U.)')
    return fig, axes


def save_rf_animation(
    rf: Array, xs: Array, ys: Array, taus: Array,
    filename: Union[Path, str],
    subplots_kwargs: Optional[dict] = None,
    play_every: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap = 'coolwarm',
    use_tqdm: bool = True,
) -> tuple[Figure, Animation]:
    r"""Saves animation of a receptive field.

    Args
    ----
    rf, xs, ys, taus:
        Receptive field and the coordinates, see `plot_rf_snapshots` for more
        details.
    filename:
        Path of saved animation.
    subplots_kwargs:
        Keyword arguments for `subplots`.
    play_every:
        Play interval, to speed up saving.
    vmin, vmax:
        Heatmap value limits, if ``None``, use the extrema from all frames.
    cmap:
        Color map.

    """
    if subplots_kwargs is None:
        subplots_kwargs = {'nrows': 1, 'ncols': 1, 'figsize': (4, 4)}
    fig, ax = plt.subplots(**subplots_kwargs)
    if vmin is None:
        vmin = rf.min()
    if vmax is None:
        vmax = rf.max()
    h_rf = plot_rf_heatmap(ax, rf[0], xs, ys, vmin=vmin, vmax=vmax, cmap=cmap)
    plot_scale_bar(ax)
    h_title = ax.set_title('', font='DejaVu Sans Mono')

    with tqdm(total=len(taus)//play_every+3, unit='frame', leave=True, disable=~use_tqdm) as pbar:
        def update(t):
            h_rf.set_data(rf[t*play_every])
            h_title.set_text(r'$\tau=$'+'{:-3d} ms'.format(int(np.round(taus[t*play_every]))))
            pbar.update()
            return h_rf, h_title

        ani = FuncAnimation(fig, update, frames=len(taus)//play_every, blit=True)
        fps = 20/(taus[1]-taus[0])/play_every
        ani.save(filename, fps=fps)
    return fig, ani


def save_rf_array_animation(
    rfs: Array, channel_idxs: Array,
    xs: Array, ys: Array, taus: Array,
    filename: Union[Path, str],
    subplots_kwargs: Optional[dict] = None,
    play_every: int = 4,
    vmin: Union[float, Array, None] = None,
    vmax: Union[float, Array, None] = None,
    cmap = 'coolwarm',
):
    r"""Saves receptive field animations of all channels.

    Args
    ----
    rf: (num_channels, num_stamps, height, width)
        Spatial temporal receptive field of all channels.
    xs, ys, taus:
        Spatial coordinates and time stamps, see `plot_rf_snapshots` for more
        details.
    filename:
        Path of saved animation.
    subplots_kwargs:
        Keyword arguments for `subplots`.
    play_every:
        Play interval, to speed up saving.
    vmin, vmax, cmap:
        Heatmap value limits and color map, see `plot_rf_heatmap` for more
        details.

    """
    num_channels = rfs.shape[0]
    if subplots_kwargs is None:
        n = int(num_channels**0.5)
        subplots_kwargs = {
            'nrows': n, 'ncols': n, 'figsize': (12, 12),
            'sharex': True, 'sharey': True,
            'gridspec_kw': {'hspace': 0.05, 'wspace': 0.05},
        }
    fig, axes = plt.subplots(**subplots_kwargs)
    if vmin is None:
        vmin = rfs.min(axis=(1, 2, 3))
    elif isinstance(vmin, float):
        vmin = [vmin]*num_channels
    assert len(vmin)==num_channels
    if vmax is None:
        vmax = rfs.max(axis=(1, 2, 3))
    elif isinstance(vmax, float):
        vmax = [vmax]*num_channels
    assert len(vmax)==num_channels
    h_rfs = {}
    for i, ax in enumerate(axes.ravel()):
        if i in channel_idxs:
            h_rfs[i] = plot_rf_heatmap(ax, rfs[i, 0], xs, ys, vmin=vmin[i], vmax=vmax[i], cmap=cmap)
        else:
            ax.axis('off')
    plot_scale_bar(axes[0, 0])
    h_title = axes[0, axes.shape[1]//2].set_title('', font='DejaVu Sans Mono')

    with tqdm(total=len(taus)//play_every+3, unit='frame', leave=False) as pbar:
        def update(t):
            for i in channel_idxs:
                h_rfs[i].set_data(rfs[i, t*play_every])
            h_title.set_text(r'$\tau=$'+'{:-3d} ms'.format(int(np.round(taus[t*play_every]))))
            pbar.update()
            return *[h_rfs[i] for i in channel_idxs], h_title

        ani = FuncAnimation(fig, update, frames=len(taus)//play_every, blit=True)
        fps = 20/(taus[1]-taus[0])/play_every
        ani.save(filename, fps=fps)
    return fig, ani
