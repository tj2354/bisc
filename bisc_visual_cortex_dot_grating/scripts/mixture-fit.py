import numpy as np
import torch
from scipy.optimize import minimize
from pathlib import Path
import os

from bisc import rcParams
rcParams['stimulus_path'] = '/mnt/d/BISC_2023/stimulus'
rcParams['response_path'] = '/mnt/d/BISC_2023/response'

from bisc.data import get_session, get_num_channels, get_valid_idxs
from bisc.dot import gabor_1d, gauss_2d, dot_triggered_average, get_rf_data
from jarvis.utils import tqdm

rng = np.random.default_rng()

cache_dir = Path(__file__).parent.parent/'cache'

saved = torch.load(cache_dir/'RF.temporal_v3.pt')
u0 = saved['u0']
u1 = saved['u1']

def fit_gabor_1d(rf_t, taus, num_optims=200):
    vlim = np.abs(rf_t).max()
    func = lambda param: gabor_1d(taus, *param)
    bounds = (
        (0, 3*vlim), (0, 150), (5, 100),
        (-2*np.pi/10, 2*np.pi/10), (-np.pi, np.pi),
    )

    min_loss, param = float('inf'), None
    for _ in range(num_optims):
        res = minimize(lambda param: ((func(param)-rf_t)**2).mean(),
            [rng.uniform(l, h) for l, h in bounds], bounds=bounds,
        )
        if res.fun<min_loss:
            min_loss = res.fun
            param = res.x
    rf_fit = func(param)
    return rf_fit, param

def fit_gauss_2d(rf_s, xs, ys, sigma_lim, num_optims=200):
    _ys, _xs = np.meshgrid(ys, xs, indexing='ij')
    vlim = np.abs(rf_s).max()
    func = lambda param: gauss_2d(_xs, _ys, *param)
    bounds = (
        (0, 3*vlim), (-2, 4), (-1, 5), sigma_lim, (-vlim, vlim),
    )

    min_loss, param = float('inf'), None
    for _ in range(num_optims):
        res = minimize(lambda param: ((func(param)-rf_s)**2).mean(),
            [rng.uniform(l, h) for l, h in bounds], bounds=bounds,
        )
        if res.fun<min_loss:
            min_loss = res.fun
            param = res.x
    rf_fit = func(param)
    return rf_fit, param

def get_rf_s(rf_data, rf_t, taus, xs, ys, sigma_lim=(0.1, 1), num_optims=200):
    # rf_fit_t, _ = fit_gabor_1d(rf_t, taus, num_optims)
    # rf_fit_t = rf_fit_t/(rf_fit_t**2).sum()**0.5
    # rf_s = (rf_data*rf_fit_t[:, None, None]).sum(axis=0)
    rf_s = (rf_data*rf_t[:, None, None]).sum(axis=0)
    rf_fit_s, param_s = fit_gauss_2d(rf_s, xs, ys, sigma_lim, num_optims)
    return rf_fit_s, param_s

TILES = [
    '39491886', '48683828', '02187077', '19889837',
    '22652138', '25394938', '27832912', '31080823',
    '05454007', '09690755', '76995123', '98782621',
    '07586668', '80605801', '37721134', '39666903',
]
CHIPS = [
    # '18712913', '99342065', '01178122',
    '07317162', '27621592', '48834689',
]

def main():
    dt = 0.5
    transform = {'type': 'remove_mean'}

    num_optims = 40
    # for tile_idx, session_id in enumerate(TILES, 1):
    #     filename = cache_dir/'RF.fit_tile{:02d}.pt'.format(tile_idx)
    #     # if os.path.exists(filename):
    #     #     continue
    #     print('Tile {:02d}'.format(tile_idx))
    for session_id in CHIPS:
        filename = cache_dir/'RF.fit_{}.pt'.format(session_id)
        # if os.path.exists(filename):
        #     continue
        session = get_session(session_id)
        num_channels = get_num_channels(session)
        _, channel_idxs = get_valid_idxs(session)
        xs, ys, taus, _, _, _ = dot_triggered_average(session, dt=dt, transform=transform)

        rfs_t0 = np.full((num_channels, len(taus)), fill_value=np.nan)
        rfs_t1 = np.full((num_channels, len(taus)), fill_value=np.nan)
        rfs_s0 = np.full((num_channels, len(ys), len(xs)), fill_value=np.nan)
        rfs_s1 = np.full((num_channels, len(ys), len(xs)), fill_value=np.nan)
        params_s0 = np.full((num_channels, 5), fill_value=np.nan)
        params_s1 = np.full((num_channels, 5), fill_value=np.nan)
        fvus_0 = np.full((num_channels,), fill_value=np.nan)
        fvus_1 = np.full((num_channels,), fill_value=np.nan)
        fvus = np.full((num_channels,), fill_value=np.nan)

        for channel_idx in tqdm(rng.permuted(channel_idxs), desc='Fit RF', unit='channel'):
            _, rf_data, _, _, _ = get_rf_data(session, channel_idx, dt=dt, transform=transform)

            # rf_t0 = u0
            # rf_fit_s0, param_s0 = get_rf_s(rf_data, rf_t0, taus, xs, ys, num_optims)
            # rf_fit0 = rf_t0[:, None, None]*rf_fit_s0[None]
            # rf_t1 = u1
            # rf_fit_s1, param_s1 = get_rf_s(rf_data-rf_fit0, rf_t1, taus, xs, ys, num_optims)
            # rf_fit1 = rf_t1[:, None, None]*rf_fit_s1[None]

            u, _, _ = np.linalg.svd(rf_data.reshape(len(taus), -1))
            if (u0*u[:, 0]).sum()>0:
                rf_t0 = u[:, 0]
            else:
                rf_t0 = -u[:, 0]
            # rf_t0 = u0
            rf_fit_s0, _ = get_rf_s(rf_data, rf_t0, taus, xs, ys, (0.1, 1.5), num_optims)
            rf_fit_s0 /= (rf_fit_s0**2).sum()**0.5
            rf_t0 = (rf_data*rf_fit_s0).sum(axis=(1, 2))
            rf_t0 /= (rf_t0**2).sum()**0.5
            rf_fit_s0, param_s0 = get_rf_s(rf_data, rf_t0, taus, xs, ys, (0.1, 1.5), num_optims)
            rf_fit0 = rf_t0[:, None, None]*rf_fit_s0[None]

            rf_t1 = u1
            rf_fit_s1, param_s1 = get_rf_s(rf_data-rf_fit0, rf_t1, taus, xs, ys, (0.3, 5), num_optims)
            rf_fit_s1 /= (rf_fit_s1**2).sum()**0.5
            rf_t1 = ((rf_data-rf_fit0)*rf_fit_s1).sum(axis=(1, 2))
            rf_t1 /= (rf_t1**2).sum()**0.5
            rf_fit_s1, param_s1 = get_rf_s(rf_data-rf_fit0, rf_t1, taus, xs, ys, (0.3, 5), num_optims)
            rf_fit1 = rf_t1[:, None, None]*rf_fit_s1[None]

            rf_t0 = u0
            rf_fit_s0, _ = get_rf_s(rf_data-rf_fit1, rf_t0, taus, xs, ys, (0.1, 1.5), num_optims)
            rf_fit_s0 /= (rf_fit_s0**2).sum()**0.5
            rf_t0 = ((rf_data-rf_fit1)*rf_fit_s0).sum(axis=(1, 2))
            rf_t0 /= (rf_t0**2).sum()**0.5
            rf_fit_s0, param_s0 = get_rf_s(rf_data-rf_fit1, rf_t0, taus, xs, ys, (0.1, 1.5), num_optims)
            rf_fit0 = rf_t0[:, None, None]*rf_fit_s0[None]

            rfs_t0[channel_idx] = rf_t0
            rfs_t1[channel_idx] = rf_t1
            rfs_s0[channel_idx] = rf_fit_s0
            rfs_s1[channel_idx] = rf_fit_s1
            params_s0[channel_idx] = param_s0
            params_s1[channel_idx] = param_s1
            fvus_0[channel_idx] = ((rf_fit0-rf_data)**2).mean()/rf_data.var()
            fvus_1[channel_idx] = ((rf_fit1-rf_data)**2).mean()/rf_data.var()
            fvus[channel_idx] = ((rf_fit0+rf_fit1-rf_data)**2).mean()/rf_data.var()

            torch.save({
                'taus': taus,
                # 'rf_t0': u0, 'rf_t1': u1,
                'rfs_t0': rfs_t0, 'rfs_t1': rfs_t1,
                'rfs_s0': rfs_s0, 'rfs_s1': rfs_s1,
                'params_s0': params_s0, 'params_s1': params_s1,
                'fvus_0': fvus_0, 'fvus_1': fvus_1, 'fvus': fvus,
            }, filename)


if __name__=='__main__':
    main()
