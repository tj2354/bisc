import os
from pathlib import Path
import numpy as np
import torch
from jarvis.utils import tqdm

from bisc import rcParams

rcParams['stimulus_path'] = '/mnt/d/BISC_2023/stimulus'
rcParams['response_path'] = '/mnt/d/BISC_2023/response'

from bisc.data import get_session, get_baseline, get_num_channels, get_valid_idxs, get_raw_responses
from bisc.data import get_session, get_trial_average
from bisc.dot import dot_triggered_average, get_rf_data, plot_rf_snapshots
from bisc.dot import gauss_2d
from scipy.optimize import minimize

def fit_gauss_2d(rf_s, xs, ys, sigma_lim=(0.1, 1.5), num_optims=200):
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

rng = np.random.default_rng()

cache_dir = Path(__file__).parent.parent/'cache'

TILES = [
    '39491886', '48683828', '02187077', '19889837',
    '22652138', '25394938', '27832912', '31080823',
    '05454007', '09690755', '76995123', '98782621',
    '07586668', '80605801', '37721134', '39666903',
]

def main():
    dt = 0.5

    for tile_idx, session_id in enumerate(TILES, 1):
        print('Tile {:02d}'.format(tile_idx))

        session = get_session(session_id)
        for freq in [8]:
            filename = cache_dir/'RF.fit_{}_[{:g}Hz].pt'.format(session_id, freq)
            if os.path.exists(filename):
                continue

            transform = {'type': 'morlet', 'freq': freq}
            _, channel_idxs = get_valid_idxs(session)
            xs, ys, taus, responses_mean, _, _ = dot_triggered_average(session, dt=dt, transform=transform)
            r_base, r_unit = get_baseline(session, dt=dt, transform=transform)

            rfs = (responses_mean-r_base[:, None, None, None])/r_unit[:, None, None, None]
            rfs_s = rfs.mean(axis=1)

            params_s = np.full((1024, 5), fill_value=np.nan)
            fvus = np.full((1024,), fill_value=np.nan)
            for channel_idx in tqdm(rng.permuted(channel_idxs), desc='Fit Gauss', unit='channel'):
                rf_s = rfs_s[channel_idx]
                rf_fit, params_s[channel_idx] = fit_gauss_2d(rf_s, xs, ys)
                fvus[channel_idx] = ((rf_fit-rf_s)**2).mean()/rf_s.var()
            torch.save({
                'params_s': params_s, 'fvus': fvus,
            }, filename)

if __name__=='__main__':
    main()