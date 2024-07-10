from bisc import rcParams
rcParams['stimulus_path'] = '/mnt/d/BISC_2023/stimulus'
rcParams['response_path'] = '/mnt/d/BISC_2023/response'

from bisc.data import get_session
from bisc.dot import dot_triggered_average

TILES = [
    '39491886', '48683828', '02187077', '19889837',
    '22652138', '25394938', '27832912', '31080823',
    '05454007', '09690755', '76995123', '98782621',
    '07586668', '80605801', '37721134', '39666903',
]

def main():
    dt = 0.5
    transform = {'type': 'remove_mean'}
    for tile_idx, session_id in enumerate(TILES, 1):
        print('Tile {:02d}'.format(tile_idx))
        session = get_session(session_id)
        dot_triggered_average(session, dt=dt, transform=transform)


if __name__=='__main__':
    main()
