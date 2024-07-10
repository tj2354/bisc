from .alias import Array

def get_oracle_groups(img_ids: Array, flags: Array) -> dict[tuple[int], list[int]]:
    r"""Returns oracle trial groups.

    Args
    ----
    img_ids: (num_trials, num_imgs)
        Image IDs of all trials, read from `stim_params`.
    flags: (num_trials,)
        Trial flags from `stim_params`, with ``False`` marking oracle trials.

    Returns
    -------
    oracle_groups:
        A dictionary with image IDs as keys, and corresponding trial indices as
        values.

    """
    oracle_idxs, = (~flags).nonzero()
    oracle_groups = {}
    for trial_idx in oracle_idxs:
        _img_ids = tuple(img_ids[trial_idx])
        if _img_ids in oracle_groups:
            oracle_groups[_img_ids].append(trial_idx)
        else:
            oracle_groups[_img_ids] = [trial_idx]
    return oracle_groups
