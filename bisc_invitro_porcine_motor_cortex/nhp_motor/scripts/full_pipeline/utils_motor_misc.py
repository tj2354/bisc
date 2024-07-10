import numpy as np
import os

def list_files_with_keyword_extension(directory, keyword, extension):
    """
    returns list of files in a "directory" that contains "keyword" in its name and
    matches the "extension"
    """
    matching_files = []

    # List all files in the directory
    all_files = os.listdir(directory)

    # Iterate over each file
    for filename in all_files:
        # Check if the file has ".csv" extension and the keyword in the filename
        if filename.endswith(extension) and keyword in filename:
            matching_files.append(filename)

    return matching_files

def load_session_spect_data(load_dir, key):
    """
    loading helper..
    """
    good_chs    = np.load(f'{load_dir}/good_channels_{key}.npy')
    spect       = np.load(f'{load_dir}/spect_{key}.npy'     ) 
    lmp_data    = np.load(f'{load_dir}/lmp_{key}.npy'       ) 
    spect_t     = np.load(f'{load_dir}/spect_t_{key}.npy'   ) 

    return good_chs, spect, lmp_data, spect_t
