import csv
import numpy as np

# recording measurement data loading and parseing
def parse_Hf_filename(fn):
    elems = fn.split('_')
    vga_gain = int(elems[3])
    vbias_pr = int(elems[6])
    en_static_elec = int(elems[10])
    input_freq = int(elems[-1].split('Hz')[0])
    
    return vga_gain, vbias_pr, en_static_elec, input_freq

def parse_noise_filename(fn):
    elems = fn.split('_')
    vga_gain = int(elems[4])
    vbias_pr = int(elems[7])
    en_static_elec = int(elems[11].split('.')[0])

    return vga_gain, vbias_pr, en_static_elec

def find_matching_filename(flist, f_parse, vga_gain, vbias_pr, en_static_elec):

    for fn in flist:
        if (vga_gain, vbias_pr, en_static_elec) == (f_parse(fn)):
            return fn

    print('file not found')
    return None


# stimulation measurement data loading and parsing
def parse_stim_filename(fn: str):
    """
    given filename, parse the BISC configuration params used to measure the corresponding data
    """
    s = fn.split('.')[0].split('_') # get rid of .csv and then split by '_'
    ibias_global    = int(s[2])
    ibias_stim      = int(s[5])
    prog_set        = int(s[8])
    polarity        = int(s[10])
    negative        = int(s[12])
    positive        = int(s[14])
    balance         = int(s[16])
    num_rep         = int(s[19])
    num_spikes      = int(s[22])
    extra_balance   = int(s[25])
    return ibias_global, ibias_stim, prog_set, polarity, negative, positive, balance, num_rep, num_spikes, extra_balance

def load_stim_csv(load_dir, fn):
    load_fpath = f'{load_dir}/{fn}'
    with open(load_fpath, "r") as file:
        reader = csv.reader(file, delimiter=",")
        x = list(reader)

    return np.array(x[1:]).astype("float")