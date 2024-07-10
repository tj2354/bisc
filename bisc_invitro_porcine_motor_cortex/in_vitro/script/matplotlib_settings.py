# plot_settings.py

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def set_plot_settings():
    plt.rcParams['font.size']=16
    plt.rcParams['axes.titlesize']=18
    plt.rcParams['lines.linewidth']=2
    plt.rcParams['xtick.labelsize']=14
    plt.rcParams['ytick.labelsize']=14

    # Set font to Helvetica
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.serif"] = "Helvetica"
    plt.rcParams["text.usetex"] = False

def reset_plot_settings():
    plt.rcParams.update(plt.rcParamsDefault)
