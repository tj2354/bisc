## Supporting Codes for BISC

## Overview
Codes for Bioelectronic Interface System to the Cortex (BISC) device characterization and ephys recording analysis

The provided notebooks are used to generate figures and movies in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.05.17.594333v1) on the following topics:

1. in vitro characterization
2. porcine SSEP recordings
3. NHP motor cortex recordings

Files are arranged by their topics
```plaintext
├── in_vitro/scripts     # Scripts for in vitro characterization
├── nhp_motor/scripts    # Scripts for analyzing NHP motor cortex recordings
└── porcine_ssep/scripts # Scripts for analyzing porcine SSEP recordings
```

## System Requirements

### Software Dependencies
- **Python**: Version 3.10
- **Jupyter Notebook**: Ensure you have Jupyter installed to run the notebooks.
- **Required Python Packages and Tested Versions**:
  - numpy 1.23.5
  - scipy 1.10.0
  - scikit-learn 1.2.1
  - matplotlib
  - seaborn

### Operating Systems
- **Fully Tested on**: 
  - Windows 11 64-bit

- **Partially Tested on**: 
  - Linux RHEL 7.9

### Non-Standard Hardware
- **None**: No special hardware requirements.

## Installation Guide

### Our workflow recommends Python Virtual Environment and Pip package installer
1. **Install Python 3.10**:
   - Download the installer from the [official Python website](https://www.python.org/downloads/release/python-3100/) and follow the instructions for your operating system.
   
2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate # On Linux, use `source .venv/bin/activate`
   ```
   
3. **Install Jupyter Notebook**:
   ```bash
   pip install notebook
   ```
   
4. **Install Required Python Packages**:
    ```bash
    pip install numpy=1.23.5
    pip install scipy=1.10.10
    pip install scikit-learn=1.2.1
    pip install matplotlib
    pip install seaborn
    ```
### Typical Install Time
- **Estimated Time**: Expect each step to take no more than 10-20 minutes on a standard desktop computer with a stable internet connection.

## Data Availability
#### Data required for running the scripts are currently available [here](https://www.dropbox.com/scl/fo/jwlu060lf0s1vucltd4ye/AA-iy21UtPEzpokDILeVXnA?rlkey=1yl247obxbnwnh92gycrxyk1a&st=znpsilcx&dl=0). Prior to publication, this data will be migrated to a Zenodo repo with open access.

Just like the script directories, data is organized by topics
```plaintext
├── in_vitro/data     # Data from in vitro characterization
├── nhp_motor/data    # Data from NHP motor cortex recordings
└── porcine_ssep/data # Data from porcine SSEP recordings
```

We recommend merging the directories such that the resulting directories are as follows:
```plaintext
├── in_vitro     
│   ├── data          # Data from in vitro characterization
│   └── scripts       # Scripts for in vitro characterization
├── nhp_motor
│   ├── data          # Data from NHP motor cortex recordings
│   └── scripts       # Scripts for analyzing NHP motor cortex recordings
└── porcine_ssep
    ├── data          # Data from porcine SSEP recordings
    └── scripts       # Scripts for analyzing porcine SSEP recordings
```
## License
This repository is made available under a [CC-BY-NC-ND 4.0 International license](https://creativecommons.org/licenses/by-nc-nd/4.0/).
