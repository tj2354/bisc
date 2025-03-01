### 1. Install the requirements
- required python version: >= 3.11.0

### 2. Run the notebook for traevling wave analysis
Full Notebook for running the traveling waves analysis is TW_Code.ipynb. Install all the python packages as shown in the notebook. This notebook loads data from lone example dense session with session_id = '39491886', resamples all data to 500 Hz sampling rate, and then runs the traveling wave analysis in the gamma frequency band with a parallel implementation to speed up the computational time. 

### 3. Supporting functions for traveling wave analysis 
Supporting functions to run the notebook TW_Code.ipynb are in par_funcs.py. The core funciton here is par_funcs.circ_lin_regress, which implements the circular-linear regression model. 

### 4. Erfan please add here your stuff
Output from the traveling waves analysis were given to the decoding analysis...
