import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles/ours')    
DATASET_FNAME = 'ours_chapter3_result_outliers.csv'
RESULT_FNAME = 'ours_chapter3_result_final.csv'
ORIG_DATASET_FNAME = 'ours_result.csv'

def main():
    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

        

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (
        dataset.index[1] - dataset.index[0]).microseconds/1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    
    # Now, for the final version. 
    # We first start with imputation by interpolation
    
    for col in [c for c in dataset.columns if not 'label' in c]:
        dataset = MisVal.impute_interpolate(dataset, col)

    # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
    periodic_measurements = ['acc_x', 'acc_y', 'acc_z', 
                             'gyr_x', 'gyr_y', 'gyr_z', 
                             'mag_x', 'mag_y', 'mag_z',]

    
    # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

    # Determine the sampling frequency.
    fs = float(1000)/milliseconds_per_instance
    cutoff = 1.5

    for col in periodic_measurements:
        dataset = LowPass.low_pass_filter(
            dataset, col, fs, cutoff, order=10)
        dataset[col] = dataset[col + '_lowpass']
        del dataset[col + '_lowpass']

    # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset
    
    selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
    
    n_pcs = 7
    
    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)
    # Store the final outcome.

    dataset.to_csv(DATA_PATH / RESULT_FNAME)

if __name__ == '__main__':
    main()