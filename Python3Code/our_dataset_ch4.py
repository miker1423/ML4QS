import copy
from pathlib import Path
import pandas as pd
import time

from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/ours/')
DATASET_FNAME = 'ours_chapter3_result_final.csv'
RESULT_FNAME = 'ours_chapter4_result.csv'

def main():
    
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000
    
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    ws = int(float(0.5*60000)/milliseconds_per_instance)
    fs = float(1000)/milliseconds_per_instance

    selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

    dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
    dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
    
    CatAbs = CategoricalAbstraction()
    
    dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)


    periodic_predictor_cols = ['acc_x', 'acc_y', 'acc_z',
                               'gyr_x', 'gyr_y', 'gyr_z',
                               'mag_x', 'mag_y', 'mag_z']

    dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)
    # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

    # The percentage of overlap we allow
    window_overlap = 0.9
    skip_points = int((1-window_overlap) * ws)
    dataset = dataset.iloc[::skip_points,:]


    dataset.to_csv(DATA_PATH / RESULT_FNAME)



if __name__ == '__main__':
    
    main()