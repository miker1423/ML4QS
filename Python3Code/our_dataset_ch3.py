from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.
DATA_PATH = Path('./intermediate_datafiles/ours')
DATASET_FNAME = 'ours_result.csv'
RESULT_FNAME = 'ours_chapter3_result_outliers.csv'


def main():
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

    except IOError as e:
        print('File not found, try to run the preceding crowdsignals scripts first!')
        raise e

    outlier_columns = ['hr_watch_rate']
    OutlierDistr = DistributionBasedOutlierDetection()
    
    # We use Chauvenet's criterion for the final version and apply it to all but the label data...
    for col in [c for c in dataset.columns if not 'label' in c]:

        print(f'Measurement is now: {col}')
        dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
        dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
        del dataset[col + '_outlier']

    dataset.to_csv(DATA_PATH / RESULT_FNAME)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")

    FLAGS, unparsed = parser.parse_known_args()

    main()