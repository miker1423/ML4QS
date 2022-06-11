##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Simple class to impute missing values of a single columns.
class ImputationMissingValues:

    # Impute the mean values in case if missing data.
    def impute_mean(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    # Impute the median values in case if missing data.
    def impute_median(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    def linear_regression(self, dataset, col, features):
        size = dataset.shape[0]
        y_value = dataset[col].values.reshape(size, 1)
        x_value = dataset[features]
        reg = LinearRegression().fit(x_value, y_value)
        score = reg.score(x_value, y_value)
        print("Score: " + str(score) + " with: " + str(features) + "+"+col)
        return reg
        
