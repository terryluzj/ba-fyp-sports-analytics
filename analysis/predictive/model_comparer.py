import numpy as np
import pandas as pd
import re

from analysis.predictive.feature_engineering import feature_engineer
from analysis.predictive.settings import DEPENDENT_COLUMNS_FEATURED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score


class ModelComparer(object):
    # Helper class to incorporate different regression models
    run_time_col_name = 'run_time_1000'
    
    def __init__(self, x_df, y_df, original_y_df_dict,
                 sampled=False, random_split=False, ratio=0.85, drop_last=True, **kwargs):
        """
        :param x_df: pandas dataframe of independent variables of the dataset
        :param y_df: pandas dataframe dependent variables of the dataset
        :param original_y_df_dict: dictionary mapping back column values
        :param sampled: boolean value indicating the dataframe is sampled or not
        :param random_split: boolean value indicating to do random split
        :param ratio: float value between 0 and 1 to indicate sampling ratio
        :param drop_last: boolean value indicating last running time information to be dropped
        :param kwargs: kwargs for train_test_split
        """

        # Get feature engineered dataframe
        self.X, self.y_rank = feature_engineer(df=x_df.reset_index(),
                                               df_name='model_featured' if not sampled else 'model_featured_sampled')
        self.y = y_df[y_df.index.isin(self.X.index)]
        self.y_original = original_y_df_dict

        # Do random split or split by date accordingly
        if random_split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                    test_size=1-ratio, **kwargs)
        else:
            # Print out date range for training set and testing set
            print('Split training and testing set by date range with %s ratio:' % ratio)
            split_index = int(self.X.shape[0] * ratio)
            self.X = self.X.reset_index().sort_values('run_date')
            last_date = self.X.iloc[split_index]['run_date']
            self.X_train = self.X.loc[self.X['run_date'] < last_date].set_index(['horse_id', 'run_date'])
            print('Training set date range: %s -> %s' % (self.X_train.index[0][1], self.X_train.index[-1][1]))
            self.X_test = self.X.loc[self.X['run_date'] >= last_date].set_index(['horse_id', 'run_date'])
            print('Testing set date range: %s -> %s' % (self.X_test.index[0][1], self.X_test.index[-1][1]))

            # Set training and testing prediction values accordingly
            self.y_train = self.y[self.y.index.isin(self.X_train.index)]
            self.y_test = self.y[self.y.index.isin(self.X_test.index)]
            self.X = self.X.set_index(['horse_id', 'run_date'])

        # Set some helper variables to be used in model training
        self.sorted_cols = list(sorted(self.y.columns, key=lambda x: len(x)))
        self.run_time_series = self.y[self.run_time_col_name]

        # Set some variables to store model information
        self.models = []
        self.model_dict = {}
        self.meta_models = {}
        self.predictions = {}
        self.meta_predictions = {}

        # Boolean variable to indicate whether to drop last run time information for training
        self.drop_last = drop_last
        
    def add_model(self, model_method, model_name, **params):

        # Initiate model dict
        self.model_dict[model_name] = {}

        # Iterate through each element name and do model training
        for y_col_name in self.sorted_cols:
            # Initiate new model to refit the training data
            model = model_method(**params)
            self.models.append(model)
            self.model_dict[model_name]['Model Spec'] = repr(model)
            
            # Uncomment the following lines to observe DV prediction without last run time info
            if self.drop_last and y_col_name != self.run_time_col_name:
                x_train = self.X_train.drop('last_run_time', axis=1)
                x_test = self.X_test.drop('last_run_time', axis=1)
            else:
                x_train = self.X_train
                x_test = self.X_test

            # Get rows without missing values only
            y_train = self.y_train[y_col_name].dropna()
            y_test = self.y_test[y_col_name].dropna()
            x_train = x_train[x_train.index.isin(y_train.index)]
            x_test = x_test[x_test.index.isin(y_test.index)]
            
            if 'normalized' in model_name.lower():
                # Normalize the dataset when specified in model name as 'normalized'
                x_train, x_test = self.get_normalized_train_test(x_train, x_test)

            # Print progress and fit the model
            self.print_progress(y_col_name=y_col_name, model_name=model_name, train_data=x_train)
            model.fit(x_train, y_train)

            # Add model information for stacking model as well
            if 'meta' in model_name.lower():
                if model_name not in self.meta_models.keys():
                    self.meta_models[model_name] = {}
                    self.meta_predictions[model_name] = {}
                self.meta_models[model_name][y_col_name] = model
                self.meta_predictions[model_name][y_col_name] = model.predict_meta_features(x_test)

            # Get prediction of testing date and evaluate
            y_pred = model.predict(x_test)
            if y_col_name != self.run_time_col_name:
                # Transform accordingly by quotient or difference
                transformed = self.get_transformed(y_col_name, y_pred, x_test)
                y_pred = transformed[0]
                run_time = transformed[1]

                # Get performance score and indexing for report dataframe
                rmse = self.get_rmse(y_pred, run_time)
                r_squared = self.get_r_squared(y_pred, run_time)
                rmse_col = 'Transformed RMSE: %s (%s)' % (y_col_name, x_train.shape[0])
                r_squared_col = 'Transformed R^2: %s (%s)' % (y_col_name, x_train.shape[0])

                # Set index values
                self.model_dict[model_name][rmse_col] = '%.6f' % rmse
                self.model_dict[model_name][r_squared_col] = '%.3f' % r_squared
            else:
                # Get performance score and indexing for report dataframe
                rmse = self.get_rmse(y_pred, y_test)
                r_squared = self.get_r_squared(y_pred, y_test)
                rmse_col = 'Original RMSE: %s (%s)' % (y_col_name, x_train.shape[0])
                r_squared_col = 'Original R^2: %s (%s)' % (y_col_name, x_train.shape[0])

                # Set index values
                self.model_dict[model_name][rmse_col] = '%.6f' % rmse
                self.model_dict[model_name][r_squared_col] = '%.3f' % r_squared

            # Store predictions for future use
            if y_col_name not in self.predictions.keys():
                self.predictions[y_col_name] = {}
            self.predictions[y_col_name][model_name] = y_pred

        # Print current progress
        print('\n' + ('Added model named %s ' % model_name))

    def get_transformed(self, y_col, pred, df):
        # Transform the predicted value by quotient or difference
        pred = pd.Series(pred, index=df.index)

        # Benchmark with original run time series
        run_time = self.run_time_series[self.run_time_series.index.isin(pred.index)]
        if y_col != self.run_time_col_name:
            series = self.y_original[y_col]
            series = series[series.index.isin(pred.index)]

            # Get original values from predictions
            if 'quo' in y_col:
                pred = pred * series
            elif 'diff' in y_col:
                pred = pred + series

        # Return prediction and actual value filtered by missing values
        return pred, run_time
        
    def get_report(self, filter_word=''):
        # Generate performance report

        def rank_index(index_string):
            # Rank index for better interpretation
            shape_number = int(re.search(r'\((\d+)\)', index_string).group(1))
            if 'original' in index_string.lower():
                return shape_number + 1
            else:
                return shape_number

        # Return the performance report as a dataframe
        try:
            df = pd.DataFrame.from_dict(self.model_dict)
            return_df = df.loc[list(filter(lambda x: filter_word in x, df.index))]
            reindex_df = return_df.reindex(index=sorted(return_df.index, reverse=True, key=lambda x: rank_index(x)))
            return reindex_df
        except IndexError:
            return

    def get_meta_report(self):
        # Return the mega-feature importance of stacking model as a dataframe
        new_dict = {}
        for meta_model_name in self.meta_models.keys():
            for key, value in self.meta_models[meta_model_name].items():
                # Get feature importance or coefficients
                try:
                    feature_importance = value.meta_regr_.feature_importances_
                except AttributeError:
                    feature_importance = value.meta_regr_.coef_

                # Create column names for meta report
                regressors = list(map(lambda x: re.search(r'(\w+)\(', repr(x)).group(1), value.regr_))
                final_dict = dict(zip(regressors, feature_importance))
                final_dict.update({'model_name': meta_model_name})
                new_dict.update(({key: final_dict}))

        # Return transposed dataframe of meta report
        return pd.DataFrame(new_dict).T

    def get_meta_models(self):
        # Return all the stacking models
        return list(self.meta_models.keys())

    @staticmethod
    def get_normalized_train_test(train, test):
        # Normalize the dataset when specified in model name as 'normalized'
        scaler = StandardScaler()
        scaler.fit(train)
        x_train = pd.DataFrame(scaler.transform(train), index=train.index)
        x_test = pd.DataFrame(scaler.transform(test), index=test.index)
        return x_train, x_test

    @staticmethod
    def print_progress(y_col_name, model_name, train_data):
        print('%s: Performing analysis on column %s for model %s (Size: %s)' %
              (ModelComparer.get_progress(y_col_name), y_col_name, model_name, str(train_data.shape)),
              end='\r', flush=True)

    @staticmethod
    def get_operator(target_column):
        # Return the operation specified in a column name
        return target_column.split('_')[-1]

    @staticmethod
    def get_progress(element):
        # Print out training process
        return '[' + \
               '>' * (DEPENDENT_COLUMNS_FEATURED.index(element) + 1) + \
               '-' * (len(DEPENDENT_COLUMNS_FEATURED) - DEPENDENT_COLUMNS_FEATURED.index(element) - 1) + \
               ']'
    
    @staticmethod
    def get_rmse(y_true, y_pred):
        # Return Root-Mean Square Error by given predictions and actual values
        diff = np.sum((y_true - y_pred) ** 2)
        return (diff / y_true.shape[0]) ** 1/2
    
    @staticmethod
    def get_r_squared(y_true, y_pred):
        # Return R-squared values
        return explained_variance_score(y_pred=y_pred, y_true=y_true)
