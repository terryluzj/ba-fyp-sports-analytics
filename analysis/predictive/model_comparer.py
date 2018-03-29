import numpy as np
import pandas as pd
import re

from feature_engineering import feature_engineer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score


class ModelComparer(object):
    # Helper class to incorporate different regression models
    run_time_col_name = 'run_time_1000'
    
    def __init__(self, X_df, y_df, original_y_df_dict,
                 sampled=False, random_split=False, ratio=0.7, drop_last=True, **kwargs):
        '''
        :param X_df: pandas dataframe of independent variables of the dataset
        :param y_df: pandas dataframe dependent variables of the dataset
        :param original_y_df_dict: dictionary mapping back column values
        :param sampled: boolean value indicating the dataframe is sampled or not
        :param random_split: boolean value indicating to do random split
        :param ratio: float value between 0 and 1 to indicate sampling ratio
        :param drop_last: boolean value indicating last running time information to be dropped
        :param kwargs: kwargs for train_test_split
        '''

        # Get feature engineered dataframe
        self.X = feature_engineer(df=X_df.reset_index(), df_name='df_combined_all' if not sampled else 'df_sampled')
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
            self.X_train = self.X.iloc[:split_index].set_index(['horse_id', 'run_date'])
            print('Training set date range: %s -> %s' % (self.X_train.index[0][1], self.X_train.index[-1][1]))
            self.X_test = self.X.iloc[split_index:].set_index(['horse_id', 'run_date'])
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

        # Boolean variable to indicate whether to drop last run time information for training
        self.drop_last = drop_last
        
    def add_model(self, model_method, model_name, **params):
        
        self.model_dict[model_name] = {}
        
        for y_col_name in self.sorted_cols:
            # Iterate through each element name and do model training
            model = model_method(**params)
            self.models.append(model)
            self.model_dict[model_name]['Model Spec'] = repr(model)
            
            # Uncomment the following lines to observe DV prediction without last run time info
            if self.drop_last and y_col_name != self.run_time_col_name:
                X_train = self.X_train.drop('last_run_time', axis=1)
                X_test = self.X_test.drop('last_run_time', axis=1)
            else:
                X_train = self.X_train
                X_test = self.X_test

            # Get rows without missing values only
            y_train = self.y_train[y_col_name].dropna()
            y_test = self.y_test[y_col_name].dropna()
            X_train = X_train[X_train.index.isin(y_train.index)]
            X_test = X_test[X_test.index.isin(y_test.index)]
            
            if 'normalized' in model_name.lower():
                # Normalize the dataset when specified in model name as 'normalized'
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

            # Fit the model
            print('%s: Performing analysis on column %s for model %s (Size: %s)' %
                  (self.get_progress(y_col_name), y_col_name, model_name, str(X_train.shape)), end='\r', flush=True)
            model.fit(X_train, y_train)

            # Add model information for stacking model as well
            if 'meta' in model_name.lower():
                if model_name not in self.meta_models.keys():
                    self.meta_models[model_name] = {}
                self.meta_models[model_name][y_col_name] = model

            # Get prediction of testing date and evaluate
            y_pred = model.predict(X_test)
            if y_col_name != self.run_time_col_name:
                # Transform accordingly by quotient or difference
                transformed = self.get_transformed(y_col_name, y_pred, X_test)
                y_pred = transformed[0]
                run_time = transformed[1]
                self.model_dict[model_name]['Transformed RMSE: %s (%s)' %
                                            (y_col_name, X_train.shape[0])] = '%.6f' % \
                                                                              self.get_rmse(y_pred, run_time)
                self.model_dict[model_name]['Transformed R^2: %s (%s)' %
                                            (y_col_name, X_train.shape[0])] = '%.3f' % \
                                                                              self.get_r_squared(y_pred, run_time)
            else:
                self.model_dict[model_name]['Original RMSE: %s (%s)' %
                                            (y_col_name, X_train.shape[0])] = '%.6f' % \
                                                                              self.get_rmse(y_pred, y_test)
                self.model_dict[model_name]['Original R^2: %s (%s)' %
                                            (y_col_name, X_train.shape[0])] = '%.3f' % \
                                                                              self.get_r_squared(y_pred, y_test)

            # Store predictions for future use
            if y_col_name not in self.predictions.keys():
                self.predictions[y_col_name] = {}
            self.predictions[y_col_name][model_name] = y_pred

        print('\n' + ('Added model named %s ' % model_name))

    def get_transformed(self, y_col, pred, df):
        # Transform the predicted value by quotient or difference
        pred = pd.Series(pred, index=df.index)
        run_time = self.run_time_series[self.run_time_series.index.isin(pred.index)]
        if y_col != self.run_time_col_name:
            series = self.y_original[y_col]
            series = series[series.index.isin(pred.index)]
            if 'quo' in y_col:
                pred = pred * series
            elif 'diff' in y_col:
                pred = pred + series

        # Return prediction and actual value filtered by missing values
        return pred, run_time
        
    def get_report(self, filter_word=''):
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
                try:
                    feature_importance = value.meta_regr_.feature_importances_
                except AttributeError:
                    feature_importance = value.meta_regr_.coef_
                regressors = list(map(lambda x: re.search(r'(\w+)\(', repr(x)).group(1), value.regr_))
                final_dict = dict(zip(regressors, feature_importance))
                final_dict.update({'model_name': meta_model_name})
                new_dict.update(({key: final_dict}))
        return pd.DataFrame(new_dict).T

    def get_meta_models(self):
        # Return all the stacking models
        return list(self.meta_models.keys())

    def get_progress(self, element):
        # Print out training process
        return '[' + \
               '>' * (self.sorted_cols.index(element) + 1) + \
               '-' * (len(self.sorted_cols) - self.sorted_cols.index(element) - 1) + \
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
