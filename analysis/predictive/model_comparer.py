import numpy as np
import pandas as pd

from feature_engineering import feature_engineer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class ModelComparer(object):

	# Helper class to incorporate different regression models
    
    run_time_col_name = 'run_time_1000'
    
    def __init__(self, X_df, y_df, original_y_df_dict, **kwargs):

        self.X = feature_engineer(X_df.reset_index())
        self.y = y_df[y_df.index.isin(self.X.index)]
        self.y_original = original_y_df_dict
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, **kwargs)

        self.run_time_series = self.y[run_time_col_name]

        self.model_dict = {}
        self.models = []
        self.train_predictions = {}
        
    def add_model(self, model_method, model_name, **params):
        
        self.model_dict[model_name] = {}
        self.models.append(model_method)
        
        for y_col_name in sorted(self.y.columns, key=lambda x: len(x)):

            model = model_method(**params)
            self.model_dict[model_name]['Model Spec'] = repr(model)
            
            # Uncomment the following lines to observe DV prediction without last run time info
            # if y_col_name not in self.run_time_col_name:
                # X_train.drop('last_run_time', axis=1, inplace=True)
                # X_test.drop('last_run_time', axis=1, inplace=True)
                
            y_train = self.y_train[y_col_name].dropna()
            y_test = self.y_test[y_col_name].dropna()
            X_train = self.X_train[self.X_train.index.isin(y_train.index)]
            X_test = self.X_test[self.X_test.index.isin(y_test.index)]
            if 'normalized' in model_name.lower():
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index)
            
            print('Performing analysis on column %s for model %s (Size: %s)' % (y_col_name, model_name, str(X_train.shape)), end='\r', flush=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if y_col_name not in self.train_predictions.keys():
                self.train_predictions[y_col_name] = {}
            self.train_predictions[y_col_name][model_name] = self.get_transformed(y_col_name, model.predict(X_train), X_train)[0]
            
            if y_col_name != self.run_time_col_name:
                transformed = self.get_transformed(y_col_name, y_pred, X_test)
                y_pred = transformed[0]
                run_time_serie = transformed[1]
                self.model_dict[model_name]['Transformed RMSE: %s (%s)' % (y_col_name, X_train.shape[0])] = '%.6f' % self.get_rmse(y_pred, run_time_serie) 
                self.model_dict[model_name]['Transformed R^2: %s (%s)' % (y_col_name, X_train.shape[0])] = '%.3f' % self.get_r_squared(y_pred, run_time_serie)
            else:
                self.model_dict[model_name]['Original RMSE: %s (%s)' % (y_col_name, X_train.shape[0])] = '%.6f' % self.get_rmse(y_pred, y_test)
                self.model_dict[model_name]['Original R^2: %s (%s)' % (y_col_name, X_train.shape[0])] = '%.3f' % self.get_r_squared(y_pred, y_test)

        print('\n' + ('Added model named %s ' % model_name))

    def get_transformed(self, y_col, pred, df):
        pred = pd.Series(pred, index=df.index)
        run_time = self.run_time_series[self.run_time_series.index.isin(pred.index)]
        if y_col != self.run_time_col_name:
            series = self.y_original[y_col]
            series = series[series.index.isin(pred.index)]
            if 'quo' in y_col:
                pred = pred * series
            elif 'diff' in y_col:
                pred = pred + series
        return (pred, run_time)
        
    def get_report(self, filter_word=''):
        try:
            df = pd.DataFrame.from_dict(self.model_dict)
            return df.loc[list(filter(lambda x: filter_word in x, df.index))].sort_values(df.columns[0])
        except IndexError:
            return
    
    @staticmethod
    def get_rmse(y_true, y_pred):
        diff = np.sum((y_true - y_pred) ** 2)
        return (diff / y_true.shape[0]) ** 1/2
    
    @staticmethod
    def get_r_squared(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
