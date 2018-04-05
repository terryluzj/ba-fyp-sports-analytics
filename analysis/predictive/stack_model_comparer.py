import pandas as pd
import re

from analysis.predictive.model_comparer import ModelComparer
from analysis.predictive.process_pred import get_pred_values
from analysis.predictive.settings import REPORT_DIRECTORY


class StackModelComparer(ModelComparer):
    # Helper class to incorporate different second-stage regression models

    MODEL_DICT = {
        'ANN': 'MLPRegressor', 'DT': 'DecisionTreeRegressor',
        'GBM': 'GradientBoostingRegressor', 'OLS': 'LinearRegression',
        'RF': 'RandomForestRegressor', 'XGB': 'XGBRegressor'
    }

    def __init__(self, x_df, y_df, original_y_df_dict,
                 original_df, prefix='tuned', **params):
        """
        :param x_df: pandas dataframe of independent variables of the dataset
        :param y_df: pandas dataframe dependent variables of the dataset
        :param original_y_df_dict: dictionary mapping back column values
        :param original_df: the original race dataframe containing place information
        :param prefix: string parameter passed to file directory in the function get_pred_values
        :param params: other parameters for ModelComparer class object
        """
        super().__init__(x_df=x_df, y_df=y_df, original_y_df_dict=original_y_df_dict, **params)

        # Get training and testing meta features
        self.meta_train_X = get_pred_values(prefix=prefix, is_meta=True,
                                            original_df_combined=original_df, join_place=False)
        print()
        self.meta_test_X = get_pred_values(prefix='{}_test'.format(prefix), is_meta=True,
                                           original_df_combined=original_df, join_place=False)

        # Get training and testing dependent column values
        self.meta_train_y = self.y_train
        self.meta_test_y = self.y_test

        # Get mapping series
        self.original_mapped_series = original_y_df_dict

        # Get preliminary RMSE report
        self.rmse_report_tuned = pd.read_csv(REPORT_DIRECTORY + 'rmse_report_tuned.csv', index_col=0)
        renamed = {name: StackModelComparer.rename_model(name) for name in self.rmse_report_tuned.columns}
        reindexed = list(map(lambda x: re.search(r'\:\ (.+)\ \(', x).group(1), self.rmse_report_tuned.index))
        self.rmse_report_tuned.rename(columns=renamed, inplace=True)
        self.rmse_report_tuned.index = reindexed

    def add_model(self, model_method, model_name, transform_then_predict=False, **params):
        # Override original add_model method
        if transform_then_predict:
            model_name = '{} - {}'.format(model_name, 'Mapped Back')
        self.model_dict[model_name] = {}

        for target_column in self.sorted_cols:
            # Get dependent column values
            meta_train_y, meta_test_y, operator, original_mapped_series = self.get_dependent_values(target_column)

            # Get independent column values
            meta_train_x = self.meta_train_X[target_column]
            meta_test_x = self.meta_test_X[target_column]
            meta_train_x, meta_test_x = self.get_normalized_train_test(model_name, meta_train_x, meta_test_x)
            self.print_progress(y_col_name=target_column, model_name=model_name, train_data=meta_train_x)

            # Initiate model and do necessary transformation
            model = model_method(**params)
            self.models.append(model)
            self.model_dict[model_name]['Model Spec'] = repr(model)

            # Get mapped series for train and testing set
            meta_train_x, meta_test_x, meta_train_y, meta_test_y = self.get_transformed_values(meta_train_x,
                                                                                               meta_test_x,
                                                                                               meta_train_y,
                                                                                               meta_test_y,
                                                                                               operator,
                                                                                               original_mapped_series,
                                                                                               transform_then_predict)
            model.fit(meta_train_x.as_matrix(), meta_train_y.as_matrix())

            # Get prediction and true value
            y_true, y_pred = self.get_prediction_true(model, meta_test_x, meta_test_y, operator,
                                                      original_mapped_series, transform_then_predict)

            # Get performance score
            self.update_score(y_true, y_pred, model_name, target_column, meta_train_x)

        print('\n' + ('Added model named %s ' % model_name))

    def forward_search_add_model(self, model_method, model_name, top_perform_column, **params):
        # Forward search algorithm to include the best models only (with prediction before transformation)
        model_name = '{} - FS'.format(model_name)
        self.model_dict[model_name] = {}

        for target_column in self.sorted_cols:
            candidates = self.meta_train_X[self.run_time_col_name].columns
            candidates_len = len(candidates)
            candidates = list(filter(lambda name: name != top_perform_column, candidates))

            # Get dependent column values
            meta_train_y, meta_test_y, operator, original = self.get_dependent_values(target_column)

            # Initiate variables for iterative search
            is_done_forward_search = False
            current_included = [top_perform_column, ]
            current_best_rmse = [float('inf') for idx in range(len(candidates) + 1)]

            # Update first value
            curr_value = self.rmse_report_tuned.loc[target_column][current_included[0]]
            current_best_rmse[0] = curr_value

            while not is_done_forward_search:
                # Initiate model
                model = model_method(**params)
                best_column_name = None

                for candidate in candidates:

                    current_included.append(candidate)

                    # Get independent column value and normalize if necessary
                    meta_train_x = pd.DataFrame(self.meta_train_X[target_column][current_included])
                    meta_test_x = pd.DataFrame(self.meta_test_X[target_column][current_included])
                    meta_train_x, meta_test_x = self.get_normalized_train_test(model_name, meta_train_x, meta_test_x)

                    # Fit the model
                    model.fit(meta_train_x.as_matrix(), meta_train_y.as_matrix())

                    # Get prediction and true value and evaluate RMSE
                    y_true, y_pred = self.get_prediction_true(model, meta_test_x, meta_test_y,
                                                              operator, original, False)
                    rmse = self.get_rmse(y_true, y_pred)
                    if rmse < current_best_rmse[len(current_included)-1]:
                        current_best_rmse[len(current_included)-1] = rmse
                        best_column_name = candidate

                    current_included.remove(candidate)

                # Decide to drop the column or include
                if len(current_included) >= candidates_len:
                    is_done_forward_search = True
                else:
                    larger_rmse = current_best_rmse[len(current_included)] > current_best_rmse[len(current_included)-1]
                    if larger_rmse:
                        is_done_forward_search = True
                    current_included.append(best_column_name)
                    candidates = list(filter(lambda name: name not in current_included, candidates))

            original_or_transformed = 'Original' if target_column == self.run_time_col_name else 'Transformed'
            rmse_col = '{} RMSE: {} ({})'.format(original_or_transformed, target_column,
                                                 self.meta_train_X[target_column].shape[0])
            self.model_dict[model_name][rmse_col] = current_best_rmse[len(current_included) - 1]

    def get_mapped_series(self, target_column):
        # Return the original mapped target column series
        if target_column != self.run_time_col_name:
            original_mapped_series = self.original_mapped_series[target_column].dropna()
        else:
            original_mapped_series = None
        return original_mapped_series

    def update_score(self, y_true, y_pred, model_name, target_column, meta_train_x):
        # Get performance score
        rmse_score = self.get_rmse(y_true, y_pred)
        r_squared_score = self.get_r_squared(y_true, y_pred)
        original_or_transformed = 'Original' if target_column == self.run_time_col_name else 'Transformed'
        rmse_col = '{} RMSE: {} ({})'.format(original_or_transformed, target_column, meta_train_x.shape[0])
        r_squared_col = '{} R^2: {} ({})'.format(original_or_transformed, target_column, meta_train_x.shape[0])
        self.model_dict[model_name][rmse_col] = '%.6f' % rmse_score
        self.model_dict[model_name][r_squared_col] = '%.3f' % r_squared_score
        return

    def get_prediction_true(self, model, meta_test_x, meta_test_y, operator,
                            original_mapped_series, transform_then_predict):
        # Get prediction and true value
        y_pred = pd.Series(model.predict(meta_test_x.as_matrix()), index=meta_test_x.index)
        if transform_then_predict:
            y_true = meta_test_y
        else:
            if original_mapped_series is not None:
                mapped = original_mapped_series[original_mapped_series.index.isin(y_pred.index)]
                y_pred = StackModelComparer.perform_operation(operator, y_pred, mapped)
            y_true = self.run_time_series[self.run_time_series.index.isin(y_pred.index)]
        return y_true, y_pred

    def get_dependent_values(self, target_column):
        # Get dependent variable value and operations
        meta_train_y = self.meta_train_y[target_column].dropna()
        meta_test_y = self.meta_test_y[target_column].dropna()
        operator = self.get_operator(target_column)
        original_mapped_series = self.get_mapped_series(target_column)
        return meta_train_y, meta_test_y, operator, original_mapped_series

    @staticmethod
    def get_transformed_values(meta_train_x, meta_test_x, meta_train_y, meta_test_y,
                               operator, original_mapped_series, transform_then_predict):
        # Get mapped series for train and testing set
        if transform_then_predict and original_mapped_series is not None:
            is_in_train_index = original_mapped_series.index.isin(meta_train_y.index)
            is_in_test_index = original_mapped_series.index.isin(meta_test_y.index)
            train_mapped = original_mapped_series[is_in_train_index]
            test_mapped = original_mapped_series[is_in_test_index]

            # Get transformed X and y
            meta_train_x = StackModelComparer.transform_features(meta_train_x, train_mapped, operator)
            meta_test_x = StackModelComparer.transform_features(meta_test_x, test_mapped, operator)
            meta_train_y = StackModelComparer.perform_operation(operator, meta_train_y, train_mapped)
            meta_test_y = StackModelComparer.perform_operation(operator, meta_test_y, test_mapped)
        return meta_train_x, meta_test_x, meta_train_y, meta_test_y

    @staticmethod
    def transform_features(meta_x, mapped, operator):
        # Transform feature columns by difference or quotient
        meta_x = meta_x.join(mapped, how='left')
        for regressor in meta_x.columns:
            # Apply by columns
            if regressor != mapped.name:
                original = meta_x[mapped.name]
                meta_x[regressor] = StackModelComparer.perform_operation(operator, meta_x[regressor], original)
        meta_x = meta_x.drop(mapped.name, axis=1)
        return meta_x

    @staticmethod
    def perform_operation(operator, untransformed, mapped):
        # Help function to perform transformation of columns by the mapped running time series
        if operator == 'diff':
            return untransformed + mapped
        elif operator == 'quo':
            return untransformed * mapped
        else:
            return untransformed

    @staticmethod
    def rename_model(model_name):
        # Map model name to those in the stacking model
        model_name_initial = model_name.split(' - ')[0]
        try:
            return StackModelComparer.MODEL_DICT[model_name_initial]
        except KeyError:
            return model_name
