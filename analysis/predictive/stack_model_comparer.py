import pandas as pd

from analysis.predictive.model_comparer import ModelComparer
from analysis.predictive.process_pred import get_pred_values


class StackModelComparer(ModelComparer):
    # Helper class to incorporate different second-stage regression models

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

    def add_model(self, model_method, model_name, transform_then_predict=False, **params):
        # Override original add_model method
        if transform_then_predict:
            model_name = '{} - {}'.format(model_name, 'Mapped Back')
        self.model_dict[model_name] = {}

        for target_column in self.sorted_cols:

            # Get dependent column values
            meta_train_y = self.meta_train_y[target_column].dropna()
            meta_test_y = self.meta_test_y[target_column].dropna()
            operator = self.get_operator(target_column)

            # Get original mapped series of current column name
            if target_column != self.run_time_col_name:
                original_mapped_series = self.original_mapped_series[target_column].dropna()
            else:
                original_mapped_series = None

            # Get independent column values
            meta_train_x = self.meta_train_X[target_column]
            meta_test_x = self.meta_test_X[target_column]
            if 'normalized' in model_name.lower():
                # Normalize the dataset when specified in model name as 'normalized'
                meta_train_x, meta_test_x = self.get_normalized_train_test(meta_train_x, meta_test_x)
            self.print_progress(y_col_name=target_column, model_name=model_name, train_data=meta_train_x)

            # Initiate model and do necessary transformation
            model = model_method(**params)
            self.models.append(model)
            self.model_dict[model_name]['Model Spec'] = repr(model)

            if transform_then_predict and original_mapped_series is not None:
                # Get mapped series for train and testing set
                is_in_train_index = original_mapped_series.index.isin(meta_train_y.index)
                is_in_test_index = original_mapped_series.index.isin(meta_test_y.index)
                train_mapped = original_mapped_series[is_in_train_index]
                test_mapped = original_mapped_series[is_in_test_index]

                # Get transformed X and y
                meta_train_x = self.transform_features(meta_train_x, train_mapped, operator)
                meta_test_x = self.transform_features(meta_test_x, test_mapped, operator)
                meta_train_y = self.perform_operation(operator, meta_train_y, train_mapped)
                meta_test_y = self.perform_operation(operator, meta_test_y, test_mapped)

            # Fit the model
            model.fit(meta_train_x, meta_train_y)
            y_pred = pd.Series(model.predict(meta_test_x), index=meta_test_x.index)
            if transform_then_predict:
                y_true = meta_test_y
            else:
                if original_mapped_series is not None:
                    mapped = original_mapped_series[original_mapped_series.index.isin(y_pred.index)]
                    y_pred = self.perform_operation(operator, y_pred, mapped)
                y_true = self.run_time_series[self.run_time_series.index.isin(y_pred.index)]

            # Get performance score
            rmse_score = self.get_rmse(y_true, y_pred)
            r_squared_score = self.get_r_squared(y_true, y_pred)
            original_or_transformed = 'Original' if target_column == self.run_time_col_name else 'Transformed'
            rmse_col = '{} RMSE: {} ({})'.format(original_or_transformed, target_column, meta_train_x.shape[0])
            r_squared_col = '{} R^2: {} ({})'.format(original_or_transformed, target_column, meta_train_x.shape[0])
            self.model_dict[model_name][rmse_col] = '%.6f' % rmse_score
            self.model_dict[model_name][r_squared_col] = '%.3f' % r_squared_score

        print('\n' + ('Added model named %s ' % model_name))

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
        assert operator in ['diff', 'quo']
        if operator == 'diff':
            return untransformed + mapped
        elif operator == 'quo':
            return untransformed * mapped
        return untransformed
