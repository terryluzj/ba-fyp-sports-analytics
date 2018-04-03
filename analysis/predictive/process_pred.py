import pandas as pd
import re

from analysis.predictive.model_comparer import ModelComparer
from analysis.predictive.settings import PRED_FILE_DIRECTORY, DEPENDENT_COLUMNS_FEATURED


def get_pred_values(prefix, is_meta, original_df_combined, join_place=True):
    # Get prediction values from models
    pred_dict = {}
    for col in DEPENDENT_COLUMNS_FEATURED:
        file_name = '{}{}{}/{}.csv'.format(PRED_FILE_DIRECTORY, 'meta_' if is_meta else '', prefix, col)
        print('{}: Reading and transforming prediction file named {}/{}.csv'.format(ModelComparer.get_progress(col),
                                                                                    prefix, col),
              end='\r', flush=True)
        pred_df = pd.read_csv(file_name)
        pred_df['run_date'] = pred_df['run_date'].apply(lambda x: pd.Timestamp(x))
        pred_df.sort_values(['horse_id', 'run_date'], inplace=True)
        pred_df.set_index(['horse_id', 'run_date'], inplace=True)
        if join_place:
            pred_df = pred_df.join(original_df_combined['place'])
            pred_df = pred_df.reset_index().set_index(['horse_id', 'run_date', 'place'])
        pred_dict[col] = pred_df
    return pred_dict


def store_values(mc, prefix):
    # Store predictions of baseline models
    print('Storing prediction files...')
    for key in sorted(mc.predictions.keys()):
        pred_value = pd.DataFrame(mc.predictions[key], index=mc.X_test.index)
        pred_value.to_csv('{}{}/{}.csv'.format(PRED_FILE_DIRECTORY, prefix, key))

    # Store trained meta-features of stacking models
    print('Storing meta-model trained features files...')
    for key in mc.meta_models.keys():
        for col in mc.meta_models[key].keys():
            curr_meta = mc.meta_models[key][col]
            regressors = list(map(lambda x: re.search(r'(\w+)\(', repr(x)).group(1), curr_meta.regr_))

            # Get training dataset with meta features
            curr_index_train = mc.X_train[mc.X_train.index.isin(mc.y_train[col].dropna().index)].index
            curr_df_train = pd.DataFrame(curr_meta.train_meta_features_, index=curr_index_train, columns=regressors)
            curr_df_train.to_csv('{}meta_{}/{}.csv'.format(PRED_FILE_DIRECTORY, prefix, col))

            # Get testing dataset with meta features
            curr_index_test = mc.X_test[mc.X_test.index.isin(mc.y_test[col].dropna().index)].index
            curr_df_test = pd.DataFrame(mc.meta_predictions[key][col], index=curr_index_test, columns=regressors)
            curr_df_test.to_csv('{}meta_{}_test/{}.csv'.format(PRED_FILE_DIRECTORY, prefix, col))
