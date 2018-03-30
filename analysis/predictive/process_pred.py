import os
import pandas as pd
import re

file_directory = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '')) + '\\'
pred_file_directory = '{}predictive\\predictions\\'.format(file_directory)


def get_pred_values(mc, prefix, is_meta, original_df_combined):
    # Get prediction values from models
    min_date = mc.X_train.reset_index()['run_date'].max()
    pred_dict = {}
    for col in mc.sorted_cols:
        file_name = '{}{}{}/{}.csv'.format(pred_file_directory, 'meta_' if is_meta else '', prefix, col)
        print('{}: Reading and transforming prediction file named {}/{}.csv'.format(mc.get_progress(col), prefix, col),
              end='\r', flush=True)
        pred_df = pd.read_csv(file_name)
        pred_df['run_date'] = pred_df['run_date'].apply(lambda x: pd.Timestamp(x))
        pred_df = pred_df[pred_df['run_date'] > min_date]
        pred_df = pred_df.set_index(['horse_id', 'run_date']).join(original_df_combined['place'])
        pred_df = pred_df.reset_index().set_index(['horse_id', 'run_date', 'place'])
        pred_dict[col] = pred_df
    return pred_dict


def store_values(mc, prefix):
    # Store predictions of baseline models
    print('Storing prediction files...')
    for key in sorted(mc.predictions.keys()):
        pred_value = pd.DataFrame(mc.predictions[key], index=mc.X_test.index)
        pred_value.to_csv('{}{}/{}.csv'.format(pred_file_directory, prefix, key))

    # Store trained meta-features of stacking models
    print('Storing meta-model trained features files...')
    for key in mc.meta_models.keys():
        for col in mc.meta_models[key].keys():
            curr_meta = mc.meta_models[key][col]
            curr_index = mc.X_train[mc.X_train.index.isin(mc.y_train[col].dropna().index)].index
            regressors = list(map(lambda x: re.search(r'(\w+)\(', repr(x)).group(1), curr_meta.regr_))
            curr_df = pd.DataFrame(curr_meta.train_meta_features_, index=curr_index, columns=regressors)
            curr_df.to_csv('{}meta_{}/{}.csv'.format(pred_file_directory, prefix, col))
