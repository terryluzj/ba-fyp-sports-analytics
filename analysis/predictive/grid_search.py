import numpy as np
import os
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

file_directory = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '')) + '\\'
report_directory = file_directory + 'predictive/report/'


def get_best_model(model_comparer, name_saved, model_method, tuned_params, scoring, randomized=False, n_iter=50,
                   filter_func=None, verbose=2, n_jobs=1, **params):
    # Get the best model from grid search
    try:
        best_df = pd.read_csv(report_directory + '%s_grid_search_report.csv' % name_saved, index_col=0)
    except FileNotFoundError:
        best_df = start_grid_search(model_comparer=model_comparer, model_method=model_method,
                                    tuned_params=tuned_params, n_jobs=n_jobs, randomized=randomized, n_iter=n_iter,
                                    filter_func=filter_func, scoring=scoring, name_to_save=name_saved, verbose=verbose,
                                    **params)
    return report_best_params(best_df, scoring=scoring)


def start_grid_search(model_comparer, model_method, tuned_params, name_to_save, n_iter,
                      scoring, n_jobs, filter_func, randomized,
                      verbose, cv=5, **params):
    # General method to trigger grid search
    best_dict = {}
    column_names = model_comparer.y_train.columns if filter_func is None else list(filter(filter_func,
                                                                                          model_comparer.y_train.columns))
    for col_name in column_names:
        if not randomized:
            grid_search = GridSearchCV(model_method(**params), tuned_params, cv=cv,
                                       scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        else:
            grid_search = RandomizedSearchCV(model_method(**params), tuned_params, cv=cv, n_iter=n_iter,
                                             scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        y = model_comparer.y[col_name].dropna()
        X = model_comparer.X[model_comparer.X.index.isin(y.index)]
        grid_search.fit(np.array(X), np.array(y))
        best_dict[col_name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_scores': grid_search.best_score_
        }
    best_df = pd.DataFrame(best_dict)
    file_name = '%s_grid_search_report.csv' % name_to_save
    print('Report file saved as %s' % file_name)
    best_df.to_csv(report_directory + file_name)
    return best_df


def report_best_params(df, scoring):
    # Get the best reported parameters
    best_dict = {}
    for column in df.columns:
        if type(df[column]['best_params']) is str:
            best_params = eval(df[column]['best_params'])
        else:
            best_params = df[column]['best_params']
        best_params.update({'score': float(df[column]['best_scores'])})
        best_dict[column] = best_params
    df_return = pd.DataFrame(best_dict).T
    if 'False' in repr(scoring):
        df_return['score'] = -df_return['score']
    return df_return
