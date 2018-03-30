import pandas as pd
import re

from analysis.predictive.settings import DATA_DIRECTORY, DATA_DIRECTORY_FEATURE_ENGINEERED

columns_to_drop = ['race', 'title', 'horse', 'sex_age',
                   'distance', 'run_time', 'breeder',
                   'jockey', 'margin', 'trainer_x', 'trainer_y', 'owner_x', 'owner_y', 'horse_name', 'date_of_birth',
                   'transaction_price', 'prize_obtained', 'race_record', 'highlight_race', 'relatives', 'status',
                   'prize']
dummy_cols = ['place', 'type', 'track', 'weather', 'condition', 'gender', 'breed', 'bracket', 'horse_number',
              'time', 'place_of_birth_jockey', 'place_of_birth_trainer', 'place_of_birth']

race_df = pd.read_csv(DATA_DIRECTORY + 'race.csv', low_memory=False, index_col=0)
horse_df = pd.read_csv(DATA_DIRECTORY + 'horse.csv', low_memory=False, index_col=0)
individual_df = pd.read_csv(DATA_DIRECTORY + 'individual.csv', low_memory=False, index_col=0)
trainer_df = pd.read_csv(DATA_DIRECTORY + 'trainer.csv', low_memory=False, index_col=0)
jockey_df = pd.read_csv(DATA_DIRECTORY + 'jockey.csv', low_memory=False, index_col=0)
horse_df['race_record'] = horse_df['race_record'].apply(lambda x: re.search(r'\[\ (.+)\ \]', x).group(1))
horse_df['race_record'] = horse_df['race_record'].apply(lambda x: list(map(lambda y: int(y), x.split('-'))))


def drop_cols(df):
    # Drop the columns in a dataframe
    for column in columns_to_drop:
        try:
            df.drop(column, axis=1, inplace=True)
        except ValueError:
            continue


def cast_individual_id(id_value):
    # Convert id value to the correct form
    try:
        return int(id_value)
    except ValueError:
        return id_value


def get_rank_series(rank_series, interval=100):
    # Convert rank series to fit as a dummy variable
    def convert_rank(rank, rank_interval):
        if rank == 0:
            return 'No Data'
        else:
            if rank // rank_interval == 0:
                if rank <= 50:
                    return '1-50'
                else:
                    return '51-99'
            else:
                return '%s-%s' % (int(rank // rank_interval * 100), int(rank // rank_interval * 100 + 100 - 1))
    return rank_series.apply(lambda x: convert_rank(x, rank_interval=interval))


def get_dummies_order_by_count(df, column_name):
    # Get dummies by descending count order
    return pd.get_dummies(df[column_name]).reindex(df[column_name].value_counts().index, axis=1).iloc[:, :-1]


def parse_time_stamp(time_string):
    # Parse timestamp expressed in hours
    time_split = time_string.split(':')
    hour = int(time_split[0])
    if hour < 12:
        return '10-12'
    elif hour < 15:
        return '12-15'
    else:
        return '15-after'


def get_trainer_jockey_profile(df, individual):
    # Merge with trainer/jockey dataframe
    assert individual in ['trainer', 'jockey']
    if individual == 'trainer':
        merge_df = trainer_df
    else:
        merge_df = jockey_df
    df = df.merge(merge_df[['%s_id' % individual, 'date_of_birth', 'place_of_birth']], 
                  on='%s_id' % individual, suffixes=['', '_%s' % individual])
    df['run_date'] = df['run_date'].apply(lambda x: pd.Timestamp(x))
    df['date_of_birth'] = df['date_of_birth'].apply(lambda x: pd.Timestamp(x))
    df['%s_age' % individual] = df['run_date'].subtract(df['date_of_birth']).dt.days / 365.0
    df.drop(['date_of_birth'], axis=1, inplace=True)
    top_10_place = df['place_of_birth_%s' % individual].value_counts().index[:10]
    df['place_of_birth_%s' % individual] = df['place_of_birth_%s' %
                                              individual].apply(lambda x: x if x in top_10_place else 'Others')
    return df


def feature_engineer(df, df_name, dummy=True, drop_columns=True):

    try:
        new_df = pd.read_csv(DATA_DIRECTORY_FEATURE_ENGINEERED + '%s.csv' % df_name, low_memory=False, index_col=0)
        new_df['run_date'] = new_df['run_date'].apply(lambda x: pd.Timestamp(x))
        new_df = new_df.set_index(['horse_id', 'run_date'])
        return new_df.drop('finishing_position', axis=1), new_df['finishing_position']
    except FileNotFoundError:
    
        new_df = df.copy()

        # Feature engineering
        has_horse_weight = new_df['horse_weight'].apply(lambda x: bool(re.search(r'(\d+)\(.+\)', x)))
        new_df = new_df[has_horse_weight]
        new_df['horse_weight_increase'] = new_df['horse_weight'].apply(lambda x: re.search(r'\(.?(\d+)\)', x).group(1))
        new_df['horse_weight_increase'] = new_df['horse_weight_increase'].astype(float)
        new_df['horse_weight'] = new_df['horse_weight'].apply(lambda x: re.search(r'(\d+)\(.+\)', x).group(1))
        new_df['horse_weight'] = new_df['horse_weight'].astype(float)

        new_df['time'] = new_df['time'].apply(lambda x: parse_time_stamp(x))

        top_20_place = new_df['place_of_birth'].value_counts().index[:20]
        new_df['place_of_birth'] = new_df['place_of_birth'].apply(lambda x: x if x in top_20_place else 'Others')
        new_df['age_stated'] = new_df['age_int'].astype(float)

        for individual in ['jockey', 'trainer']:
            new_df = get_trainer_jockey_profile(new_df, individual)

        # Get individual ranking information
        target_cols = ['rank', 'first', 'second', 'third', 'out', 'races_major', 'wins_major', 'races_special',
                       'wins_special', 'races_flat', 'wins_flat', 'races_grass', 'wins_grass',
                       'races_dirt', 'wins_dirt', 'wins_percent', 'wins_percent_2nd',
                       'wins_percent_3rd']
        individual_column = ['jockey_id', 'owner_id', 'trainer_id', 'breeder_id']
        eng_jap_dict = {'jockey': u'騎手', 'owner': u'馬主', 'trainer': u'調教師', 'breeder': u'生産者'}

        new_df['year_minus_one'] = new_df['run_date'].apply(lambda x: x.year - 1)
        individual_df['individual_id'] = individual_df['individual_id'].apply(lambda x: cast_individual_id(x))

        for col_name in list(filter(lambda x: 'id' in x, new_df.columns)):
            new_df[col_name] = new_df[col_name].apply(lambda x: cast_individual_id(x))

        for col in individual_column:
            new_df = new_df[new_df[col].isin(individual_df.loc[individual_df['individual_type'] == eng_jap_dict[col.split('_')[0]], 'individual_id'])]

        for col in individual_column:
            filtered = individual_df[individual_df['individual_type'] == eng_jap_dict[col.split('_')[0]]]
            original_cols = list(new_df.columns)
            new_df = new_df.merge(filtered, left_on=[col, 'year_minus_one'], right_on=['individual_id', 'year'],
                                  how='left', suffixes=['', col])
            new_df.fillna(0, inplace=True)
            new_df['rank'] = get_rank_series(new_df['rank'])
            new_df = new_df[original_cols + target_cols]
            new_df.columns = original_cols + list(map(lambda x: x + '_last_year_%s' % col.split('_')[0], target_cols))

            rank_col_name = 'rank_last_year_%s' % col.split('_')[0]
            rank_dummy = pd.get_dummies(new_df[rank_col_name]).iloc[:, :-1]
            rank_dummy.columns = list(map(lambda x: 'ranking-%s-%s' % (col.split('_')[0], x), rank_dummy.columns))

            new_df = new_df.join(rank_dummy)
            new_df.drop(rank_col_name, axis=1, inplace=True)

        # Get dummy columns
        if dummy:
            for cols in dummy_cols:
                new_df = new_df.join(get_dummies_order_by_count(new_df, cols).rename(columns=lambda x: '-'.join([cols, str(x)])))
                try:
                    new_df.drop(cols, axis=1, inplace=True)
                except ValueError:
                    continue

        # Drop some other columns
        columns_to_drop_again = ['corner_position', 'run_time_last_600',
                                 'jockey_id', 'owner_id', 'trainer_id', 'breeder_id',
                                 'parents', 'age_int']
        if drop_columns:
            for cols in columns_to_drop_again:
                try:
                    new_df.drop(cols, axis=1, inplace=True)
                except ValueError:
                    continue

        for object_col in new_df.dtypes[new_df.dtypes == 'object'].index:
            try:
                new_df[object_col] = new_df[object_col].astype(float)
            except ValueError:
                new_df[object_col] = new_df[object_col].apply(lambda x: x if type(x) is int else float(x.replace(',', '')))

        new_df = new_df.sort_values(['horse_id', 'run_date'])
        new_df.to_csv(DATA_DIRECTORY_FEATURE_ENGINEERED + '%s.csv' % df_name, encoding='utf-8')
        new_df = new_df.set_index(['horse_id', 'run_date'])

        return new_df.drop('finishing_position', axis=1), new_df['finishing_position']
