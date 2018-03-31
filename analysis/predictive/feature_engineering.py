import pandas as pd
import re

from analysis.predictive.settings import DATA_DIRECTORY, DATA_DIRECTORY_FEATURE_ENGINEERED

# Helper variables to be used for feature engineering
columns_to_drop = ['race', 'title', 'horse', 'sex_age', 'distance', 'run_time', 'breeder', 'jockey', 'margin',
                   'trainer_x', 'trainer_y', 'owner_x', 'owner_y', 'horse_name', 'date_of_birth',
                   'transaction_price', 'prize_obtained', 'race_record', 'highlight_race',
                   'relatives', 'status', 'prize']
columns_to_drop_again = ['corner_position', 'run_time_last_600',
                         'jockey_id', 'owner_id', 'trainer_id', 'breeder_id',
                         'parents', 'age_int']
dummy_cols = ['place', 'type', 'track', 'weather', 'condition', 'gender', 'breed', 'bracket', 'horse_number',
              'time', 'place_of_birth_jockey', 'place_of_birth_trainer', 'place_of_birth']
eng_jap_dict = {'jockey': u'騎手', 'owner': u'馬主', 'trainer': u'調教師', 'breeder': u'生産者'}
individual_column = ['jockey_id', 'owner_id', 'trainer_id', 'breeder_id']
target_cols = ['rank', 'first', 'second', 'third', 'out', 'races_major', 'wins_major', 'races_special',
               'wins_special', 'races_flat', 'wins_flat', 'races_grass', 'wins_grass',
               'races_dirt', 'wins_dirt', 'wins_percent', 'wins_percent_2nd', 'wins_percent_3rd']

# Load dataframes for feature engineering and do some data transformation
race_df = pd.read_csv(DATA_DIRECTORY + 'race.csv', low_memory=False, index_col=0)
horse_df = pd.read_csv(DATA_DIRECTORY + 'horse.csv', low_memory=False, index_col=0)
individual_df = pd.read_csv(DATA_DIRECTORY + 'individual.csv', low_memory=False, index_col=0)
trainer_df = pd.read_csv(DATA_DIRECTORY + 'trainer.csv', low_memory=False, index_col=0)
jockey_df = pd.read_csv(DATA_DIRECTORY + 'jockey.csv', low_memory=False, index_col=0)
horse_df['race_record'] = horse_df['race_record'].apply(lambda x: re.search(r'\[\ (.+)\ \]', x).group(1))
horse_df['race_record'] = horse_df['race_record'].apply(lambda x: list(map(lambda y: int(y), x.split('-'))))


def feature_engineer(df, df_name, dummy=True, drop_columns=True):
    # Main function for feature engineering

    def cast_value(value):
        # Helper function to cast value to float type
        return value if type(value) is int else float(value.replace(',', ''))

    try:
        # Load stored filed if possible and do some data type transformation
        new_df = pd.read_csv(DATA_DIRECTORY_FEATURE_ENGINEERED + '%s.csv' % df_name, low_memory=False, index_col=0)
        new_df['run_date'] = new_df['run_date'].apply(lambda x: pd.Timestamp(x))
        new_df = new_df.set_index(['horse_id', 'run_date'])
        return new_df.drop('finishing_position', axis=1), new_df['finishing_position']
    except FileNotFoundError:

        # Get a copy of the dataframe to avoid cascaded alteration
        new_df = df.copy()

        # Feature engineering related to horse and race
        has_horse_weight = new_df['horse_weight'].apply(lambda x: bool(re.search(r'(\d+)\(.+\)', x)))
        new_df = new_df[has_horse_weight]
        new_df['horse_weight_increase'] = new_df['horse_weight'].apply(lambda x: re.search(r'\(.?(\d+)\)', x).group(1))
        new_df['horse_weight_increase'] = new_df['horse_weight_increase'].astype(float)
        new_df['horse_weight'] = new_df['horse_weight'].apply(lambda x: re.search(r'(\d+)\(.+\)', x).group(1))
        new_df['horse_weight'] = new_df['horse_weight'].astype(float)

        # Parse time information
        new_df['time'] = new_df['time'].apply(lambda x: parse_time_stamp(x))

        # Parse horse-related information
        top_20_place = new_df['place_of_birth'].value_counts().index[:20]
        new_df['place_of_birth'] = new_df['place_of_birth'].apply(lambda x: x if x in top_20_place else 'Others')
        new_df['age_stated'] = new_df['age_int'].astype(float)

        # Get jockey and trainer profile information
        for individual in ['jockey', 'trainer']:
            new_df = get_trainer_jockey_profile(new_df, individual)

        # Get individual ranking information
        new_df['year_minus_one'] = new_df['run_date'].apply(lambda x: x.year - 1)
        individual_df['individual_id'] = individual_df['individual_id'].apply(lambda x: cast_individual_id(x))

        # Cast individual id as integer value or keep as an object
        for col_name in list(filter(lambda x: 'id' in x, new_df.columns)):
            new_df[col_name] = new_df[col_name].apply(lambda x: cast_individual_id(x))

        # Filter the dataframe by availability of individual information
        for col in individual_column:
            is_type = individual_df['individual_type'] == eng_jap_dict[col.split('_')[0]]
            individual_id = individual_df.loc[is_type, 'individual_id']
            new_df = new_df[new_df[col].isin(individual_id)]

        # Get individual information
        for col in individual_column:
            # Get individual information from another dataframe and do merging
            filtered = individual_df[individual_df['individual_type'] == eng_jap_dict[col.split('_')[0]]]
            original_cols = list(new_df.columns)
            new_df = new_df.merge(filtered, left_on=[col, 'year_minus_one'], right_on=['individual_id', 'year'],
                                  how='left', suffixes=['', col])

            # Fill in missing values and transform the rank information
            new_df.fillna(0, inplace=True)
            new_df['rank'] = get_rank_series(new_df['rank'])

            # Transform column names for better representation
            new_df = new_df[original_cols + target_cols]
            new_df.columns = original_cols + list(map(lambda x: x + '_last_year_%s' % col.split('_')[0], target_cols))

            # Get dummy variables accordingly
            rank_col_name = 'rank_last_year_%s' % col.split('_')[0]
            rank_dummy = pd.get_dummies(new_df[rank_col_name]).iloc[:, :-1]
            rank_dummy.columns = list(map(lambda x: 'ranking-%s-%s' % (col.split('_')[0], x), rank_dummy.columns))

            # Join and drop original columns
            new_df = new_df.join(rank_dummy)
            new_df.drop(rank_col_name, axis=1, inplace=True)

        # Get dummy columns
        if dummy:
            for cols in dummy_cols:
                # Get count and join accordingly
                dummy_count = get_dummies_order_by_count(new_df, cols)
                dummy_count = dummy_count.rename(columns=lambda x: '-'.join([cols, str(x)]))
                new_df = new_df.join(dummy_count)
                try:
                    new_df.drop(cols, axis=1, inplace=True)
                except ValueError:
                    continue

        # Drop unwanted columns again
        if drop_columns:
            for cols in columns_to_drop_again:
                try:
                    new_df.drop(cols, axis=1, inplace=True)
                except ValueError:
                    continue

        # Deal with rank information to avoid ValueError
        if 'finishing_position' in new_df.columns:
            # Cast all rank info to string value to avoid TypeError
            new_df['finishing_position'] = new_df['finishing_position'].astype(str)

            # Filter out irrelevant ranking information
            new_df = new_df[new_df['finishing_position'].apply(lambda x: bool(re.search(r'\d+', x)))]
            rank_number = new_df['finishing_position'].apply(lambda x: re.search(r'\d+', x).group(0))
            new_df['finishing_position'] = rank_number.astype(float)

        # Cast all object values to float values
        for object_col in new_df.dtypes[new_df.dtypes == 'object'].index:
            try:
                new_df[object_col] = new_df[object_col].astype(float)
            except ValueError:
                new_df[object_col] = new_df[object_col].apply(lambda x: cast_value(x))

        # Sort, store and multi-index the dataframe
        new_df = new_df.sort_values(['horse_id', 'run_date'])
        new_df.to_csv(DATA_DIRECTORY_FEATURE_ENGINEERED + '%s.csv' % df_name, encoding='utf-8')
        new_df = new_df.set_index(['horse_id', 'run_date'])

        return new_df.drop('finishing_position', axis=1), new_df['finishing_position']


def get_dummies_order_by_count(df, column_name):
    # Get dummies by descending count order
    return pd.get_dummies(df[column_name]).reindex(df[column_name].value_counts().index, axis=1).iloc[:, :-1]


def get_rank_series(rank_series, interval=100):
    # Convert rank series to fit as a dummy variable
    def convert_rank(rank, rank_interval):
        # Helper function to convert ranking information
        if rank == 0:
            return 'No Data'
        else:
            # Separate ranks by intervals to convert it to dummy variables later
            if rank // rank_interval == 0:
                if rank <= 50:
                    return '1-50'
                else:
                    return '51-99'
            else:
                return '%s-%s' % (int(rank // rank_interval * 100), int(rank // rank_interval * 100 + 100 - 1))

    # Return the series applied through the helper function
    return rank_series.apply(lambda x: convert_rank(x, rank_interval=interval))


def get_trainer_jockey_profile(df, individual):
    # Merge with trainer/jockey dataframe
    assert individual in ['trainer', 'jockey']

    # Get trainer or jockey dataframe accordingly
    if individual == 'trainer':
        merge_df = trainer_df
    else:
        merge_df = jockey_df

    # Do merging with the target dataframe
    df = df.merge(merge_df[['%s_id' % individual, 'date_of_birth', 'place_of_birth']], 
                  on='%s_id' % individual,
                  suffixes=['', '_%s' % individual])

    # Do some data type casting and drop unwanted columns
    df['run_date'] = df['run_date'].apply(lambda x: pd.Timestamp(x))
    df['date_of_birth'] = df['date_of_birth'].apply(lambda x: pd.Timestamp(x))
    df['%s_age' % individual] = df['run_date'].subtract(df['date_of_birth']).dt.days / 365.0
    df.drop(['date_of_birth'], axis=1, inplace=True)

    # Do feature engineering on place of birth
    top_10_place = df['place_of_birth_%s' % individual].value_counts().index[:10]
    df['place_of_birth_%s' % individual] = df['place_of_birth_%s' %
                                              individual].apply(lambda x: x if x in top_10_place else 'Others')

    # Return the final dataframe
    return df


def parse_time_stamp(time_string):
    # Parse timestamp expressed in hours
    time_split = time_string.split(':')
    hour = int(time_split[0])

    # Convert hour values into time intervals
    if hour < 12:
        return '10-12'
    elif hour < 15:
        return '12-15'
    else:
        return '15-after'


def cast_individual_id(id_value):
    # Convert id value to the correct form
    try:
        return int(id_value)
    except ValueError:
        return id_value


def drop_cols(df):
    # Drop the columns in a dataframe
    for column in columns_to_drop:
        try:
            # Drop column safely and raise ValueError if column does not exist
            df.drop(column, axis=1, inplace=True)
        except ValueError:
            continue
