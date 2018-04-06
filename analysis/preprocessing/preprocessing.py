import pandas as pd


def select_return_table(curs, table_name):
    # Select from all records and convert to pandas dataframe
    data = curs.execute('SELECT * FROM %s' % table_name).fetchall()
    column = [element[1] for element in curs.execute('PRAGMA table_info(%s)' % table_name).fetchall()]
    return pd.DataFrame(data, columns=column)


def get_missing_value_perc(df, cond=lambda x: x == 'null'):
    # Check missing value and output percentage
    df_sum = df.applymap(cond).sum()
    df_percentage = df.applymap(cond).sum() / df.applymap(lambda data: data == 'null').count()
    df_percentage = df_percentage.apply(lambda x: '{0:.2f}%'.format(x * 100))
    return pd.concat([df_sum, df_percentage], axis=1, keys=['Missing Value', 'Missing Value (%)'])
