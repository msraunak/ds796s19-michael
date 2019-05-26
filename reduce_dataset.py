import pandas as pd

class map_dict(dict):
    '''Subclass of dict to return value rather than NaN for values not in dictionary provided to map function.'''
    def __missing__(self, key):
        return key

if __name__ == "__main__":
    df = pd.read_pickle("ids_df_formatted.pickle")

    # drop rows with missing values
    temp_series = df.isna().sum(axis=1)
    indexes_to_drop = temp_series[temp_series != 0].index
    print("Number of rows with missing values that were dropped: {}".format(len(indexes_to_drop)))
    print("\nRemoved Data\n" + "-" * 30 + "\n{}".format(df.loc[indexes_to_drop, "label"].value_counts()))
    df.drop(indexes_to_drop, axis=0, inplace=True)

    # remove rows with any value less than 0
    temp_df = df.copy()
    for col in temp_df.select_dtypes(include=['int','float']).columns:
        temp_df[col] = temp_df[col].apply(lambda x: x < 0)
    temp_df = temp_df.sum(axis=1)
    indexes_to_drop = temp_df[temp_df > 0].index
    print("\n\nNumber of rows with any negative values that were dropped: {}".format(len(indexes_to_drop)))
    print("\nRemoved Data\n" + "-" * 30 + "\n{}".format(df.loc[indexes_to_drop, "label"].value_counts()))
    df.drop(indexes_to_drop, axis=0, inplace=True)

    # deal with infinity values
    df_float_cols = df.select_dtypes(include=['float']).columns
    for col in df_float_cols:
        vals = list(filter(lambda x: x != float('inf'), df[col].values))
        # get largest value in series other than inf
        max_val = sorted(vals)[-1]
        # replace infinity values with maximum value of the column
        df[col] = df[col].map(map_dict({float('inf'): max_val}))

    # reset index
    df.reset_index(drop=True, inplace=True)
    # pickle dataframe for reuse
    df.to_pickle("ids_df_formatted_and_reduced.pickle")