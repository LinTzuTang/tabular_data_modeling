import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def find_cat_col(X):
    """find the start column of categorical data from sorted dtype X (X = (X_num, X_cat))

    Args:
        X (DataFrame): sorted X, to sort X, execute X[X.dtypes.sort_values().index]

    Returns:
        int: which column of the categorical data starts from
    """
    for i, v in enumerate(X.dtypes.values):
        if v == "object":
            return i
            break
    print("There are no categorical data in X")
    return None


def sort_and_find_cat_col_X(X_path):
    """sort X then find the start col of cat data

    Args:
        X_path (str):  the path to the X, for example: X_path = "data/2019_join_data_WithLabel_pure_X.csv"

    Returns:
        int: which column of the categorical data starts from
    """
    # read X
    X = pd.read_csv(X_path)
    # sort X by dtypes
    X = X[X.dtypes.sort_values().index]
    # find the start col of cat data from sorted dtype X
    cat_col = find_cat_col(X)
    return cat_col


def fill_X_num_nan(X_num, by="median"):
    """fill numeric nan in X_num

    Args:
        X_num (DataFrame): numeric dataframe
        by (str, optional): "median" or "mean", defaults is "median".

    Returns:
        DataFrame: filled dataframe
    """
    assert by in ["median", "mean"]
    if by == "median":
        return X_num.fillna(X_num.median())
    elif by == "mean":
        return X_num.fillna(X_num.mean())


def fill_X_cat_nan(X_cat, by="Unknown"):
    """fill categorical nan in X_cat

    Args:
        X_cat (DataFrame): categorical dataframe
        by (str, optional): "Unknown" or "most_occurring", defaults to "Unknown".

    Returns:
         DataFrame: filled dataframe
    """
    assert by in ["Unknown", "most_occurring"]
    if by == "Unknown":
        return X_cat.fillna("Unknown")
    elif by == "most_occurring":
        return X_cat.apply(lambda x: x.fillna(x.value_counts().index[0]))


def get_cardinalities_from_X(
    X_path, fill_nan_by="Unknown", save=True, output_data_dir=None
):
    """get the cardinalities from a given X

    Args:
        X_path (str):  the path to the X, for example: X_path = "data/2019_join_data_WithLabel_pure_X.csv"
        fill_nan_by (str, optional): "Unknown" or "most_occurring", defaults to "Unknown".
        save (bool, optional): save or not. Defaults to False.
        output_data_dir (str, optional): where to save the output X_cat_cardinalities.json, the default is the dir same as the X_path
    """
    # read X
    X = pd.read_csv(X_path)
    # sort X by dtypes
    X = X[X.dtypes.sort_values().index]
    # find the start col of cat data from sorted dtype X
    cat_col = find_cat_col(X)
    # get categorical X
    X_cat = X.iloc[:, cat_col:]
    # fill nan
    assert fill_nan_by in ["Unknown", "most_occurring"]
    print('fill nan method: %s' % fill_nan_by)
    X_cat = fill_X_cat_nan(X_cat, by=fill_nan_by)
    # get cardinalities
    cardinalities = []
    for col in X_cat:
        cardinalities.append(len(X_cat[col].unique()))
    if save:
        # save setting
        output_data_dir = output_data_dir or os.path.dirname(X_path)
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        name = (
            os.path.splitext(os.path.basename(X_path))[0]
            + "_fill_nan_by_%s" % fill_nan_by
        )

        # save cardinalities as a json file
        with open(
            os.path.join(output_data_dir, "%s_cat_cardinalities.json" % name), "w"
        ) as f:
            json.dump(cardinalities, f, indent=4)
        print(
            'output: \n%s'
            % os.path.join(output_data_dir, "%s_cat_cardinalities.json" % name)
        )
    else:
        return cardinalities


def ordinal_enode(X_cat):
    """ordinal encode the given X_cat

    Args:
        X_cat (DataFrame): categorical data X_cat

    Returns:
        DataFrame: encoded X_cat
    """
    enc = OrdinalEncoder()
    enc.fit(X_cat)
    return enc.transform(X_cat).astype("int64")


def tabular_data_preprocessing(
    X_path, y_path, fill_nan_num="median", fill_nan_cat="Unknown", output_data_dir=None
):
    """tabular data preprocessing(encode then save as nparray)

    Args:
        X_path (str): the path to the X, for example: X_path = "data/2019_join_data_WithLabel_pure_X.csv"
        y_path (str): the path to the y, for example: y_path = "data/2019_join_data_WithLabel_pure_y.csv"
        fill_nan_num (str, optional): "median" or "mean", defaults is "median".
        fill_nan_cat (str, optional): "Unknown" or "most_occurring", defaults to "Unknown".
        output_data_dir (str, optional):  where to save the output X_encoded.npy and y.npy, the default is the dir same as the X_path
    """
    # read X and y
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # sort X by dtypes
    X = X[X.dtypes.sort_values().index]

    # find the start col of cat data from sorted dtype X
    cat_col = find_cat_col(X)

    # split numeric X and categorical X
    X_num = X.iloc[:, :cat_col].astype("float32")
    X_cat = X.iloc[:, cat_col:]

    # fill nan
    X_num = fill_X_num_nan(X_num, by=fill_nan_num)
    X_cat = fill_X_cat_nan(X_cat, by=fill_nan_cat)

    # X_encoded and y to np.array format
    X_cat = ordinal_enode(X_cat)
    X_encoded = np.concatenate((X_num, X_cat), axis=1)
    y = np.array(y.astype("float32"))

    # save setting
    output_data_dir = output_data_dir or os.path.dirname(X_path)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    X_name = os.path.splitext(os.path.basename(X_path))[0]
    y_name = os.path.splitext(os.path.basename(y_path))[0]

    # save X_encoded, y, cat_col
    np.save(os.path.join(output_data_dir, "%s_encoded.npy" % X_name), X_encoded)
    np.save(os.path.join(output_data_dir, "%s.npy" % y_name), y)

    print(
        'output: \n%s'
        % os.path.join(output_data_dir, "%s_encoded.npy" % X_name) + '\n' + os.path.join(output_data_dir, "%s.npy" % y_name)
    )
