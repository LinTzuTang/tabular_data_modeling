import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

# scaler methods
SCALER = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
}


# Scalerer
def build_scaler(scaler_type):
    return SCALER[scaler_type]()


def train_val_test_split(
    X_encoded_npy_path,
    y_npy_path,
    scaler_type="standard",
    cat_col=None,
    save=False,
    output_data_dir=None,
):
    """split train, val, test data from given X_encoded.npy and y.npy

    Args:
        X_encoded_npy_path (str): _description_
        y_npy_path (str): _description_
        scaler_type (str, optional): X_num scaler method. Defaults to "standard".
        cat_col (int, optional): which column of the categorical data starts from. Defaults to None. if not sure, use cat_col = sort_and_find_cat_col_X(X_path) to find and check.
        save (bool, optional): save or not. Defaults to False.
        output_data_dir (str, optional): where to save the output X_splited.json and y_splited.json, the default is the dir same as the X_encoded_npy_path

    Returns:
        two dict: return X_splited and y_splited if not save
    """
    # load nparray X and y
    X_encoded = np.load(X_encoded_npy_path)
    y = np.load(y_npy_path)

    # split train test from the whole dataset
    X_splited = {}
    y_splited = {}
    (
        X_splited["train"],
        X_splited["test"],
        y_splited["train"],
        y_splited["test"],
    ) = train_test_split(X_encoded, y, train_size=0.75, stratify=y)

    # split val test from the splited testing dataset
    (
        X_splited["test"],
        X_splited["val"],
        y_splited["test"],
        y_splited["val"],
    ) = train_test_split(
        X_splited["test"], y_splited["test"], train_size=0.5, stratify=y_splited["test"]
    )

    if cat_col == None:
        print("data in encoded_X are numerical, normalize all")
    else:
        print("normalize the first numerical %s columns of encoded_X" % cat_col)
    # Normalization
    preprocess = build_scaler(scaler_type).fit(X_splited["train"][:, :cat_col])
    for k, v in X_splited.items():
        X_splited[k][:, :cat_col] = preprocess.transform(v[:, :cat_col])
    print(
        "Train:%s" % len(X_splited["train"]),
        "Val:%s" % len(X_splited["val"]),
        "Test:%s" % len(X_splited["test"]),
    )

    if save:
        # save setting
        output_data_dir = output_data_dir or os.path.dirname(X_encoded_npy_path)
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        X_name = os.path.splitext(os.path.basename(X_encoded_npy_path))[0]
        y_name = os.path.splitext(os.path.basename(y_npy_path))[0]

        # save class_weights as a json file
        for k, v in X_splited.items():
            X_splited[k] = v.tolist()
        for k, v in y_splited.items():
            y_splited[k] = v.squeeze().tolist()
        with open(os.path.join(output_data_dir, "%s_splited.json" % X_name), "w") as f:
            json.dump(X_splited, f, indent=4)
        with open(os.path.join(output_data_dir, "%s_splited.json" % y_name), "w") as f:
            json.dump(y_splited, f, indent=4)
        print(
            "output: \n%s"
            % os.path.join(output_data_dir, "%s_splited.json" % X_name) + '\n' + os.path.join(output_data_dir, "%s_splited.json" % y_name)
        )

    else:
        return X_splited, y_splited
