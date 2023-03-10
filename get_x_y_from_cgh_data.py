import os

import pandas as pd


def get_X_y_from_cgh_data(cgh_data_path, output_data_dir=None):
    """get features data (X) and labels (y) from the cathay general hospital dataset

    Args:
        cgh_data_path (str): the path to the cgh_data, for example: cgh_data_path = 'data/2019_join_data_WithLabel_pure.csv'
        output_data_dir (str, optional): where to save the output datasets (X and y), the default is the dir same as the cgh_data_path
    """
    # read data
    df = pd.read_csv(cgh_data_path)

    # get X, y
    X = df.iloc[:, 4:]
    y = df["SEPSIS"]

    # save setting
    output_data_dir = output_data_dir or os.path.dirname(cgh_data_path)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    name = os.path.splitext(os.path.basename(cgh_data_path))[0]

    # save X, y
    X.to_csv(os.path.join(output_data_dir, "%s_X.csv" % name), index=None)
    y.to_csv(os.path.join(output_data_dir, "%s_y.csv" % name), index=None)
    print(
        "output: \n%s"
            % os.path.join(output_data_dir, "%s_X.csv" % name)+'\n'+os.path.join(output_data_dir, "%s_y.csv" % name)
    )