import json
import os

import numpy as np
import pandas as pd


def get_class_weights(y_path, output_data_dir=None):
    """get class weights by y

    Args:
        y_path (str): the path to the y, for example: y_path = "data/2019_join_data_WithLabel_pure_y.csv"
        output_data_dir (str, optional): where to save the output class_weights, the default is the dir same as the y_path
    """
    # read y
    y = pd.read_csv(y_path)

    # count uniques and get class weights
    labels_unique, counts = np.unique(y, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    print(
        "labels unique:%s\t" % labels_unique,
        "class weights:%s" % np.round(class_weights),
    )

    # save setting
    output_data_dir = output_data_dir or os.path.dirname(y_path)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    name = os.path.splitext(os.path.basename(y_path))[0]

    # save class_weights as a json file
    with open(os.path.join(output_data_dir, "%s_class_weights.json" % name), "w") as f:
        json.dump(class_weights, f, indent=4)
    print(
        "output: \n%s"
        % os.path.join(output_data_dir, "%s_class_weights.json" % name)
    )