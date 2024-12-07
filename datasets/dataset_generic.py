import numpy as np
import pandas as pd


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    """
    Save the segmented dataset to a CSV file
    :param split_datasets:
    :param column_keys:
    :param filename:
    :param boolean_style:
    :return:
    """
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()
