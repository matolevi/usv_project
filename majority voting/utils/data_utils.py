import random


def split_mouse_idx(values, proportions):
    if sum(proportions) != 1:
        raise ValueError("Proportions must sum to 1")

    # Shuffle the list to ensure random distribution
    random.shuffle(values)

    # Calculate the indices for splitting
    split_indices = [
        int(len(values) * sum(proportions[:i])) for i in range(1, len(proportions))
    ]

    # Split the list according to the calculated indices
    splits = [values[i:j] for i, j in zip([0] + split_indices, split_indices + [None])]

    return splits[0], splits[1], splits[2]


def split_dataset(dataset, train_idx, val_idx, test_idx, column="mouse_idx"):
    train_set = dataset[dataset[column].isin(train_idx)]
    val_set = dataset[dataset[column].isin(val_idx)]
    test_set = dataset[dataset[column].isin(test_idx)]
    return train_set, val_set, test_set


def get_train_val_test_dataset(dataset, prop=0.7):
    mouse_idx_propotion = prop
    unique_mouse_ids = dataset["mouse_idx"].unique()
    random.shuffle(unique_mouse_ids)
    train_validation_ids = unique_mouse_ids[
        : int(len(unique_mouse_ids) * mouse_idx_propotion)
    ]
    test_ids = unique_mouse_ids[int(len(unique_mouse_ids) * mouse_idx_propotion) :]
    train_validation_dataset = dataset[dataset["mouse_idx"].isin(train_validation_ids)]
    test_dataset = dataset[dataset["mouse_idx"].isin(test_ids)]
    return train_validation_dataset, test_dataset, train_validation_ids, test_ids
