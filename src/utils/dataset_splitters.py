import torch


def split_into_train_validation_test_sets(full_dataset, train_proportion, validation_proportion, test_proportion):
    train_and_validation_set_count = int(len(full_dataset) * (train_proportion + validation_proportion))
    test_set_count = int(len(full_dataset) * test_proportion)

    train_and_validation_set, test_set = torch.utils.data.random_split(full_dataset, [train_and_validation_set_count
        , test_set_count])

    train_set_count = int(len(train_and_validation_set) * train_proportion / (train_proportion + validation_proportion))
    validation_set_count = int(
        len(train_and_validation_set) * validation_proportion / (train_proportion + validation_proportion))

    train_set, validation_set = torch.utils.data.random_split(train_and_validation_set,
                                                              [train_set_count, validation_set_count])
    return train_set, validation_set, test_set


def split_set_into_examples_and_labels(dataset):
    all_examples = []
    all_labels = []
    for data_tuple in dataset:
        all_examples.append(data_tuple[0])
        all_labels.append(data_tuple[1])

    return all_examples, all_labels
