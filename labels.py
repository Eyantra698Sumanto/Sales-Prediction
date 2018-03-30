# Assigning the labels to the unique values of different columns in train dataset
def assign_labels(series_list: list) -> dict:
    labels = {}
    for i in range(len(series_list)):
        if series_list[i] not in labels:
            labels[series_list[i]] = i + 1
    return labels


# Changing the values in the DataFrame by previously calculated labels
def update_dataframe(data, column_name:  str, labels_dict:  dict):
    for key, value in labels_dict.items():
        data[column_name].replace(key, value, inplace=True)


# Assigning the labels to the unique values of different columns in test dataset
def assign_labels_test(train_cols_list: list, test_cols_list: list, labels: dict) -> dict:
    notIn = []
    for i in test_cols_list:
        if i not in train_cols_list:
            notIn.append(i)
    length = len(labels)
    for i in range(len(notIn)):
        labels[notIn[i]] = length + i + 1

    return labels
