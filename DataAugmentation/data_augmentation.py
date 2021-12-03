import numpy as np

def transform(X_train,Y_train,Y_columns):
    sub_ids = Y_train[:, Y_columns['subId']]
    unique_sub_ids = list(sub_ids)
    use_sub_ids = unique_sub_ids[0:25]
    use_ids = np.isin(sub_ids,use_sub_ids)
    X_train = X_train[use_ids]
    Y_train = Y_train[use_ids]
    return X_train, Y_train