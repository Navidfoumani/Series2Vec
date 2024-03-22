import os
import numpy as np
from sklearn import model_selection
import logging

logger = logging.getLogger(__name__)


def load_UCR_dataset(file_name, folder_path):

    train_path = folder_path + '/' + file_name + '/' + file_name + "_TRAIN.tsv"
    test_path = folder_path + '/' + file_name + '/' + file_name + "_TEST.tsv"

    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    _, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    val_data = data[val_indices]
    val_label = label[val_indices]
    return val_data, val_label


for problem in os.listdir('UCR'):
    print(problem)
    Data = {}
    X_train, y_train, X_test, y_test = load_UCR_dataset(file_name=problem, folder_path=os.getcwd()+'/UCR')
    X_val, Data['val_label'] = split_dataset(X_train, y_train, 0.5)
    Data['val_data'] = np.expand_dims(X_val, axis=1)
    max_seq_len = X_train.shape[1]
    Data['max_len'] = max_seq_len
    Data['train_data'] = np.expand_dims(X_train, axis=1)
    Data['train_label'] = y_train
    Data['All_train_data'] = np.expand_dims(X_train, axis=1)
    Data['All_train_label'] = y_train

    Data['test_data'] = np.expand_dims(X_test, axis=1)
    Data['test_label'] = y_test
    np.save(os.getcwd()+'/UCR/' + problem + "/" + problem, Data, allow_pickle=True)
    logger.info("{} samples will be used for training".format(len(Data['train_label'])))


