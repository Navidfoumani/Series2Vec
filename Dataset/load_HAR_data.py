import os
import numpy as np
import logging
import torch
from tslearn.clustering import TimeSeriesKMeans

logger = logging.getLogger(__name__)


def load(config):
    # Build data
    Data = {}
    problem = config['data_dir'].split('/')[-1]

    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + problem + '.npy', allow_pickle=True)

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['All_train_data'] = Data_npy.item().get('All_train_data')
        Data['All_train_label'] = Data_npy.item().get('All_train_label')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')
        # Data['All_train_Cluster'] = Data_npy.item().get('All_train_Cluster')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        logger.info("Loading and preprocessing data ...")
        train_dataset = torch.load(os.path.join(config['data_dir'], "train.pt"))
        valid_dataset = torch.load(os.path.join(config['data_dir'], "val.pt"))
        test_dataset = torch.load(os.path.join(config['data_dir'], "test.pt"))

        Data['max_len'] = train_dataset['samples'].shape[-1]
        Data['All_train_data'] = train_dataset['samples'].numpy()
        Data['All_train_label'] = train_dataset['labels'].numpy()
        Data['train_data'] = train_dataset['samples'].numpy()
        Data['train_label'] = train_dataset['labels'].numpy()
        Data['val_data'] = valid_dataset['samples'].numpy()
        Data['val_label'] = valid_dataset['labels'].numpy()
        Data['test_data'] = test_dataset['samples'].numpy()
        Data['test_label'] = test_dataset['labels'].numpy()

        # Data['All_train_Cluster'] = k_means_clustering(Data['All_train_data'])

        np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)
    return Data


def k_means_clustering(Data):
    sdtw_km = TimeSeriesKMeans(n_clusters=12, metric="softdtw", metric_params={"gamma": .01},
                               max_iter=10, verbose=True)
    y_pred = sdtw_km.fit_predict(Data)
    return y_pred