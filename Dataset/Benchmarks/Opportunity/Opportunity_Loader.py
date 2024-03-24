import os
import logging
import requests
import zipfile

import csv
import numpy as np
from pandas import Series

logger = logging.getLogger(__name__)
# Main function for downloading and processing the Opportunity Datasets
# Returns the train and test sets

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
test_files = ['S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat']
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]


def Opportunity(window_size, step):
    # Build data
    Data = {}
    # Get the current directory path
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'Datasets/Opportunity/Opportunity.npy')
    if os.path.exists(data_path):
        logger.info("Loading preprocessed Opportunity data ...")

        Data_npy = np.load(data_path, allow_pickle=True)
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        Downloader(current_path)
        train_x, test_x, train_y, test_y = generate_data(current_path, label="gestures")
        X_train, y_train = Windowed_majority_labeling(train_x, np.int64(train_y), window_size, step)
        X_test, y_test = Windowed_majority_labeling(test_x, np.int64(test_y), window_size, step)

        logger.info("{} samples will be used for training".format(len(y_train)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = X_train
        Data['train_label'] = y_train
        Data['test_data'] = X_test
        Data['test_label'] = y_test

        if not os.path.exists(current_path +'/Datasets/Opportunity/'):
            os.makedirs(current_path + '/Datasets/Opportunity/')
        np.save(current_path + '/Datasets/Opportunity/Opportunity.npy', Data, allow_pickle=True)
    return Data


def Downloader(current_path):
    # Define the path to check
    path_to_check = os.path.join(current_path, 'Opportunity/OpportunityUCIDataset')
    # Check if the path exists
    if not os.path.exists(path_to_check):
        # URL to download the PAMAP2 from
        file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        # Send a GET request to download the file
        response = requests.get(file_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(path_to_check, exist_ok=True)

            # Save the downloaded file
            file_path = os.path.join(path_to_check, 'PAMAP2_Dataset.zip')
            with open(file_path, 'wb') as file:
                # Track the progress of the download
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024 * 100  # 1KB
                downloaded_size = 0

                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded_size += len(data)

                    # Calculate the download progress percentage
                    progress = (downloaded_size / total_size) * 100

                    # Print the progress message
                    print(f'OpportunityUCIDataset Download in progress: {progress:.2f}%')

            # Extract the contents of the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path_to_check)

            # Remove the downloaded zip file
            os.remove(file_path)

            print('OpportunityUCIDataset Datasets downloaded and extracted successfully.')
        else:
            print('Failed to download the OpportunityUCIDataset please update the file_url')
    else:
        print('OpportunityUCIDataset Datasets Raw file already exists.')
    return


def read_file(current_path):
    data = []
    labels = []
    label_map = [
        (0, 'Other'),
        (406516, 'Open Door 1'),
        (406517, 'Open Door 2'),
        (404516, 'Close Door 1'),
        (404517, 'Close Door 2'),
        (406520, 'Open Fridge'),
        (404520, 'Close Fridge'),
        (406505, 'Open Dishwasher'),
        (404505, 'Close Dishwasher'),
        (406519, 'Open Drawer 1'),
        (404519, 'Close Drawer 1'),
        (406511, 'Open Drawer 2'),
        (404511, 'Close Drawer 2'),
        (406508, 'Open Drawer 3'),
        (404508, 'Close Drawer 3'),
        (408512, 'Clean Table'),
        (407521, 'Drink from Cup'),
        (405506, 'Toggle Switch')
    ]
    labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
    folder_path = os.path.join(current_path, 'Opportunity/OpportunityUCIDataset/OpportunityUCIDataset/dataset')
    cols = [38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            # Read the file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind-1])
                    if sum([x == 'NaN' for x in elem]) < 5:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

    return data, labels


def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #                     included-excluded
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 1:114]
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, 115]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)
    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(current_path, label):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param current_path: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """
    train_x = np.empty((0, NB_SENSOR_CHANNELS))
    train_y = np.empty(0)

    test_x = np.empty((0, NB_SENSOR_CHANNELS))
    test_y = np.empty(0)

    folder_path = os.path.join(current_path, 'Opportunity/OpportunityUCIDataset/OpportunityUCIDataset/dataset')

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            # Read the file
            file_path = os.path.join(folder_path, filename)
            data = np.loadtxt(file_path)
            x, y = process_dataset_file(data, label)
            if filename in test_files:
                test_x = np.vstack((test_x, x))
                test_y = np.concatenate([test_y, y])
            else:
                train_x = np.vstack((train_x, x))
                train_y = np.concatenate([train_y, y])

    return train_x, test_x, train_y, test_y


def Windowed_majority_labeling(values, labels, window_size, step):

    # Initialize empty lists to store windowed samples and labels
    windowed_samples = []
    window_labels = []

    for i in range(0, len(values) - window_size + 1, step):
        # Extract the windowed sample
        windowed_sample = values[i:i + window_size]

        # Assign the majority label to the window
        window_label = np.argmax(np.bincount(labels[i:i + window_size]))

        # Append the windowed sample and label to the lists
        windowed_samples.append(list(windowed_sample))
        window_labels.append(window_label)

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.transpose(np.array(windowed_samples), (0, 2, 1))
    window_labels = np.array(window_labels)
    return windowed_samples, window_labels
