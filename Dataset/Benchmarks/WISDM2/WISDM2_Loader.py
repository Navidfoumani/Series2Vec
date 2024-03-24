import os
import logging
import requests
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
# Main function for downloading and processing the WISDM Datasets
# Returns the train and test sets


def WISDM2(window_size, step):
    # Build data
    Data = {}
    # Get the current directory path
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'Datasets/WISDM2/WISDM2.npy')
    if os.path.exists(data_path):
        logger.info("Loading preprocessed WISDM2 data ...")

        Data_npy = np.load(data_path, allow_pickle=True)
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        Downloader(current_path)
        train_x, train_y, test_x, test_y = generate_data(current_path)

        X_train, y_train = Windowed_majority_labeling(train_x, np.int64(train_y), window_size, step)
        X_test, y_test = Windowed_majority_labeling(test_x, np.int64(test_y), window_size, step)

        logger.info("{} samples will be used for training".format(len(y_train)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = X_train
        Data['train_label'] = y_train
        Data['test_data'] = X_test
        Data['test_label'] = y_test

        if not os.path.exists(current_path + '/Datasets/WISDM2/'):
            os.makedirs(current_path + '/Datasets/WISDM2/')
        np.save(current_path + '/Datasets/WISDM2/WISDM2.npy', Data, allow_pickle=True)

    return Data


def Downloader(current_path):
    # Define the path to check
    path_to_check = os.path.join(current_path, 'WISDM2/WISDM2_at_latest')
    # Check if the path exists
    if not os.path.exists(path_to_check):
        # URL to download the PAMAP2 from
        file_url = 'https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_at_latest.tar.gz'
        # Send a GET request to download the file
        response = requests.get(file_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(path_to_check, exist_ok=True)

            # Save the downloaded file
            file_path = os.path.join(path_to_check, 'WISDM_at_latest.tar.gz')
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
                    print(f'WISDM Download in progress: {progress:.2f}%')

            # Extract the contents of the zip file
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(path_to_check)

            # Remove the downloaded zip file
            os.remove(file_path)

            print('WISDM2.tar.gz Datasets downloaded and extracted successfully.')
        else:
            print('Failed to download the WISDM2.tar.gz please update the file_url')
    else:
        print('WISDM2 Datasets Raw file already exists')
    return


def generate_data(current_path):
    directory = current_path + '/WISDM2/WISDM2_at_latest/home/share/data/public_sets/WISDM_at_v2.0/WISDM_at_v2.0_raw.txt'

    file = open(directory)
    lines = file.readlines()

    processedList = []

    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                break
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            processedList.append(temp)
        except:
            print('Error at line number: ', i)
    columns = ['series', 'label', 'timestamp', 'x', 'y', 'z']
    data = pd.DataFrame(data=processedList, columns=columns)
    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')
    x_train, y_train, x_test, y_test = load_activity(data)
    return x_train, y_train, x_test, y_test


def load_activity(df):
    norm = True
    verbose = 1
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['label'])
    all_series = df.series.unique()
    train_series, test_series = train_test_split([x for x in range(len(all_series))], test_size=22, random_state=1)
    train_series = all_series[train_series]
    test_series = all_series[test_series]

    train_data = np.empty((0, 3))
    train_label = np.empty(0)
    print("[Data_Loader] Loading Train Data")
    for series in train_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        train_data = np.vstack((train_data, series_data))
        train_label = np.hstack((train_label, series_labels))

    test_data = np.empty((0, 3))
    test_label = np.empty(0)
    print("[Data_Loader] Loading Test Data")
    for series in test_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        test_data = np.vstack((test_data, series_data))
        test_label = np.hstack((test_label, series_labels))
    return train_data, train_label, test_data, test_label


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