import os
import logging
import requests
import zipfile
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# Main function for downloading and processing the Skoda Datasets
# Returns the train and test sets


def Skoda(window_size, step):
    # Build data
    Data = {}
    # Get the current directory path
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'Datasets/Skoda/Skoda.npy')
    if os.path.exists(data_path):
        logger.info("Loading preprocessed Skoda data ...")

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

        if not os.path.exists(current_path + '/Datasets/Skoda/'):
            os.makedirs(current_path + '/Datasets/Skoda/')
        np.save(current_path + '/Datasets/Skoda/Skoda.npy', Data, allow_pickle=True)

    return Data


def Downloader(current_path):
    # Define the path to check
    path_to_check = os.path.join(current_path, 'Skoda/SkodaMiniCP_2015_08')
    # Check if the path exists
    if not os.path.exists(path_to_check):
        # URL to download the PAMAP2 from
        file_url = 'http://har-dataset.org/lib/exe/fetch.php?media=wiki:dataset:skodaminicp:skodaminicp_2015_08.zip'
        # Send a GET request to download the file
        response = requests.get(file_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(path_to_check, exist_ok=True)

            # Save the downloaded file
            file_path = os.path.join(path_to_check, 'skodaminicp_2015_08.zip')
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
                    print(f'Skoda Download in progress: {progress:.2f}%')

            # Extract the contents of the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path_to_check)

            # Remove the downloaded zip file
            os.remove(file_path)

            print('Skoda.zip Datasets downloaded and extracted successfully.')
        else:
            print('Failed to download the Skoda.zip please update the file_url')
    else:
        print('Skoda Datasets Raw file already exists')
    return


def generate_data(current_path):
    directory = current_path + '/Skoda/SkodaMiniCP_2015_08/SkodaMiniCP_2015_08/right_classall_clean.mat'
    data_dict = sio.loadmat(directory, squeeze_me=True)
    all_data = data_dict[list(data_dict.keys())[3]]
    x_train, y_train, x_test, y_test = get_train_val_test(all_data)
    return x_train, y_train, x_test, y_test


def get_train_val_test(data):
    # removing sensor ids
    for i in range(1, 60, 6):
        data = np.delete(data, i, 1)

    # data = data[data[:, 0] != 32]  # remove null class activity

    data = label_count_from_zero(data)
    data = normalize(data)

    activity_id = np.unique(data[:, 0])
    number_of_activity = len(activity_id)

    for i in range(number_of_activity):

        data_for_a_single_activity = data[np.where(data[:, 0] == activity_id[i])]
        trainx, trainy, testx, testy = split(data_for_a_single_activity)

        if i == 0:
            x_train, y_train, x_test, y_test = trainx, trainy, testx, testy

        else:
            x_train = np.concatenate((x_train, trainx))
            y_train = np.concatenate((y_train, trainy))

            x_test = np.concatenate((x_test, testx))
            y_test = np.concatenate((y_test, testy))

    return x_train, y_train, x_test, y_test


def standardize(mat):
    """ standardize each sensor data columnwise"""
    for i in range(mat.shape[1]):
        mean = np.mean(mat[:, [i]])
        std = np.std(mat[:, [i]])
        mat[:, [i]] -= mean
        mat[:, [i]] /= std

    return mat


def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)


def label_count_from_zero(all_data):
    """ start all labels from 0 to total number of activities"""

    labels = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
              51: 'check gaps front door', 52: 'open left front door',
              53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
              56: 'open/close trunk', 57: 'check steering wheel'}

    # new_pi_plot(all_data, labels)
    a = np.unique(all_data[:, 0])

    for i in range(len(a)):
        all_data[:, 0][all_data[:, 0] == a[i]] = i
    #         print(i, labels[a[i]])

    return all_data


def split(data):
    """ get 80% train, 20% test and 10% validation data from each activity """

    y = data[:, 0]  # .reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, y_train, X_test, y_test


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


def new_pi_plot(data, map):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
    labels, sizes = np.unique(data[:, 0], return_counts=True)
    sorted_indices = np.argsort(-sizes)
    labels = labels[sorted_indices]
    sizes = sizes[sorted_indices]
    # Create the pie plot
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.5), startangle=0)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-", linewidth=3),
              bbox=bbox_props, zorder=0, va="center")
    # Calculate total size
    total = sum(sizes)

    for i, (p, label, size) in enumerate(zip(wedges, labels, sizes)):
        percentage = 100 * size / total
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(f'{percentage:.1f}%: {map[label]}', xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)
    ax.set_title("Label Distribution")
    # Save the figure in EPS format
    plt.savefig('Skoda.eps', format='eps')
    plt.show()
