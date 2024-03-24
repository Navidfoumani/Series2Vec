import os
import logging
import requests
import zipfile
import numpy as np
import pandas as pd
import scipy.io


logger = logging.getLogger(__name__)
# Main function for downloading and processing the USC_HAD Datasets
# Returns the train and test sets
test_subject = {'13', '14'}
feature_column = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
def USC_HAD(window_size, step):
        # Build data
        Data = {}
        # Get the current directory path
        current_path = os.getcwd()
        data_path = os.path.join(current_path, 'Datasets/USC_HAD/USC_HAD.npy')
        if os.path.exists(data_path):
            logger.info("Loading preprocessed USC_HAD data ...")

            Data_npy = np.load(data_path, allow_pickle=True)
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')

            logger.info("{} samples will be used for training".format(len(Data['train_label'])))
            logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

        else:
            Downloader(current_path)
            train_x, test_x, train_y, test_y = generate_data(current_path)
            X_train, y_train = Windowed_majority_labeling(train_x, train_y, window_size, step)
            X_test, y_test = Windowed_majority_labeling(test_x, test_y, window_size, step)

            logger.info("{} samples will be used for training".format(len(y_train)))
            logger.info("{} samples will be used for testing".format(len(y_test)))

            Data['train_data'] = X_train
            Data['train_label'] = y_train
            Data['test_data'] = X_test
            Data['test_label'] = y_test

            if not os.path.exists(current_path + '/Datasets/USC_HAD/'):
                os.makedirs(current_path + '/Datasets/USC_HAD/')
            np.save(current_path + '/Datasets/USC_HAD/USC_HAD.npy', Data, allow_pickle=True)
        return Data


def Downloader(current_path):
    # Define the path to check
    path_to_check = os.path.join(current_path, 'USC_HAD/USC-HAD')
    # Check if the path exists
    if not os.path.exists(path_to_check):
        # URL to download the PAMAP2 from
        file_url = 'http://sipi.usc.edu/had/USC-HAD.zip'
        # Send a GET request to download the file
        response = requests.get(file_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(path_to_check, exist_ok=True)

            # Save the downloaded file
            file_path = os.path.join(path_to_check, 'USC-HAD.zip')
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
                    print(f'USC-HAD  Download in progress: {progress:.2f}%')

            # Extract the contents of the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path_to_check)

            # Remove the downloaded zip file
            os.remove(file_path)

            print('USC-HAD.zip Datasets downloaded and extracted successfully.')
        else:
            print('Failed to download the USC-HAD.zip please update the file_url')
    else:
        print('USC-HAD Datasets Raw file already exists')
    return


def generate_data(current_path):

    subject, act_num, sensor_readings = read_dir(current_path+'/USC_HAD/USC-HAD')

    acc_x = []
    acc_y = []
    acc_z = []
    gyr_x = []
    gyr_y = []
    gyr_z = []

    act_label = []
    subject_id = []

    for i in range(len(subject)):
        for j in sensor_readings[i]:
            acc_x.append(j[0])  # acc_x
            acc_y.append(j[1])  # acc_y
            acc_z.append(j[2])  # acc_z
            gyr_x.append(j[3])  # gyr_x
            gyr_y.append(j[4])  # gyr_y
            gyr_z.append(j[5])  # gyr_z
            act_label.append(act_num[i])
            subject_id.append(subject[i])

    df = pd.DataFrame({'subject': subject_id, 'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
                       'gyr_x': gyr_x, 'gyr_y': gyr_y, 'gyr_z': gyr_z, 'activity': act_label})
    df = df[['subject', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'activity']]
    df['activity'] = df['activity'].astype(int)
    # new_pi_plot(df)
    test_df = df.loc[df['subject'].isin(test_subject)]
    test_x = test_df[feature_column].values
    test_y = test_df['activity'].values

    train_df = df.loc[~df['subject'].isin(test_subject)]
    train_x = train_df[feature_column].values
    train_y = train_df['activity'].values

    return train_x, test_x, train_y, test_y


def read_dir(directory):
    subject = []
    act_num = []
    sensor_readings = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(path, name))
                subject.extend(mat['subject'])
                sensor_readings.append(mat['sensor_readings'])

                if mat.get('activity_number') is None:
                    act_num.append('11')
                else:
                    act_num.append(mat['activity_number'])
    return subject, act_num, sensor_readings


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
    windowed_samples =np.transpose(np.array(windowed_samples), (0, 2, 1))
    window_labels = np.array(window_labels)
    return windowed_samples, window_labels


def load_activity_map():
    map = {}
    map[1] = 'walking forward'
    map[2] = 'walking left'
    map[3] = 'walking right'
    map[4] = 'walking upstairs'
    map[5] = 'walking downstairs'
    map[6] = 'running forward '
    map[7] = 'jumping'
    map[8] = 'sitting'
    map[9] = 'standing'
    map[10] = 'sleeping'
    map[11] = 'elevator up '
    map[12] = 'elevator down'
    return map


def new_pi_plot(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))
    labels, sizes = np.unique(data['activity'], return_counts=True)
    sorted_indices = np.argsort(-sizes)
    labels = labels[sorted_indices]
    sizes = sizes[sorted_indices]
    map = load_activity_map()
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
    plt.savefig('USC_HAD.eps', format='eps')
    plt.show()