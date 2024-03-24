import os
import logging
import requests
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# Main function for downloading and processing the PAMAP2 Datasets
# Returns the train and test sets


def PAMAP2(window_size, step):
    # Build data
    Data = {}
    # Get the current directory path
    current_path = os.getcwd()

    data_path = os.path.join(current_path, 'Datasets/PAMAP2/PAMAP2.npy')
    if os.path.exists(data_path):
        logger.info("Loading preprocessed PAMAP2 data ...")

        Data_npy = np.load(data_path, allow_pickle=True)
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        Downloader(current_path)
        data = load_subjects('PAMAP2/PAMAP2_Dataset')
        data = fix_data(data)  # Data Cleaning
        new_pi_plot(data)
        # Pi_plot(data)
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        # Fit LabelEncoder on the original labels and transform them
        data['activity_id'] = label_encoder.fit_transform(data['activity_id'])
        X_train, X_test, y_train, y_test = split_train_test(data)
        X_train, y_train = Windowed_majority_labeling(X_train, y_train, window_size, step)
        X_test, y_test = Windowed_majority_labeling(X_test, y_test, window_size, step)

        logger.info("{} samples will be used for training".format(len(y_train)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = X_train
        Data['train_label'] = y_train
        Data['test_data'] = X_test
        Data['test_label'] = y_test
        if not os.path.exists(current_path + '/Datasets/PAMAP2/'):
            os.makedirs(current_path + '/Datasets/PAMAP2/')
        np.save(current_path + '/Datasets/PAMAP2/PAMAP2.npy', Data, allow_pickle=True)

    return Data


def Downloader(current_path):
    # Define the path to check
    path_to_check = os.path.join(current_path, 'PAMAP2/PAMAP2_Dataset')
    # Check if the path exists
    if not os.path.exists(path_to_check):
        # URL to download the PAMAP2 from
        file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip'
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
                    print(f'PAMAP2 Download in progress: {progress:.2f}%')

            # Extract the contents of the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path_to_check)

            # Remove the downloaded zip file
            os.remove(file_path)

            print('PAMAP2 Datasets downloaded and extracted successfully.')
        else:
            print('Failed to download the PAMAP2 please update the file_url')
    else:
        print('PAMAP2 Datasets Raw file already exists.')
    return


def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map


def generate_three_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    return [x, y, z]


def generate_four_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    w = name + '_w'
    return [x, y, z, w]


def generate_cols_IMU(name):
    # temp
    temp = name + '_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name + '_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name + '_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name + '_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name + '_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name + '_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output


def load_IMU():
    output = ['time_stamp', 'activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output


def load_subjects(directory):
    output = pd.DataFrame()
    cols = load_IMU()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                file_path = os.path.join(root, file)
                subject = pd.read_table(file_path, header=None, sep='\s+')
                subject.columns = cols
                subject['id'] = int(file[-5])
                output = pd.concat([output, subject], ignore_index=True)
    output.reset_index(drop=True, inplace=True)

    return output


def fix_data(data):
    """
    As we can see, there are NaN values in our dataset.
    To handle this, we will replace each NaN value with the mean value of its respective column.
    Additionally, we observed from the map that activity_id = 0 is not a valid activity.
    """
    data = data.drop(data[data['activity_id'] == 0].index)
    data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())

    # Count the number of unique subjects for each activity
    activity_subject_counts = data.groupby('activity_id')['id'].nunique()
    # Filter activities with less than 6 subjects
    activities_to_drop = activity_subject_counts[activity_subject_counts < 6].index

    # Drop rows corresponding to activities with less than 6 subjects
    data_filtered = data[~data['activity_id'].isin(activities_to_drop)]

    print('Size of the data: ', data_filtered.size)
    print('Shape of the data: ', data_filtered.shape)
    print('Number of columns in the data: ', len(data_filtered.columns))
    result_id = data_filtered.groupby(['id']).mean().reset_index()
    print('Number of uniqe ids in the data: ', len(result_id))
    result_act = data_filtered.groupby(['activity_id']).mean().reset_index()
    print('Numbe of uniqe activitys in the data: ', len(result_act))

    return data_filtered


def split_train_test(data):
    # create the test data
    subject107 = data[data['id'] == 8]
    subject108 = data[data['id'] == 9]
    test = pd.concat([subject107, subject108], ignore_index=True)

    # create the train data
    train = data[data['id'] != 8]
    train = train[train['id'] != 9]

    # drop the columns id and time
    test = test.drop(["id"], axis=1)
    train = train.drop(["id"], axis=1)

    # split train and test to X and y
    X_train = train.drop(['activity_id', 'time_stamp'], axis=1).values
    X_test = test.drop(['activity_id', 'time_stamp'], axis=1).values

    # make data scale to min max beetwin 0 to 1
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    min_max_scaler.fit(X_test)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    y_train = train['activity_id'].values
    y_test = test['activity_id'].values

    print('Train shape X :', X_train.shape, ' y ', y_train.shape)
    print('Test shape X :', X_test.shape, ' y ', y_test.shape)
    return X_train, X_test, y_train, y_test


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


def Pi_plot(data):
    # Count the frequency of each label
    labels, sizes = np.unique(data['activity_id'], return_counts=True)
    sorted_indices = np.argsort(-sizes)
    labels = labels[sorted_indices]
    sizes = sizes[sorted_indices]
    map = load_activity_map()
    # Create the pie plot
    wedges, texts, autotexts = plt.pie(sizes, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})

    # Adjust layout and aspect ratio
    plt.axis('equal')
    plt.tight_layout()

    # Calculate total size
    total = sum(sizes)

    # Add annotations
    annotations = []
    for i, (wedge, label, size) in enumerate(zip(wedges, labels, sizes)):
        percentage = 100 * size / total
        annotation_text = f'{percentage:.1f}%: {map[label]}'
        annotations.append(annotation_text)
        plt.setp(wedge, edgecolor='white')

    # Display annotations on the right side
    plt.legend(wedges, annotations, loc='center left', bbox_to_anchor=(0.8, 0.5), title='Labels')

    # Add title
    plt.title('Label Distribution')

    # Show the plot
    plt.show()


def new_pi_plot(data):
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))
    labels, sizes = np.unique(data['activity_id'], return_counts=True)
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
    plt.savefig('PAMAP2.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    window_size = 100
    step = 50
    Data = PAMAP2(window_size, step)


