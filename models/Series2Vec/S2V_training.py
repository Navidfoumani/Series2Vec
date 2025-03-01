import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.fft as fft
import torch.nn.functional as F
from utils import utils, analysis

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.Series2Vec.soft_dtw_cuda import SoftDTW
from models.Series2Vec.fft_filter import filter_frequencies

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']
        self.problem = config['problem']

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class S2V_SS_Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(S2V_SS_Trainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, _, IDs = batch

            Distance_out, Distance_out_f, rep_out, rep_out_f = self.model.Pretrain_forward(X.to(self.device))
            '''
            y = rep_out - rep_out.mean(dim=0)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))
            cov_y = (y.T @ y) / (len(rep_out) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            '''

            # Create a mask to select only the lower triangular values
            mask = torch.tril(torch.ones_like(Distance_out), diagonal=-1).bool()
            Distance_out = torch.masked_select(Distance_out, mask)
            Distance_out = Distance_normalizer(Distance_out)
            Distance_out_f = torch.masked_select(Distance_out_f, mask)
            Distance_out_f = Distance_normalizer(Distance_out_f)
            Dtw_Distance = cuda_soft_DTW(self.sdtw, X, len(X))
            Dtw_Distance = Distance_normalizer(Dtw_Distance)
            X_f = filter_frequencies(X)
            Euclidean_Distance_f = Euclidean_Dis(X_f, len(X_f))
            Euclidean_Distance_f = Distance_normalizer(Euclidean_Distance_f)
            temporal_loss = F.smooth_l1_loss(Distance_out, Dtw_Distance)
            frequency_loss = F.smooth_l1_loss(Distance_out_f, Euclidean_Distance_f)

            total_loss = temporal_loss + frequency_loss
            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if (epoch_num+1) % 5 == 0:
            self.model.eval()
            train_repr, train_labels = S2V_make_representation(self.model, self.train_loader)
            test_repr, test_labels = S2V_make_representation(self.model, self.test_loader)
            clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            y_hat = clf.predict(test_repr.cpu().detach().numpy())
            acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
            print('Test_acc:', acc_test)
            result_file = open(self.save_path + '/' + self.problem + '_linear_result.txt', 'a+')
            print('{0}, {1}, {2}'.format(epoch_num, acc_test, epoch_loss), file=result_file)
            result_file.close()

        return self.epoch_metrics

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class S2V_S_Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(S2V_S_Trainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,
                                            dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def Strain_runner(config, model, trainer, evaluator, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(evaluator, tensorboard_writer, config, best_metrics,
                                                                best_value, epoch)
        save_best_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


def SS_train_runner(config, model, trainer, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    save_best_model = utils.SaveBestModel()
    Total_loss = []
    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))
        Total_loss.append(aggr_metrics_train['loss'])
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    # plot_loss(Total_loss,Time_loss,Freq_loss)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


def cuda_soft_DTW(sdtw, X, size):
    # index = list(product([*range(size)], repeat=2))
    index = generate_list(size-1)
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    Dtw_Distance = sdtw(combination1, combination2)
    return Dtw_Distance


def Euclidean_Dis(X, size):
    index = generate_list(size - 1)
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    combination1_flat = combination1.view(combination1.size(0), -1)
    combination2_flat = combination2.view(combination2.size(0), -1)
    distances = torch.norm(combination1_flat - combination2_flat, dim=1)
    return distances


def generate_list(num):
    result = []
    for i in range(1, num+1):
        for j in range(0, i):
            result.append((i, j))
    return result


def Distance_normalizer(distance):

    if len(distance) == 1:
        Normal_distance = distance/distance
    else:
        min_val = torch.min(distance)
        max_val = torch.max(distance)

        # Normalize the distances between 0 and 1
        Normal_distance = (distance - min_val) / (max_val - min_val)
    return Normal_distance


def S2V_make_representation(model, data):
    out = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            X, targets, IDs = batch
            rep = model.linear_prob(X.to('cuda'))
            out.append(rep)
            labels.append(targets)

        out = torch.cat(out, dim=0)
        labels = torch.cat(labels, dim=0)
    return out, labels


def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe


def moving_average_smooth(data, window_size):
    """
    Smooth the input multivariate time series using a moving average separately for each channel.

    Parameters:
    - data (torch.Tensor): Input time series data of shape (sequence_length, num_variables).
    - window_size (int): Size of the moving average window.

    Returns:
    - smoothed_data (torch.Tensor): Smoothed time series data of the same shape as input.
    """
    num_variables = data.shape[1]

    # Use a 1D convolution with a uniform filter to perform the moving average for each channel
    smoothed_data = torch.zeros_like(data)
    for i in range(num_variables):
        kernel = torch.ones(window_size) / window_size
        kernel = kernel.view(1, 1, window_size)

        # Use padding to handle the edges of the time series
        padding = (window_size - 1) // 2
        channel_data = data[:, i:i + 1, :]
        smoothed_channel = F.conv1d(channel_data, kernel, padding=padding)
        smoothed_data[:, i:i + 1, :] = smoothed_channel

    return smoothed_data


def filter_low_frequencies_batch(batch_data, threshold_freq=40):
    # Convert to PyTorch tensor
    batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)

    # Apply 2D FFT
    fft_result = fft.fft2(batch_data_tensor, dim=(-2, -1))

    # Shift zero frequency component to the center
    fft_shifted = fft.fftshift(fft_result, dim=(-2, -1))

    # Get frequency components
    rows, cols = batch_data.shape[-2], batch_data.shape[-1]
    # Calculate corresponding frequencies in Hz
    freq_rows_hz = fft.fftfreq(rows, d=1.0)
    freq_cols_hz = fft.fftfreq(cols, d=1.0)

    # Create meshgrid for frequency components
    freq_rows_mesh, freq_cols_mesh = torch.meshgrid(torch.tensor(freq_rows_hz), torch.tensor(freq_cols_hz))

    # Filter frequencies lower than the threshold
    mask = (torch.abs(freq_rows_mesh) < threshold_freq) & (torch.abs(freq_cols_mesh) < threshold_freq)
    fft_shifted_filtered = fft_shifted * mask

    # Inverse FFT to get filtered signal
    filtered_data = fft.ifft2(fft.ifftshift(fft_shifted_filtered), dim=(-2, -1)).real

    return filtered_data


