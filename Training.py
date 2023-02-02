import os
import logging
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from itertools import product

from Models.loss import l2_reg_loss
from Models import utils, analysis
from Models.optimizers import get_optimizer
from Models.soft_dtw_cuda import SoftDTW

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, R_model, dataloader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):

        self.model = model
        self.R_model = R_model
        self.dataloader = dataloader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.R_optimizer = config['R_optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

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


class Self_Supervised_Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):

        super(Self_Supervised_Trainer, self).__init__(*args, **kwargs)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()
        self.R_model = self.R_model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.dataloader):
            # contrastive_loss = NTXentLoss(self.device, batch[0].shape[0], 0.2, use_cosine_similarity=True)
            X, targets, IDs = batch
            Representation, _ = self.model(X.to(self.device))
            Con_Rep, Rep_Euclidean = data_generator(Representation, len(targets))

            Distance_out = self.R_model(Con_Rep.to(self.device))
            Dtw_Distance = cuda_soft_DTW(self.sdtw, X, len(targets))
            Dtw_Distance = torch.nn.functional.normalize(Dtw_Distance, p=2.0, dim=1, eps=1e-12, out=None)
            total_loss = torch.nn.functional.mse_loss(Distance_out.squeeze(), Dtw_Distance)

            # Ecludian_Distance = cuda_Ecludian(Representation, len(targets))
            # total_loss = torch.nn.functional.mse_loss(Ecludian_Distance, Dtw_Distance)
            # Ecludian_Distance = cuda_Ecludian(X, len(targets))
            # total_loss = torch.nn.functional.mse_loss(Distance_out.squeeze(), Ecludian_Distance)


            '''
            Distance_out = torch.zeros(len(targets), len(targets))
            for i in range(len(targets)):
                for j in range(i+1, len(targets)):
                    Distance_out[i, j] = self.R_model(Representation[i], Representation[j])
                    Distance_out[j, i] = Distance_out[j, i]
            
            # SS_out2, _ = self.model(X2.to(self.device))
            sim_temp = torch.einsum('ijk,ijk->ik', X, X)
            Raw_sim = torch.matmul(sim_temp, sim_temp.T)/sim_temp.shape[0]
            Encode_sim = torch.matmul(Sim_out, Sim_out.T)

            Raw_sim.to(self.device)
            Encode_sim.to(self.device)
            ind = np.diag_indices(Raw_sim.size(dim=1))
            Raw_sim[ind[0], ind[1]] = torch.zeros(Raw_sim.size(dim=1))
            Encode_sim[ind[0], ind[1]] = torch.zeros(Encode_sim.size(dim=1)).to(self.device)
            '''
            # Sim_out = torch.nn.functional.normalize(Sim_out, p=2.0, dim=1, eps=1e-12, out=None)
            # Sim = torch.nn.functional.normalize(Sim, p=2.0, dim=1, eps=1e-12, out=None)

            # total_loss = torch.nn.functional.cosine_similarity(Sim.to(self.device), Sim_out[:, 0:Sim.shape[-1]])
            # total_loss = contrastive_loss(SS_out1, X, cluster)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            self.R_optimizer.zero_grad()

            total_loss.sum().backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            self.R_optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.mean().item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics


def data_generator(representation, size):

    index = list(product([*range(size)], repeat=2))
    df_rep = pd.DataFrame(representation.cpu().detach().numpy())
    combination1 = df_rep.iloc[[i[0] for i in index]].reset_index(drop=True)
    combination2 = df_rep.iloc[[i[1] for i in index]].reset_index(drop=True)
    combination = pd.concat([combination1, combination2], axis=1)
    out = torch.tensor(combination.values)
    Rep_Euclidean = np.linalg.norm(combination1.values - combination2.values, axis=1)
    # X2 = torch.tensor(combination2.values)
    return out, Rep_Euclidean


def cuda_soft_DTW(sdtw, X, size):
    index = list(product([*range(size)], repeat=2))
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    Dtw_Distance = sdtw(combination1, combination2)
    return Dtw_Distance


def cuda_Ecludian(X, size):
    index = list(product([*range(size)], repeat=2))
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    Ecludian_Distance = ((combination1 - combination2)**2).sum(axis=1)
    # Ecludian_Distance = torch.einsum('ijk,ijk->i', combination1, combination2)
    return Ecludian_Distance

def SS_train_runner(config, model, trainer, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    save_best_model = utils.SaveBestModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
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


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, clusters):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Cluster based Negative sampling ---------------------------------------
        filter_same_cluster = cluster_masking(clusters).type(torch.bool)
        similarity_matrix[filter_same_cluster] = 0
        # -----------------------------------------------------------------------
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def cluster_masking(clusters):
    clusters = torch.cat([clusters, clusters], dim=0)
    hot_clusters = torch.nn.functional.one_hot(clusters.to(torch.int64))
    sim_cluster = torch.matmul(hot_clusters, hot_clusters.T)
    ind = np.diag_indices(sim_cluster.size(dim=1))
    sim_cluster[ind[0], ind[1]] = torch.zeros(sim_cluster.size(dim=1), dtype=torch.int64)
    return sim_cluster


class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.dataloader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            _, predictions = self.model(X.to(self.device))
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
        for i, batch in enumerate(self.dataloader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            _, predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            #if i % self.print_interval == 0:
                #ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                #self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        '''
        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)
        '''
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
    save_best_model = utils.SaveBestModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(evaluator, tensorboard_writer, config, best_metrics,
                                                                best_value, epoch)
        save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
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