import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, dataset_class, load_model, dataset_class_cluster
from Models.model import Encoder_factory, Rep_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Training import Self_Supervised_Trainer, SS_train_runner, SupervisedTrainer, Strain_runner
from Models.collate_function import collate_fn

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_dir', default='Dataset/All/', help='Data directory')
# /Epilepsy/Epilepsy
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Encoder_Type', default=['C-T'], choices={'T', 'C-T'}, help="Encoder Architecture")
parser.add_argument('--Rep_Type', default=['S2V'], choices={'S2V', 'TS-TCC'}, help="Self-Supervised Model")
# Transformers Parameters ----------------------------
parser.add_argument('--emb_size', type=int, default=64, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=128, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--rep_size', type=int, default=320, help='Representation dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'Sin', 'Learn', 'None'}, default='Sin',
                    help='Fix Position Embedding Type')
parser.add_argument('--Rel_pos_encode', choices={'Scalar', 'Vector', 'None'}, default='Scalar',
                    help='Relative Position Embedding Type')
# Training Parameters/ Hyper-Parameters ---------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    config['device'] = Initialization(config)
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Load Data ---------------------------------------------------------------
All_Results = ['Datasets', 'Series2Vec_Tran']
data_path = config['data_dir']
for problem in os.listdir(data_path):
    config['data_dir'] = data_path + problem
    print(problem)

    logger.info("Loading Data ...")
    Data = Data_Loader(config)
    # ---------------------------------------- Self Supervised Data -------------------------------------
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
    #                           collate_fn=lambda x: collate_fn(x))

    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Creating Distance based Self Supervised model ...")
    config['Data_shape'] = Data['All_train_data'].shape
    config['num_labels'] = int(max(Data['All_train_label']))+1
    Encoder = Encoder_factory(config)
    R_model = Rep_factory(config)
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)
    config['R_optimizer'] = optim_class(R_model.parameters(), lr=config['lr'], weight_decay=0)
    config['problem_type'] = 'Self-Supervised'
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    Encoder.to(config['device'])
    R_model.to(config['device'])
    # ---------------------------------------------- Training The Model ------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = Self_Supervised_Trainer(Encoder, R_model, train_loader, config, l2_reg=0, print_conf_mat=False)
    SS_train_runner(config, Encoder, SS_trainer, save_path)

    # **************************************************************************************************************** #
    # ----------------------------------------- Supervised Linear Training   -------------------------------------------

    # --------------------------------- Load Data -----------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    logger.info('Starting training linear layer on top of representation...')

    # ----------- Loading the model and freezing layers except FC layer ---------------------
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])

    for param in SS_Encoder.parameters():  # freezing layers
        param.requires_grad = False

    for param in SS_Encoder.C_out.parameters():  # except FC layer
        param.requires_grad = True

    S_trainer = SupervisedTrainer(SS_Encoder, R_model, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(SS_Encoder, R_model, val_loader, config, print_conf_mat=False)
    Strain_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(SS_Encoder, save_path, config['optimizer'])

    best_Encoder.to(config['device'])
    best_test_evaluator = SupervisedTrainer(best_Encoder, R_model, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)

    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(problem)
    print(print_str)
    dic_position_results = [problem, all_metrics['total_accuracy']]
    All_Results = np.vstack((All_Results, dic_position_results))

All_Results_df = pd.DataFrame(All_Results)
All_Results_df.to_csv(os.path.join(config['output_dir'], 'Self-Supervised.csv'))