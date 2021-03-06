import os
import argparse

import random
import time

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim

from script import dataloader, utility, earlystopping
from model import models

import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='SSGC')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable or disable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--mode', type=str, default='test', choices=['tuning', 'test'], \
                        help='running mode, default as test, tuning as alternative')
    parser.add_argument('--dataset', type=str, default='corar')
    parser.add_argument('--model', type=str, default='ssgc', help='graph neural network model')
    parser.add_argument('--gso_type', type=str, default='sym_renorm_adj', \
                        choices=['sym_renorm_adj', 'rw_renorm_adj'], \
                        help='graph shift operator, default as sym_renorm_adj, rw_renorm_adj as alternative')
    parser.add_argument('--K', type=int, default=2, help='K order')
    parser.add_argument('--beta', type=float, default=0, \
                        help='A hyperparameter for balancing between local and global graph information')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    SEED = args.seed
    set_env(SEED)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ??? 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode != 'test' and args.mode != 'tuning':
        raise ValueError(f'ERROR: Wrong running mode')
    else:
        mode = args.mode

    dataset = args.dataset

    if args.model != 'ssgc':
        raise ValueError(f'ERROR: This model is undefined.')
    else:
        model_name = args.model

    if args.gso_type != 'sym_renorm_adj' and args.gso_type != 'rw_renorm_adj':
        raise ValueError(f'ERROR: This graph shift operator is not adopted in this mode.')
    else:
        gso_type = args.gso_type

    if mode == 'tuning':
        param = nni.get_next_parameter()
        K, beta, lr, weight_decay = [*param.values()]
    else:
        if args.K < 0:
            raise ValueError(f'ERROR: The order K is smaller than 1!')
        else:
            K = args.K
        if args.beta < 0 or args.beta > 1:
            raise ValueError(f'ERROR: The balance parameter beta should be in [0, 1).')
        else:
            beta = args.beta
        lr = args.lr
        weight_decay = args.weight_decay
    
    enable_bias = args.enable_bias
    epochs = args.epochs
    opt = args.opt
    patience = args.patience

    model_save_dir = os.path.join('./model/save', dataset)
    os.makedirs(name=model_save_dir, exist_ok=True)
    model_save_path = model_name + '_' + gso_type + '_' + str(K) + '_order_' + str(beta) + '_beta' + '.pth'
    model_save_path = os.path.join(model_save_dir, model_save_path)

    return device, dataset, model_name, gso_type, lr, weight_decay, enable_bias, K, beta, epochs, opt, patience, model_save_path
    
def process_data(device, dataset, gso_type, K, beta):
    if dataset == 'corar' or dataset == 'citeseerr' or dataset == 'pubmed' or dataset == 'ogbn-arxiv':
        feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_citation_data(dataset)
    elif dataset == 'cornell' or dataset == 'texas' or dataset == 'washington' or dataset == 'wisconsin':
        feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_webkb_data(dataset)

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    gso = utility.calc_gso(adj, gso_type)
    
    if device == torch.device('cpu'):
        feature = utility.calc_ssgc_feature(gso, feature, K, beta, device)
        if sp.issparse(feature) == True:
            feature = utility.cnv_sparse_mat_to_coo_tensor(feature, device)
        else:
            feature = torch.from_numpy(feature).to(device)
        feature = feature.type(torch.FloatTensor)
    else:
        gso = utility.cnv_sparse_mat_to_coo_tensor(gso, device)
        if sp.issparse(feature) == True:
            feature = utility.cnv_sparse_mat_to_coo_tensor(feature, device)
        else:
            feature = torch.from_numpy(feature).to(device)
        feature = utility.calc_ssgc_feature(gso, feature, K, beta, device)
        feature = feature.type(torch.cuda.FloatTensor)

    label = torch.LongTensor(label).to(device)

    return feature, label, idx_train, idx_val, idx_test, n_feat, n_class

def prepare_model(n_feat, n_class, enable_bias, patience, model_save_path, opt, lr, weight_decay):
    model = models.SSGC(n_feat, n_class, enable_bias).to(device)

    loss = nn.NLLLoss()
    early_stopping = earlystopping.EarlyStopping(patience=patience, path=model_save_path, verbose=True)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    else:
        raise ValueError(f'ERROR: The {opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return model, loss, early_stopping, optimizer, scheduler

def train(epochs, model, optimizer, scheduler, early_stopping, feature, label, loss, idx_train, idx_val):
    train_time_list = []
    for epoch in range(epochs):
        train_epoch_begin_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        output = model(feature)
        loss_train = loss(output[idx_train], label[idx_train])
        acc_train = utility.calc_accuracy(output[idx_train], label[idx_train])
        loss_train.backward()
        optimizer.step()
        #scheduler.step()
        train_epoch_end_time = time.perf_counter()
        train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
        train_time_list.append(train_epoch_time_duration)

        loss_val, acc_val = val(model, label, output, loss, idx_val)
        print('Epoch: {:03d} | Learning rate: {:.8f} | Train loss: {:.6f} | Train acc: {:.6f} | Val loss: {:.6f} | Val acc: {:.6f} | Training duration: {:.6f}'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), train_epoch_time_duration))
        #nni.report_intermediate_result(acc_val.item())

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print('Early stopping.')
            break
    
    mean_train_epoch_time_duration = np.mean(train_time_list)
    print('\nTraining finished.\n')

    return mean_train_epoch_time_duration

def val(model, label, output, loss, idx_val):
    model.eval()
    with torch.no_grad():
        loss_val = loss(output[idx_val], label[idx_val])
        acc_val = utility.calc_accuracy(output[idx_val], label[idx_val])

    return loss_val, acc_val

def test(model, model_save_path, feature, label, loss, idx_test, model_name, dataset, mean_train_epoch_time_duration):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        output = model(feature)
        loss_test = loss(output[idx_test], label[idx_test])
        acc_test = utility.calc_accuracy(output[idx_test], label[idx_test])
        print('Model: {} | Dataset: {} | Test loss: {:.6f} | Test acc: {:.6f} | Training duration: {:.6f}'.format(model_name, dataset, loss_test.item(), acc_test.item(), mean_train_epoch_time_duration))
    #nni.report_final_result(acc_test.item())

if __name__ == "__main__":
    device, dataset, model_name, gso_type, lr, weight_decay, enable_bias, K, beta, epochs, opt, patience, model_save_path = get_parameters()
    feature, label, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device, dataset, gso_type, K, beta)
    model, loss, early_stopping, optimizer, scheduler = prepare_model(n_feat, n_class, enable_bias, patience, model_save_path, opt, lr, weight_decay)
    mean_train_epoch_time_duration = train(epochs, model, optimizer, scheduler, early_stopping, feature, label, loss, idx_train, idx_val)
    test(model, model_save_path, feature, label, loss, idx_test, model_name, dataset, mean_train_epoch_time_duration)
