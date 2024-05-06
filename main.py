import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import utils

from preprocess.Dataset import get_dataloader
from transformer.Encoder import THP
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_2_8crypto.pkl', 'train')
    # print('[Info] Loading dev data...')
    # dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test_2_8crypto.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    model.train()
    total_event_log_likelihood = 0  # cumulative event log-likelihood
    total_time_square_error = 0  # cumulative time prediction squared error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    for batch in tqdm(training_data, mininterval=2, desc='  - [Training]  ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        event_ll, non_event_ll = utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll, non_event_ll)

        # type prediction
        pred_loss, pred_num_event = utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        time_se = utils.time_loss(prediction[1], event_time)

        # square errors are usually large, scale down to stabilize gradient descent
        scale_time_loss = 100
        loss = event_loss + pred_loss + time_se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_log_likelihood += -event_ll.item()
        total_time_square_error += time_se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constant.PAD).sum().item()
        total_num_pred += event_type.ne(Constant.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_square_error / total_num_pred)
    return total_event_log_likelihood / total_num_event, total_event_rate / total_num_pred, rmse


def val_epoch(model, validation_data, pred_loss_func, opt):
    model.eval()

    total_event_log_likelihood = 0
    total_time_square_error = 0
    total_event_rate = 0
    total_num_event = 0
    total_num_pred = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - [Validation]  ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_log_likelihood, non_event_log_likelihood = utils.log_likelihood(model, enc_out, event_time,
                                                                                  event_type)
            event_loss = -torch.sum(event_log_likelihood, non_event_log_likelihood)
            _, pred_num = utils.type_loss(prediction[0], event_type, pred_loss_func)
            time_se = utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_log_likelihood += -event_loss.item()
            total_time_square_error += time_se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_square_error / total_num_pred)
    return total_event_log_likelihood / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, pred_loss_func, opt):
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE

    for epoch_i in range(opt.epochs):
        epoch = epoch_i + 1
        print('[Info] Epoch {}/{}'.format(epoch, opt.epochs))
        start = time.time()
        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = val_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    # Data path toward where data file saved
    parser.add_argument('--data_path', type=str, default='./data/')
    # Data file
    parser.add_argument('--data', type=str, default='.pkl')
    # Training epoch
    parser.add_argument('--epochs', type=int, default=100)
    # Model dimension
    parser.add_argument('--d_model', type=int, default=512)
    # Batch size
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # RNN dimension
    parser.add_argument('--d_rnn', type=int, default=256)
    # Inner hidden dimension
    parser.add_argument('--d_inner_hid', type=int, default=128)
    # K dimension
    parser.add_argument('--d_k', type=int, default=64)
    # V dimension
    parser.add_argument('--d_v', type=int, default=64)
    # Heads
    parser.add_argument('--n_head', type=int, default=8)
    # Layers
    parser.add_argument('--n_layers', type=int, default=6)
    # Dropout
    parser.add_argument('--dropout', type=float, default=0.1)
    # Learning rate
    parser.add_argument('--lr', type=float, default=1e-4)
    # Smooth
    parser.add_argument('--smooth', type=float, default=0.1)
    # Weight decay
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # Log
    parser.add_argument('--log', type=str, default='log.txt')
    # Seed
    parser.add_argument('--seed', type=int, default=0)
    # Device
    parser.add_argument('--device', type=str, default='mps')

    opt = parser.parse_args()
    print('Device: {}'.format(opt.device))
    print('Loading data...')

    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-Likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """prepare dataloader"""
    train_loader, test_loader, num_types = prepare_dataloader(opt)

    model = THP(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of trainable parameters: {}'.format(num_params))

    """ train the model """
    train(model, train_loader, test_loader, optimizer, scheduler, pred_loss_func, opt)
