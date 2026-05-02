import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch_geometric.data import DenseDataLoader

from config.paths import CACHE_ROOT, CHECKPOINT_DIR, ensure_runtime_dirs, fold_file, output_file, processed_file
from config.training import CUDA_NAME, REGRESSION_TRAINING, SEEDS
from MMDLoss import *
from model import Diff_DTA_GAT, Diff_DTA_GCN, Diff_DTA_GIN, Diff_DTA_SAGE
from utils import *


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    criterion = MMDLoss()
    loss_train = torch.zeros(1, device=device)
    num_sample = 0
    attention_sum = None
    num_batches = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, a, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        cl_loss = criterion(x_local, x_global)
        mmd_beta = float(os.environ.get('HAG_DTA_MMD_BETA', 0.05))
        loss_all = loss + mmd_beta * l_loss + mmd_beta * e_loss + mmd_beta * cl_loss
        loss_all.backward()
        loss_train = loss_train + data.y.shape[0] * loss.detach()
        num_sample = num_sample + data.y.shape[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                'Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss_all: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data.x),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    loss_all.item(),
                )
            )
        batch_attention = a.detach().mean(dim=0).flatten()
        if attention_sum is None:
            attention_sum = batch_attention
        else:
            attention_sum = attention_sum + batch_attention
        num_batches += 1
    epoch_attention = (attention_sum / num_batches).cpu().numpy()
    epoch_loss = (loss_train / num_sample).item()
    print("注意力分数 x_H_1, x_H_2, xt, x_G:", epoch_attention)
    print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, epoch_loss))
    return epoch_loss, epoch_attention


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _, _, _, _, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels, total_preds


datasets = [['davis', 'kiba'][int(sys.argv[1])]]
model_select = [Diff_DTA_GIN, Diff_DTA_GCN, Diff_DTA_GAT, Diff_DTA_SAGE][int(sys.argv[2])]
fold_id = int(sys.argv[3])
model_name = model_select.__name__
print(model_name)
print('fold_id:', fold_id)
cuda_name = CUDA_NAME
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = REGRESSION_TRAINING['train_batch_size']
TEST_BATCH_SIZE = REGRESSION_TRAINING['test_batch_size']
LR = REGRESSION_TRAINING['lr']
LOG_INTERVAL = REGRESSION_TRAINING['log_interval']
NUM_EPOCHS = REGRESSION_TRAINING['num_epochs']
VAL_INTERVAL = REGRESSION_TRAINING['val_interval']
EARLY_STOP_PATIENCE = REGRESSION_TRAINING['early_stop_patience']

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Validation interval: ', VAL_INTERVAL)
print('Early stop patience: ', EARLY_STOP_PATIENCE)
mse_list = []
ci_list = []
r2_list = []

ensure_runtime_dirs()


def load_fold_indices(dataset, fold_id):
    with open(fold_file(dataset, fold_id), 'r') as f:
        return json.load(f)


def build_position_map(dataset):
    pos_map = {}
    for pos in range(len(dataset)):
        pos_map[int(dataset[pos].index)] = pos
    return pos_map


def subset_from_original_ids(dataset, pos_map, original_ids):
    positions = [pos_map[idx] for idx in original_ids if idx in pos_map]
    return torch.utils.data.Subset(dataset, positions)


def evaluate_regression(model, device, loader):
    labels, preds = predicting(model, device, loader)
    return labels.numpy().flatten(), preds.numpy().flatten()


def regression_metrics(labels, preds):
    return {
        'mse': mse(labels, preds),
        'r2': get_rm2(labels, preds),
        'ci': ci(labels, preds),
    }


for seed in SEEDS:
    same_seeds(seed)
    loss_train_list = []
    loss_val_list = []
    a_list = []
    for dataset in datasets:
        print('\nrunning on ', dataset)
        # ⚡ 直接加载预分割的 per-fold .pt 文件（避免 Subset + DenseDataLoader 极慢）
        train_pt = processed_file(f'{dataset}_fold{fold_id}_train')
        val_pt = processed_file(f'{dataset}_fold{fold_id}_val')
        test_pt = processed_file(f'{dataset}_fold{fold_id}_test')
        if not all(os.path.isfile(p) for p in [train_pt, val_pt, test_pt]):
            print('please run create_data_davis_kiba.py to prepare per-fold .pt files!')
        else:
            train_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_train')
            val_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_val')
            test_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_test')

            train_loader = DenseDataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DenseDataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            test_loader = DenseDataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            if dataset == 'kiba':
                # ⚡ 直接加载预分割的 per-fold sub1 .pt 文件（大分子分支）
                train_pt1 = processed_file(f'{dataset}_sub1_fold{fold_id}_train')
                val_pt1 = processed_file(f'{dataset}_sub1_fold{fold_id}_val')
                test_pt1 = processed_file(f'{dataset}_sub1_fold{fold_id}_test')
                if not all(os.path.isfile(p) for p in [train_pt1, val_pt1, test_pt1]):
                    print('please run create_data_davis_kiba.py to prepare kiba sub1 per-fold .pt files!')
                    continue
                train_data1 = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_sub1_fold{fold_id}_train')
                val_data1 = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_sub1_fold{fold_id}_val')
                test_data1 = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_sub1_fold{fold_id}_test')
                train_loader1 = DenseDataLoader(train_data1, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
                val_loader1 = DenseDataLoader(val_data1, batch_size=TEST_BATCH_SIZE, shuffle=False)
                test_loader1 = DenseDataLoader(test_data1, batch_size=TEST_BATCH_SIZE, shuffle=False)

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            n1 = int(os.environ.get('HAG_DTA_N1', 6))
            n2 = int(os.environ.get('HAG_DTA_N2', 3))
            model = model_select(num_nodes_1=n1, num_nodes_2=n2)
            model = model.to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_val_ci = -1.0
            best_val_mse = float('inf')
            best_epoch = -1
            best_state_dict = None     # 内存暂存最优权重，不落盘
            epochs_since_improvement = 0

            for epoch in range(NUM_EPOCHS):
                time_begin = time.time()
                if dataset == 'kiba':
                    loss_train, a = train(model, device, train_loader, optimizer, epoch + 1)
                    loss_train1, a1 = train(model, device, train_loader1, optimizer, epoch + 1)
                    train_sample_count = len(train_data) + len(train_data1)
                    loss_train = (loss_train * len(train_data) + loss_train1 * len(train_data1)) / train_sample_count
                    loss_train_list.append(loss_train)
                    a = (a * len(train_data) + a1 * len(train_data1)) / train_sample_count
                    a_list.append(list(a))
                else:
                    loss_train, a = train(model, device, train_loader, optimizer, epoch + 1)
                    loss_train_list.append(loss_train)
                    a_list.append(list(a))

                should_validate = ((epoch + 1) == 1) or ((epoch + 1) % VAL_INTERVAL == 0)
                if should_validate:
                    if dataset == 'kiba':
                        G, P = evaluate_regression(model, device, val_loader)
                        G1, P1 = evaluate_regression(model, device, val_loader1)
                        val_metrics = regression_metrics(np.concatenate((G, G1)), np.concatenate((P, P1)))
                    else:
                        G, P = evaluate_regression(model, device, val_loader)
                        val_metrics = regression_metrics(G, P)

                    val_mse = val_metrics['mse']
                    val_ci = val_metrics['ci']
                    val_r2 = val_metrics['r2']
                    loss_val_list.append(val_mse)
                    if val_ci > best_val_ci:
                        best_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1
                        best_val_ci = val_ci
                        best_val_mse = val_mse
                        best_val_r2 = val_r2
                        epochs_since_improvement = 0
                        print(
                            'val CI improved at epoch ',
                            best_epoch,
                            '; best_val_ci, best_val_mse, best_val_r2:',
                            best_val_ci,
                            best_val_mse,
                            best_val_r2,
                            dataset
                        )
                    else:
                        epochs_since_improvement += 1
                        print(
                            val_ci,
                            'No improvement since epoch ',
                            best_epoch,
                            '; best_val_ci, best_val_mse, best_val_r2:',
                            best_val_ci,
                            best_val_mse,
                            best_val_r2,
                            dataset
                        )
                else:
                    loss_val_list.append(np.nan)
                    print(
                        'skip validation at epoch ',
                        epoch + 1,
                        '; next validation every',
                        VAL_INTERVAL,
                        'epochs'
                    )
                time_end = time.time()
                print("spend time：", time_end - time_begin, "s")
                d = pd.DataFrame(loss_train_list, columns=['train_loss'])
                d['val_loss'] = loss_val_list
                d.to_csv(output_file(f'{dataset}训练损失_fold{fold_id}_{seed}_{model_name}.csv'), index=0)
                d = pd.DataFrame(a_list, columns=['x_H_1', 'x_H_2', 'xt', 'x_G'])
                d.to_csv(output_file(f'{dataset}注意力分数_fold{fold_id}_{seed}_{model_name}.csv'), index=0)
                if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                    print('early stopping at epoch ', epoch + 1, ' due to no validation improvement.')
                    break

            model.load_state_dict(best_state_dict)
            if dataset == 'kiba':
                G, P = evaluate_regression(model, device, test_loader)
                G1, P1 = evaluate_regression(model, device, test_loader1)
                test_metrics = regression_metrics(np.concatenate((G, G1)), np.concatenate((P, P1)))
            else:
                G, P = evaluate_regression(model, device, test_loader)
                test_metrics = regression_metrics(G, P)
            final_mse = test_metrics['mse']
            final_ci = test_metrics['ci']
            final_r2 = test_metrics['r2']
            print('final test metrics at best val CI epoch ', best_epoch, ': mse=', final_mse, ' ci=', final_ci, ' rm2=', final_r2)
    mse_list.append(final_mse)
    ci_list.append(final_ci)
    r2_list.append(final_r2)
    d = pd.DataFrame({'mse': mse_list, 'ci': ci_list, 'r2': r2_list})
    d.to_csv(output_file(f'{dataset}_{model_name}_fold{fold_id}_random.csv'), index=0)
