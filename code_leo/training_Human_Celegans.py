import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch_geometric.data import DenseDataLoader

from config.paths import CACHE_ROOT, CHECKPOINT_DIR, ensure_runtime_dirs, fold_file, output_file, processed_file
from config.training import CLASSIFICATION_TRAINING, CUDA_NAME, SEEDS
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
    loss_train = 0
    num_sample = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, a, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        criterion = MMDLoss()
        cl_loss = criterion(x_local, x_global)
        mmd_beta = float(os.environ.get('HAG_DTA_MMD_BETA', 0.05))
        loss_all = loss + mmd_beta * l_loss + mmd_beta * e_loss + mmd_beta * cl_loss
        loss_all.backward()
        loss_train = loss_train + data.y.shape[0] * loss.item()
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
    epoch_loss = loss_train / num_sample
    print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, epoch_loss))
    return epoch_loss


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _, _, _, _, _ = model(data)
            predicted_values = torch.sigmoid(output)
            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)
    return (
        total_true_labels.numpy().flatten(),
        total_pred_values.numpy().flatten(),
        total_pred_labels.numpy().flatten(),
    )


datasets = [['Human', 'Celegans'][int(sys.argv[1])]]
model_select = [Diff_DTA_GIN, Diff_DTA_GCN, Diff_DTA_GAT, Diff_DTA_SAGE][int(sys.argv[2])]
fold_id = int(sys.argv[3])
model_name = model_select.__name__
print(model_name)
print('fold_id:', fold_id)
cuda_name = CUDA_NAME

TRAIN_BATCH_SIZE = CLASSIFICATION_TRAINING['train_batch_size']
TEST_BATCH_SIZE = CLASSIFICATION_TRAINING['test_batch_size']
LR = CLASSIFICATION_TRAINING['lr']
LOG_INTERVAL = CLASSIFICATION_TRAINING['log_interval']
NUM_EPOCHS = CLASSIFICATION_TRAINING['num_epochs']
VAL_INTERVAL = CLASSIFICATION_TRAINING['val_interval']
EARLY_STOP_PATIENCE = CLASSIFICATION_TRAINING['early_stop_patience']

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Validation interval: ', VAL_INTERVAL)
print('Early stop patience: ', EARLY_STOP_PATIENCE)
ret_list = []
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


for seed in SEEDS:
    same_seeds(seed)
    loss_train_list = []
    val_auroc_list = []
    val_auprc_list = []
    val_precision_list = []
    val_recall_list = []
    for dataset in datasets:
        print('\nrunning on ', dataset)
        # ⚡ 直接加载预分割的 per-fold .pt 文件（避免 Subset + DenseDataLoader 极慢）
        train_pt = processed_file(f'{dataset}_fold{fold_id}_train')
        val_pt = processed_file(f'{dataset}_fold{fold_id}_val')
        test_pt = processed_file(f'{dataset}_fold{fold_id}_test')
        if not all(os.path.isfile(p) for p in [train_pt, val_pt, test_pt]):
            print('please run create_data_Human_Celegans.py to prepare per-fold .pt files!')
        else:
            train_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_train')
            val_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_val')
            test_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_fold{fold_id}_test')

            train_loader = DenseDataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DenseDataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
            test_loader = DenseDataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            n1 = int(os.environ.get('HAG_DTA_N1', 6))
            n2 = int(os.environ.get('HAG_DTA_N2', 3))
            model = model_select(num_nodes_1=n1, num_nodes_2=n2)
            model = model.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_val_roc = -1
            best_epoch = -1
            epochs_since_improvement = 0
            model_file_name = os.path.join(CHECKPOINT_DIR, f'model_{dataset}_{model_name}_fold{fold_id}_{seed}.model')
            for epoch in range(NUM_EPOCHS):
                time_begin = time.time()
                train_loss = train(model, device, train_loader, optimizer, epoch + 1)
                loss_train_list.append(train_loss)
                should_validate = ((epoch + 1) == 1) or ((epoch + 1) % VAL_INTERVAL == 0)
                if should_validate:
                    G, P, pred = predicting(model, device, val_loader)
                    precision_val, recall_val, _ = precision_recall_curve(G, P)
                    val_ret = [
                        roc_auc_score(G, P),
                        auc(recall_val, precision_val),
                        precision_score(G, pred),
                        recall_score(G, pred),
                    ]
                    val_auroc_list.append(val_ret[0])
                    val_auprc_list.append(val_ret[1])
                    val_precision_list.append(val_ret[2])
                    val_recall_list.append(val_ret[3])
                    if val_ret[0] > best_val_roc:
                        torch.save(model.state_dict(), model_file_name)
                        best_epoch = epoch + 1
                        best_val_roc = val_ret[0]
                        epochs_since_improvement = 0
                        G_test, P_test, pred_test = predicting(model, device, test_loader)
                        precision_test, recall_test, _ = precision_recall_curve(G_test, P_test)
                        auprc_test = auc(recall_test, precision_test)
                        best_ret = [
                            roc_auc_score(G_test, P_test),
                            auprc_test,
                            precision_score(G_test, pred_test),
                            recall_score(G_test, pred_test),
                        ]
                        print(
                            'val AUROC improved at epoch ',
                            best_epoch,
                            '; best_AUROC, best_AUPRC, best_precision, best_recall:',
                            best_ret[0],
                            best_ret[1],
                            best_ret[2],
                            best_ret[3],
                            dataset,
                        )
                    else:
                        epochs_since_improvement += 1
                        print(
                            val_ret[0],
                            'No improvement since epoch ',
                            best_epoch,
                            '; best_AUROC, best_AUPRC, best_precision, best_recall:',
                            best_ret[0],
                            best_ret[1],
                            best_ret[2],
                            best_ret[3],
                            dataset,
                        )
                else:
                    val_auroc_list.append(np.nan)
                    val_auprc_list.append(np.nan)
                    val_precision_list.append(np.nan)
                    val_recall_list.append(np.nan)
                    print(
                        'skip validation at epoch ',
                        epoch + 1,
                        '; next validation every',
                        VAL_INTERVAL,
                        'epochs'
                    )
                time_end = time.time()
                print("spend time：", time_end - time_begin, "s")
                d = pd.DataFrame({
                    'train_loss': loss_train_list,
                    'val_AUROC': val_auroc_list,
                    'val_AUPRC': val_auprc_list,
                    'val_precision': val_precision_list,
                    'val_recall': val_recall_list,
                })
                d.to_csv(output_file(f'{dataset}训练损失_fold{fold_id}_{seed}_{model_name}.csv'), index=0)
                if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                    print('early stopping at epoch ', epoch + 1, ' due to no validation improvement.')
                    break
    ret_list.append(best_ret)
    d = pd.DataFrame(ret_list, columns=['AUROC', 'AUPRC', 'Precision', 'Recall'])
    d.to_csv(output_file(f'{dataset}_{model_name}_fold{fold_id}_random.csv'), index=0)
