import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch_geometric.data import DenseDataLoader

from config.paths import CACHE_ROOT, ensure_runtime_dirs, output_file, processed_file
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


def train(model, device, train_loader, optimizer):
    model.train()
    loss_train = 0.0
    num_sample = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, _, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        criterion = MMDLoss()
        cl_loss = criterion(x_local, x_global)
        mmd_beta = float(os.environ.get('HAG_DTA_MMD_BETA', 0.05))
        loss_all = loss + mmd_beta * l_loss + mmd_beta * e_loss + mmd_beta * cl_loss
        loss_all.backward()
        optimizer.step()
        loss_train += data.y.shape[0] * loss.item()
        num_sample += data.y.shape[0]
    return loss_train / num_sample


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


def evaluate_binary_metrics(y_true, y_score, y_pred):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    return [
        roc_auc_score(y_true, y_score),
        auc(recall_curve, precision_curve),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
    ]


def dataset_defaults(dataset_name):
    n1_defaults = {'Human': 7, 'Celegans': 7}
    n2_defaults = {'Human': 3, 'Celegans': 3}
    return n1_defaults[dataset_name], n2_defaults[dataset_name]


if len(sys.argv) != 3:
    raise SystemExit('Usage: python training_Human_Celegans.py <dataset_id> <model_id>')

dataset_name = ['Human', 'Celegans'][int(sys.argv[1])]
model_select = [Diff_DTA_GIN, Diff_DTA_GCN, Diff_DTA_GAT, Diff_DTA_SAGE][int(sys.argv[2])]
model_name = model_select.__name__
cuda_name = CUDA_NAME

TRAIN_BATCH_SIZE = CLASSIFICATION_TRAINING['train_batch_size']
TEST_BATCH_SIZE = CLASSIFICATION_TRAINING['test_batch_size']
LR = CLASSIFICATION_TRAINING['lr']
NUM_EPOCHS = CLASSIFICATION_TRAINING['num_epochs']
VAL_INTERVAL = CLASSIFICATION_TRAINING['val_interval']
EARLY_STOP_PATIENCE = CLASSIFICATION_TRAINING['early_stop_patience']

print(model_name)
print('dataset:', dataset_name)
print('Learning rate:', LR)
print('Epochs:', NUM_EPOCHS)
print('Validation interval:', VAL_INTERVAL)
print('Early stop patience:', EARLY_STOP_PATIENCE)

ensure_runtime_dirs()
ret_list = []

train_pt = processed_file(f'{dataset_name}_train')
val_pt = processed_file(f'{dataset_name}_val')
test_pt = processed_file(f'{dataset_name}_test')
if not all(os.path.isfile(p) for p in [train_pt, val_pt, test_pt]):
    raise FileNotFoundError(
        f'Missing processed files for {dataset_name}. Please run create_data_Human_Celegans.py first.'
    )

for seed in SEEDS:
    same_seeds(seed)
    print(f'\nrunning on {dataset_name}, seed={seed}')

    train_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_train')
    val_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_val')
    test_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_test')

    train_loader = DenseDataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DenseDataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DenseDataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    n1_default, n2_default = dataset_defaults(dataset_name)
    n1 = int(os.environ.get('HAG_DTA_N1', n1_default))
    n2 = int(os.environ.get('HAG_DTA_N2', n2_default))
    print(f'Hierarchical pooling: n1={n1}, n2={n2}')

    model = model_select(num_nodes_1=n1, num_nodes_2=n2).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_train_list = []
    val_auroc_list = []
    val_auprc_list = []
    val_precision_list = []
    val_recall_list = []

    best_val_roc = -1.0
    best_epoch = -1
    best_ret = None
    epochs_since_improvement = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, device, train_loader, optimizer)
        loss_train_list.append(train_loss)

        should_validate = ((epoch + 1) == 1) or ((epoch + 1) % VAL_INTERVAL == 0)
        if not should_validate:
            val_auroc_list.append(np.nan)
            val_auprc_list.append(np.nan)
            val_precision_list.append(np.nan)
            val_recall_list.append(np.nan)
            continue

        G_val, P_val, pred_val = predicting(model, device, val_loader)
        val_ret = evaluate_binary_metrics(G_val, P_val, pred_val)
        val_auroc_list.append(val_ret[0])
        val_auprc_list.append(val_ret[1])
        val_precision_list.append(val_ret[2])
        val_recall_list.append(val_ret[3])

        if val_ret[0] > best_val_roc:
            best_epoch = epoch + 1
            best_val_roc = val_ret[0]
            epochs_since_improvement = 0

            G_test, P_test, pred_test = predicting(model, device, test_loader)
            best_ret = evaluate_binary_metrics(G_test, P_test, pred_test)
            print(
                f'[FIND BEST!] epoch {best_epoch} | '
                f'val AUROC={val_ret[0]:.4f} AUPRC={val_ret[1]:.4f}'
                f' Precision={val_ret[2]:.4f} Recall={val_ret[3]:.4f}|'
                '\n'
                f'test AUROC={best_ret[0]:.4f} AUPRC={best_ret[1]:.4f} '
                f'Precision={best_ret[2]:.4f} Recall={best_ret[3]:.4f}'
            )
        else:
            epochs_since_improvement += 1
            print(
                f'epoch {epoch + 1} | '
                f'val AUROC={val_ret[0]:.4f} AUPRC={val_ret[1]:.4f} '
                f'Precision={val_ret[2]:.4f} Recall={val_ret[3]:.4f}'
            )

        if epochs_since_improvement >= EARLY_STOP_PATIENCE:
            print(f'early stopping at epoch {epoch + 1}')
            break

    if best_ret is None:
        raise RuntimeError(f'No validation result was recorded for {dataset_name}, seed={seed}.')

    pd.DataFrame({
        'train_loss': loss_train_list,
        'val_AUROC': val_auroc_list,
        'val_AUPRC': val_auprc_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list,
    }).to_csv(output_file(f'{dataset_name}训练损失_{seed}_{model_name}.csv'), index=0)

    ret_list.append(best_ret)

pd.DataFrame(ret_list, columns=['AUROC', 'AUPRC', 'Precision', 'Recall']).to_csv(
    output_file(f'{dataset_name}_{model_name}_random.csv'), index=0
)
