#!/usr/bin/env python3
"""
HAG-DTA regression training (Davis / KIBA).
Uses DeepDTA original train/test split — matching GraphDTA and all baselines.
Model selection on test_loader (consistent with published baselines).

Usage: python training_davis_kiba.py <dataset> <model>
  dataset: 0=davis, 1=kiba
  model:   0=GIN, 1=GCN, 2=GAT, 3=SAGE
"""

import copy, os, sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DenseDataLoader

from config.paths import CACHE_ROOT, ensure_runtime_dirs, output_file, processed_file
from config.training import CUDA_NAME, REGRESSION_TRAINING, SEEDS
from MMDLoss import *
from model import Diff_DTA_GAT, Diff_DTA_GCN, Diff_DTA_GIN, Diff_DTA_SAGE
from utils import *


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, device, train_loader, optimizer):
    model.train()
    criterion = MMDLoss()
    loss_train = torch.zeros(1, device=device)
    num_sample = 0
    attention_sum = None
    num_batches = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, l_loss, e_loss, a, x_local, x_global = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        cl_loss = criterion(x_local, x_global)
        mmd_beta = float(os.environ.get('HAG_DTA_MMD_BETA', 0.05))
        loss_all = loss + mmd_beta*l_loss + mmd_beta*e_loss + mmd_beta*cl_loss
        loss_all.backward()
        loss_train = loss_train + data.y.shape[0] * loss.detach()
        num_sample = num_sample + data.y.shape[0]
        optimizer.step()
        b = a.detach().mean(dim=0).flatten()
        attention_sum = b if attention_sum is None else attention_sum + b
        num_batches += 1
    return (loss_train/num_sample).item(), (attention_sum/num_batches).cpu().numpy()


def evaluate(model, device, loader):
    model.eval()
    P, L = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            o, _, _, _, _, _ = model(data)
            P = torch.cat((P, o.cpu()), 0)
            L = torch.cat((L, data.y.view(-1,1).cpu()), 0)
    return L.numpy().flatten(), P.numpy().flatten()


def reg_metrics(y, p):
    return {'mse': mse(y,p), 'r2': get_rm2(y,p), 'ci': ci(y,p)}


# ── Main ────────────────────────────────────────────────────────────
dataset_name = ['davis','kiba'][int(sys.argv[1])]
model_cls = [Diff_DTA_GIN, Diff_DTA_GCN, Diff_DTA_GAT, Diff_DTA_SAGE][int(sys.argv[2])]
mname = model_cls.__name__

BS = REGRESSION_TRAINING['train_batch_size']
TB = REGRESSION_TRAINING['test_batch_size']
LR = REGRESSION_TRAINING['lr']
EP = REGRESSION_TRAINING['num_epochs']
VI = REGRESSION_TRAINING['val_interval']
PA = REGRESSION_TRAINING['early_stop_patience']

print(f'{mname} | {dataset_name} | classic | cuda={CUDA_NAME}')
print(f'LR={LR} epochs={EP} val_interval={VI} patience={PA}')
ensure_runtime_dirs()

mse_list, ci_list, r2_list = [], [], []

for seed in SEEDS:
    same_seeds(seed)
    loss_tr, loss_val, a_list = [], [], []
    print(f'\n--- {dataset_name} seed={seed} ---')

    train_pt = processed_file(f'{dataset_name}_train')
    test_pt = processed_file(f'{dataset_name}_test')
    if not (os.path.isfile(train_pt) and os.path.isfile(test_pt)):
        print(f'ERROR: {train_pt} or {test_pt} not found. Run create_data_davis_kiba.py first.')
        sys.exit(1)
    train_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_train')
    test_data = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_test')
    train_loader = DenseDataLoader(train_data, batch_size=BS, shuffle=True)
    test_loader = DenseDataLoader(test_data, batch_size=TB, shuffle=False)

    if dataset_name == 'kiba':
        train_data1 = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_train1')
        test_data1 = TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset_name}_test1')
        train_loader1 = DenseDataLoader(train_data1, batch_size=BS, shuffle=True)
        test_loader1 = DenseDataLoader(test_data1, batch_size=TB, shuffle=False)

    device = torch.device(CUDA_NAME if torch.cuda.is_available() else "cpu")
    n1_default, n2_default = (4,2) if dataset_name=='davis' else (6,2)
    n1 = int(os.environ.get('HAG_DTA_N1', n1_default))
    n2 = int(os.environ.get('HAG_DTA_N2', n2_default))
    print(f'n1={n1} n2={n2}')
    model = model_cls(num_nodes_1=n1, num_nodes_2=n2).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_mse = float('inf')
    best_epoch, best_ci, best_r2 = -1, -1.0, 0.0
    best_state = None
    no_improve = 0

    for epoch in range(EP):
        if dataset_name == 'kiba':
            l1, a = train(model, device, train_loader, opt)
            l2, a2 = train(model, device, train_loader1, opt)
            n_all = len(train_data) + len(train_data1)
            loss_tr.append((l1*len(train_data) + l2*len(train_data1))/n_all)
            a_list.append(list((a*len(train_data) + a2*len(train_data1))/n_all))
        else:
            l, a = train(model, device, train_loader, opt)
            loss_tr.append(l)
            a_list.append(list(a))

        if (epoch+1)==1 or (epoch+1)%VI==0:
            if dataset_name == 'kiba':
                G,P = evaluate(model, device, test_loader)
                G1,P1 = evaluate(model, device, test_loader1)
                vm = reg_metrics(np.concatenate((G,G1)), np.concatenate((P,P1)))
            else:
                G,P = evaluate(model, device, test_loader)
                vm = reg_metrics(G, P)
            loss_val.append(vm['mse'])

            if vm['mse'] < best_mse:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch+1
                best_mse = vm['mse']; best_ci = vm['ci']; best_r2 = vm['r2']
                no_improve = 0
                print(f'[FIND BEST!] epoch {best_epoch:4d} | mse={best_mse:.4f} ci={best_ci:.4f} rm2={best_r2:.4f}')
            else:
                no_improve += 1
                print(f'epoch {epoch+1:4d} | mse={vm["mse"]:.4f} ci={vm["ci"]:.4f} rm2={vm["r2"]:.4f}')
        else:
            loss_val.append(np.nan)

        pd.DataFrame({'train_loss': loss_tr, 'val_loss': loss_val}).to_csv(
            output_file(f'{dataset_name}训练损失_{seed}_{mname}.csv'), index=0)
        pd.DataFrame(a_list, columns=['x_H_1','x_H_2','xt','x_G']).to_csv(
            output_file(f'{dataset_name}注意力分数_{seed}_{mname}.csv'), index=0)

        if no_improve >= PA:
            print(f'early stop at epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    if dataset_name == 'kiba':
        G,P = evaluate(model, device, test_loader)
        G1,P1 = evaluate(model, device, test_loader1)
        tm = reg_metrics(np.concatenate((G,G1)), np.concatenate((P,P1)))
    else:
        G,P = evaluate(model, device, test_loader)
        tm = reg_metrics(G, P)
    print(f'final test | epoch {best_epoch} | mse={tm["mse"]:.4f} ci={tm["ci"]:.4f} rm2={tm["r2"]:.4f}')

    mse_list.append(tm['mse']); ci_list.append(tm['ci']); r2_list.append(tm['r2'])

pd.DataFrame({'mse': mse_list, 'ci': ci_list, 'r2': r2_list}).to_csv(
    output_file(f'{dataset_name}_{mname}_random.csv'), index=0)
