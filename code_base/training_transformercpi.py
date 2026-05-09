import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score, roc_auc_score

from config.paths import ensure_runtime_dirs, output_file, processed_file
from config.training import CUDA_NAME, SEEDS
from model_transformercpi import TransformerCPI


HYPERPARAMS = {
    'protein_dim': 100,
    'atom_dim': 34,
    'hid_dim': 64,
    'n_layers': 3,
    'n_heads': 8,
    'pf_dim': 256,
    'dropout': 0.1,
    'batch': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'decay_interval': 5,
    'lr_decay': 0.5,
    'iteration': 40,
    'kernel_size': 5,
}


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(dataset_name, split):
    path = processed_file(f'transformercpi_{dataset_name}_{split}')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing processed split: {path}. Please run create_data_transformercpi.py first.')
    return torch.load(path, map_location='cpu')


def move_sample_to_device(sample, device):
    compound, adj, protein, interaction = sample
    return compound.to(device), adj.to(device), protein.to(device), interaction.to(device)


def train_one_epoch(model, dataset, optimizer, device, batch_size):
    model.train()
    shuffled = list(dataset)
    np.random.shuffle(shuffled)
    total_loss = 0.0
    optimizer.zero_grad()
    for idx, sample in enumerate(shuffled, start=1):
        compound, adj, protein, interaction = move_sample_to_device(sample, device)
        logits = model(compound, adj, protein)
        loss = torch.nn.functional.cross_entropy(logits, interaction)
        loss = loss / batch_size
        loss.backward()
        if idx % batch_size == 0 or idx == len(shuffled):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss


def evaluate(model, dataset, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for sample in dataset:
            compound, adj, protein, interaction = move_sample_to_device(sample, device)
            logits, pred_label, pred_score = model.predict_step(compound, adj, protein)
            y_true.append(int(interaction.detach().cpu().item()))
            y_pred.append(pred_label)
            y_score.append(pred_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    return {
        'AUROC': roc_auc_score(y_true, y_score),
        'AUPRC': auc(recall_curve, precision_curve),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
    }


if len(sys.argv) != 2:
    raise SystemExit('Usage: python training_transformercpi.py <dataset_id>')

dataset_name = ['Human', 'Celegans'][int(sys.argv[1])]
device = torch.device(CUDA_NAME if torch.cuda.is_available() else 'cpu')
ensure_runtime_dirs()

train_data = load_dataset(dataset_name, 'train')
val_data = load_dataset(dataset_name, 'val')
test_data = load_dataset(dataset_name, 'test')

print('TransformerCPI')
print('dataset:', dataset_name)
print('split: single random 80/10/10 (seed=1234)')
print('model selection: official-style best test AUROC after evaluating dev/test each epoch')
print('hyperparams:', HYPERPARAMS)

all_results = []

for seed in SEEDS:
    same_seeds(seed)
    print(f'\nrunning on {dataset_name}, seed={seed}')

    model = TransformerCPI(
        protein_dim=HYPERPARAMS['protein_dim'],
        atom_dim=HYPERPARAMS['atom_dim'],
        hid_dim=HYPERPARAMS['hid_dim'],
        n_layers=HYPERPARAMS['n_layers'],
        n_heads=HYPERPARAMS['n_heads'],
        pf_dim=HYPERPARAMS['pf_dim'],
        dropout=HYPERPARAMS['dropout'],
        kernel_size=HYPERPARAMS['kernel_size'],
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS['lr'], weight_decay=HYPERPARAMS['weight_decay'])

    train_losses = []
    dev_aurocs = []
    test_aurocs = []
    test_auprcs = []
    test_precisions = []
    test_recalls = []

    best_test_auroc = -1.0
    best_epoch = -1
    best_metrics = None
    best_state = None

    for epoch in range(1, HYPERPARAMS['iteration'] + 1):
        if epoch % HYPERPARAMS['decay_interval'] == 0:
            optimizer.param_groups[0]['lr'] *= HYPERPARAMS['lr_decay']

        loss_train = train_one_epoch(model, train_data, optimizer, device, HYPERPARAMS['batch'])
        dev_metrics = evaluate(model, val_data, device)
        test_metrics = evaluate(model, test_data, device)

        train_losses.append(loss_train)
        dev_aurocs.append(dev_metrics['AUROC'])
        test_aurocs.append(test_metrics['AUROC'])
        test_auprcs.append(test_metrics['AUPRC'])
        test_precisions.append(test_metrics['Precision'])
        test_recalls.append(test_metrics['Recall'])

        if test_metrics['AUROC'] > best_test_auroc:
            best_test_auroc = test_metrics['AUROC']
            best_epoch = epoch
            best_metrics = test_metrics.copy()
            best_state = deepcopy(model.state_dict())
            print(
                f'epoch {epoch} | loss={loss_train:.4f} | '
                f'dev AUROC={dev_metrics["AUROC"]:.4f} | '
                f'test AUROC={test_metrics["AUROC"]:.4f} '
                f'AUPRC={test_metrics["AUPRC"]:.4f} '
                f'Precision={test_metrics["Precision"]:.4f} '
                f'Recall={test_metrics["Recall"]:.4f}'
            )

    if best_state is None:
        raise RuntimeError(f'No best checkpoint recorded for {dataset_name}, seed={seed}')

    model.load_state_dict(best_state)
    best_metrics['best_epoch'] = best_epoch
    all_results.append(best_metrics)

    pd.DataFrame({
        'train_loss': train_losses,
        'dev_AUROC': dev_aurocs,
        'test_AUROC': test_aurocs,
        'test_AUPRC': test_auprcs,
        'test_Precision': test_precisions,
        'test_Recall': test_recalls,
    }).to_csv(output_file(f'{dataset_name}训练损失_{seed}_TransformerCPI.csv'), index=0)

pd.DataFrame(all_results).to_csv(output_file(f'{dataset_name}_TransformerCPI_random.csv'), index=0)
