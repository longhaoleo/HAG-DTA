import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split

from config.paths import CACHE_ROOT, DATA_ROOT, FOLD_DIR, ensure_runtime_dirs, processed_file, raw_data_dir
from utils import *

NUM_FOLDS = 5
TEST_RATIO = 0.2


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
             'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
             'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'Unknown']
        ) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("bad smile:", smile)
        return None

    num_atoms = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return num_atoms, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def encode_dataframe(df):
    drugs = np.asarray(list(df[0]))
    prots = np.asarray([seq_cat(t) for t in df[1]])
    labels = np.asarray(list(df[2]))
    return drugs, prots, labels


def rebuild_processed_dataset(dataset_name, drugs, prots, labels, smile_graph, pre_transform):
    processed_path = processed_file(dataset_name)
    if os.path.isfile(processed_path):
        os.remove(processed_path)
    print('preparing ', dataset_name + '.pt in pytorch format!')
    TestbedDataset(
        root=CACHE_ROOT,
        dataset=dataset_name,
        xd=drugs,
        xt=prots,
        y=labels,
        smile_graph=smile_graph,
        pre_transform=pre_transform,
    )


def save_fold_pt(dataset, fold_id, fold_indices, drugs, prots, labels, smile_graph, pre_transform):
    """Save pre-split .pt files for a single fold (instead of Subset at runtime)."""
    for split_name in ['train', 'val', 'test']:
        idx = fold_indices[split_name]
        ds_name = f'{dataset}_fold{fold_id}_{split_name}'
        print(f'  creating {ds_name}.pt ({len(idx)} samples) ...')
        rebuild_processed_dataset(ds_name, drugs[idx], prots[idx], labels[idx], smile_graph, pre_transform)


def write_fold_indices(dataset, labels):
    all_indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(all_indices, test_size=TEST_RATIO, stratify=labels, random_state=42)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold_id, (inner_train_pos, val_pos) in enumerate(skf.split(trainval_idx, labels[trainval_idx])):
        train_idx = trainval_idx[inner_train_pos]
        val_idx = trainval_idx[val_pos]
        fold_indices = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
        }
        fold_path = os.path.join(FOLD_DIR, f'{dataset}_fold{fold_id}.json')
        with open(fold_path, 'w') as f:
            json.dump(fold_indices, f)
        print(f'fold {fold_id}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

datasets = ['Human', 'Celegans']
full_frames = {}
ensure_runtime_dirs()

for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = os.path.join(raw_data_dir(dataset), dataset + '.txt')
    df = pd.read_table(fpath, sep=' ', header=None)
    df.to_csv(os.path.join(DATA_ROOT, f'{dataset}_all.csv'), index=False)
    full_frames[dataset] = df
    print('num_samples:', len(df))
    write_fold_indices(dataset, df[2].to_numpy())


print("\nCreating graph for all SMILES...")
compound_iso_smiles = []
for dataset in datasets:
    compound_iso_smiles += list(full_frames[dataset][0])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    if g is not None:
        smile_graph[smile] = g
print("Finished.")


print("\nConvert to pytorch data format...")
for dataset in datasets:
    df = full_frames[dataset]
    pre_transform = T.ToDense(184)
    drugs, prots, labels = encode_dataframe(df)

    # 仍生成 _all.pt（兼容）但训练不再用它
    rebuild_processed_dataset(dataset + '_all', drugs, prots, labels, smile_graph, pre_transform)

    # ⚡ 核心优化：预生成 per-fold 分割 .pt 文件
    print(f'\nGenerating per-fold .pt files for {dataset} ...')
    for fold_id in range(NUM_FOLDS):
        fold_path = os.path.join(FOLD_DIR, f'{dataset}_fold{fold_id}.json')
        with open(fold_path, 'r') as f:
            fold_indices = json.load(f)
        save_fold_pt(dataset, fold_id, fold_indices, drugs, prots, labels, smile_graph, pre_transform)
