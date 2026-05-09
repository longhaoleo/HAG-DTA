import json
import os
import pickle
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

from config.paths import CACHE_ROOT, DATA_ROOT, ensure_runtime_dirs, processed_file, raw_data_dir
from utils import *


class MyFilter(object):
    def __call__(self, data):
        return data.x.shape[0] <= 46


class MyFilter1(object):
    def __call__(self, data):
        return data.x.shape[0] > 46


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
    c_size = mol.GetNumAtoms()

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
    for i in range(c_size):
        edge_index.append([i, i])
    return c_size, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def build_full_dataframe(drugs, prots, affinity):
    rows, cols = np.where(np.isnan(affinity) == False)
    return pd.DataFrame({
        'compound_iso_smiles': [drugs[row_idx] for row_idx in rows],
        'target_sequence': [prots[col_idx] for col_idx in cols],
        'affinity': [affinity[row_idx, col_idx] for row_idx, col_idx in zip(rows, cols)],
    })


def encode_dataframe(df):
    drugs = np.asarray(list(df['compound_iso_smiles']))
    prots = np.asarray([seq_cat(t) for t in df['target_sequence']])
    labels = np.asarray(list(df['affinity']))
    return drugs, prots, labels


def rebuild_processed_dataset(dataset_name, drugs, prots, labels, smile_graph, pre_transform, pre_filter=None):
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
        pre_filter=pre_filter,
    )


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

datasets = ['davis', 'kiba']
full_frames = {}
ensure_runtime_dirs()

for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = raw_data_dir(dataset) + '/'
    ligands = json.load(open(fpath + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + 'Y', 'rb'), encoding='latin1')

    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])

    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    full_df = build_full_dataframe(drugs, prots, affinity)
    full_df.to_csv(os.path.join(DATA_ROOT, f'{dataset}_all.csv'), index=False)
    full_frames[dataset] = full_df
    print('\ndataset:', dataset)
    print('num_samples:', len(full_df))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))


print('\nCreating graph for all SMILES...')
compound_iso_smiles = []
for dataset in datasets:
    compound_iso_smiles += list(full_frames[dataset]['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    smile_graph[smile] = smile_to_graph(smile)
print('Finished.')


for dataset in ['davis', 'kiba']:
    df = full_frames[dataset]
    all_drugs, all_prots, all_y = encode_dataframe(df)

    # 兼容文件（_all.pt）
    rebuild_processed_dataset(dataset + '_all', all_drugs, all_prots, all_y, smile_graph, T.ToDense(46), MyFilter())
    if dataset == 'kiba':
        rebuild_processed_dataset(dataset + '_all1', all_drugs, all_prots, all_y, smile_graph, T.ToDense(268), MyFilter1())

# ═══════════════════════════════════════════════════════════════════════
# Classic .pt 文件（DeepDTA 原始 split，训练用）
# ═══════════════════════════════════════════════════════════════════════
print('\nGenerating classic train/test .pt files ...')

for dataset in ['davis', 'kiba']:
    fpath = raw_data_dir(dataset) + '/'
    train_fold = json.load(open(fpath + 'folds/train_fold_setting1.txt'))
    test_fold = json.load(open(fpath + 'folds/test_fold_setting1.txt'))
    train_flat = [ee for e in train_fold for ee in e]  # list of lists → flatten
    test_flat = test_fold  # already flat list of ints

    df = full_frames[dataset]
    all_drugs, all_prots, all_y = encode_dataframe(df)

    train_drugs = all_drugs[train_flat]
    train_prots = all_prots[train_flat]
    train_y = all_y[train_flat]
    test_drugs = all_drugs[test_flat]
    test_prots = all_prots[test_flat]
    test_y = all_y[test_flat]

    if not os.path.isfile(processed_file(f'{dataset}_train')):
        print(f'  {dataset}_train.pt ...')
        TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_train',
                       xd=train_drugs, xt=train_prots, y=train_y,
                       smile_graph=smile_graph, pre_transform=T.ToDense(46), pre_filter=MyFilter())
    if not os.path.isfile(processed_file(f'{dataset}_test')):
        print(f'  {dataset}_test.pt ...')
        TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_test',
                       xd=test_drugs, xt=test_prots, y=test_y,
                       smile_graph=smile_graph, pre_transform=T.ToDense(46), pre_filter=MyFilter())

    if dataset == 'kiba':
        if not os.path.isfile(processed_file(f'{dataset}_train1')):
            print(f'  {dataset}_train1.pt ...')
            TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_train1',
                           xd=train_drugs, xt=train_prots, y=train_y,
                           smile_graph=smile_graph, pre_transform=T.ToDense(268), pre_filter=MyFilter1())
        if not os.path.isfile(processed_file(f'{dataset}_test1')):
            print(f'  {dataset}_test1.pt ...')
            TestbedDataset(root=CACHE_ROOT, dataset=f'{dataset}_test1',
                           xd=test_drugs, xt=test_prots, y=test_y,
                           smile_graph=smile_graph, pre_transform=T.ToDense(268), pre_filter=MyFilter1())

    print(f'  {dataset}: train={len(train_flat)} test={len(test_flat)}')

print('Done.')
