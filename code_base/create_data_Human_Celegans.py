import os

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

from config.paths import CACHE_ROOT, DATA_ROOT, ensure_runtime_dirs, processed_file, raw_data_dir
from utils import *


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
        print('bad smile:', smile)
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
    print('preparing', dataset_name + '.pt', 'in pytorch format!')
    TestbedDataset(
        root=CACHE_ROOT,
        dataset=dataset_name,
        xd=drugs,
        xt=prots,
        y=labels,
        smile_graph=smile_graph,
        pre_transform=pre_transform,
    )


def split_dataframe(df):
    indices = np.arange(len(df))
    np.random.seed(1234)
    np.random.shuffle(indices)
    shuffled_df = df.iloc[indices].reset_index(drop=True)
    train_end = int(0.8 * len(shuffled_df))
    dev_test_end = int(0.9 * len(shuffled_df))
    train_df = shuffled_df.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled_df.iloc[train_end:dev_test_end].reset_index(drop=True)
    test_df = shuffled_df.iloc[dev_test_end:].reset_index(drop=True)
    return train_df, val_df, test_df


seq_voc = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

datasets = ['Human', 'Celegans']
full_frames = {}
ensure_runtime_dirs()

for dataset in datasets:
    print('convert data from DeepDTA for', dataset)
    fpath = os.path.join(raw_data_dir(dataset), dataset + '.txt')
    df = pd.read_table(fpath, sep=' ', header=None)
    df.to_csv(os.path.join(DATA_ROOT, f'{dataset}_all.csv'), index=False)
    full_frames[dataset] = df
    print('num_samples:', len(df))

print('\nCreating graph for all SMILES...')
compound_iso_smiles = []
for dataset in datasets:
    compound_iso_smiles += list(full_frames[dataset][0])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    if g is not None:
        smile_graph[smile] = g
print('Finished.')

print('\nConvert to pytorch data format...')
for dataset in datasets:
    df = full_frames[dataset]
    pre_transform = T.ToDense(184)
    drugs_all, prots_all, labels_all = encode_dataframe(df)
    rebuild_processed_dataset(dataset + '_all', drugs_all, prots_all, labels_all, smile_graph, pre_transform)

    train_df, val_df, test_df = split_dataframe(df)
    print(f'{dataset}: train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        drugs, prots, labels = encode_dataframe(split_df)
        rebuild_processed_dataset(f'{dataset}_{split_name}', drugs, prots, labels, smile_graph, pre_transform)
