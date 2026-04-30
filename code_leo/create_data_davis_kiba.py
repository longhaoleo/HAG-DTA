import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import *

class MyFilter(object):
    def __call__(self, data):
        return data.x.shape[0] <= 46
class MyFilter1(object):
    def __call__(self, data):
        return data.x.shape[0] > 46
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

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


def write_split_csv(dataset, split_name, rows, cols, drugs, prots, affinity):
    with open('data/' + dataset + '_' + split_name + '.csv', 'w') as f:
        f.write('compound_iso_smiles,target_sequence,affinity\n')
        for row_idx, col_idx in zip(rows, cols):
            ls = [drugs[row_idx], prots[col_idx], affinity[row_idx, col_idx]]
            f.write(','.join(map(str, ls)) + '\n')


def encode_split(df):
    drugs = np.asarray(list(df['compound_iso_smiles']))
    prots = np.asarray([seq_cat(t) for t in df['target_sequence']])
    labels = np.asarray(list(df['affinity']))
    return drugs, prots, labels


def rebuild_processed_dataset(dataset_name, split_name, drugs, prots, labels, smile_graph, pre_transform, pre_filter=None):
    processed_path = 'data/processed/' + dataset_name + '_' + split_name + '.pt'
    if os.path.isfile(processed_path):
        os.remove(processed_path)
    print('preparing ', dataset_name + '_' + split_name + '.pt in pytorch format!')
    TestbedDataset(
        root='data',
        dataset=dataset_name + '_' + split_name,
        xd=drugs,
        xt=prots,
        y=labels,
        smile_graph=smile_graph,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
    )

datasets = ['davis','kiba']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    rows, cols = np.where(np.isnan(affinity) == False)
    pair_indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(pair_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    split_indices = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }
    for split_name, split_idx in split_indices.items():
        write_split_csv(dataset, split_name, rows[split_idx], cols[split_idx], drugs, prots, affinity)
    print('\ndataset:', dataset)
    print('train/val/test:', len(train_idx), len(val_idx), len(test_idx))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['davis','kiba']:
    opts = ['train','val','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis','kiba']
# convert to PyTorch data format
for dataset in datasets:
    pre_transform = T.ToDense(46)
    for split_name in ['train', 'val', 'test']:
        df = pd.read_csv('data/' + dataset + '_' + split_name + '.csv')
        split_drugs, split_prots, split_y = encode_split(df)
        rebuild_processed_dataset(dataset, split_name, split_drugs, split_prots, split_y, smile_graph, pre_transform, MyFilter())
    if dataset == 'kiba':
        pre_transform = T.ToDense(268)
        for split_name in ['train', 'val', 'test']:
            df = pd.read_csv('data/' + dataset + '_' + split_name + '.csv')
            split_drugs, split_prots, split_y = encode_split(df)
            rebuild_processed_dataset(dataset, split_name + '1', split_drugs, split_prots, split_y, smile_graph, pre_transform, MyFilter1())
