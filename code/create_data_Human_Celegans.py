import pandas as pd
from rdkit import Chem
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import *
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
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
    if (mol is None):
        print("bad smile:", smile)
    else:
        # 1.num of atoms
        num_atoms = mol.GetNumAtoms()
        # 2.features
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))
        # 3.edges
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        # if(edge_index == []):
        #     print(smile)  # molecules that have no bonds

        return num_atoms, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


datasets = ['Human', 'Celegans']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/' + dataset + '.txt'
    data = pd.read_table(fpath, sep=' ', header=None)

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 2. create graph for all SMILES
print("\nCreating graph for all SMILES...")
compound_iso_smiles = []
for dataset in datasets:
    df = pd.read_table('data/' + dataset + '/' + dataset + '.txt', sep=' ', header=None)
    compound_iso_smiles += list(df[0])  # the first column is drug SMILES
compound_iso_smiles = set(compound_iso_smiles)  # function set() can remove redundant automatically
smile_graph = {}

for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
print("Finished.")

# 3. convert to PyTorch data format
print("\nConvert to pytorch data format...")
for dataset in datasets:
    df = pd.read_table('data/' + dataset + '/' + dataset + '.txt', sep=' ', header=None)
    print(len(list(df[0])))
    if dataset=='Human':
        pre_transform = T.ToDense(184)
    if dataset=='Celegans':
        pre_transform = T.ToDense(184)
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df[2], random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data[2], random_state=42)
    train_drugs, train_prots, train_Y = list(train_data[0]), list(train_data[1]), list(train_data[2])
    train_prots = [seq_cat(t) for t in train_prots]
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    # make data PyTorch Geometric ready
    # train set
    data_path = 'data/processed/' + dataset + '_train' + '.pt'
    if not os.path.isfile(data_path):  # if not exists
        print(data_path+ ' in pytorch format!')
        TestbedDataset(root='data', dataset=dataset + '_train' ,
                       xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph, pre_transform=pre_transform)
        print(data_path, 'created successfully.')
    else:
        print(data_path, 'are already created.')
    val_drugs, val_prots, val_Y = list(val_data[0]), list(val_data[1]), list(val_data[2])
    val_prots = [seq_cat(t) for t in val_prots]
    val_drugs, val_prots, val_Y = np.asarray(val_drugs), np.asarray(val_prots), np.asarray(val_Y)
    # test set
    data_path = 'data/processed/' + dataset + '_val' + '.pt'
    if not os.path.isfile(data_path):  # if not exists
        print('preparing ', dataset + '_val' + '.pt in pytorch format!')
        TestbedDataset(root='data', dataset=dataset + '_val',
                       xd=val_drugs, xt=val_prots, y=val_Y, smile_graph=smile_graph,  pre_transform=pre_transform)
    else:
        print(data_path, 'are already created.')
    test_drugs, test_prots, test_Y = list(test_data[0]), list(test_data[1]), list(test_data[2])
    test_prots = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)
    # test set
    data_path = 'data/processed/' + dataset + '_test' + '.pt'
    if not os.path.isfile(data_path):  # if not exists
        print('preparing ', dataset + '_test' + '.pt in pytorch format!')
        TestbedDataset(root='data', dataset=dataset + '_test',
                       xd=test_drugs, xt=test_prots, y=test_Y, smile_graph=smile_graph,  pre_transform=pre_transform)
    else:
        print(data_path, 'are already created.')