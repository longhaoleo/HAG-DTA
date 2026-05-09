import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from rdkit import Chem
from torch import FloatTensor, LongTensor

from config.paths import CACHE_ROOT, ensure_runtime_dirs, processed_file, raw_data_dir


NUM_ATOM_FEAT = 34
DATASETS = ['Human', 'Celegans']
RESOURCE_DIR = Path(__file__).resolve().parent / 'resources' / 'transformercpi'
WORD2VEC_PATH = Path(os.environ.get('TRANSFORMERCPI_W2V_MODEL', RESOURCE_DIR / 'word2vec_30.model'))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f'input {x} not in allowable set {allowable_set}')
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_h=False, use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]
    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbol)
        + one_of_k_encoding(atom.GetDegree(), degree)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(atom.GetHybridization(), hybridization)
        + [atom.GetIsAromatic()]
    )
    if not explicit_h:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])
        except Exception:
            results += [False, False]
        results += [atom.HasProp('_ChiralityPossible')]
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])


def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f'SMILES cannot be parsed: {smiles}')
    atom_feat = np.zeros((mol.GetNumAtoms(), NUM_ATOM_FEAT), dtype=np.float32)
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol).astype(np.float32)
    return atom_feat, adj_matrix


def seq_to_kmers(seq, k=3):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def get_protein_embedding(model, protein):
    vec = np.zeros((len(protein), model.vector_size), dtype=np.float32)
    for i, word in enumerate(protein):
        vec[i] = model.wv[word]
    return vec


def split_dataframe(df):
    indices = np.arange(len(df))
    np.random.seed(1234)
    np.random.shuffle(indices)
    shuffled_df = df.iloc[indices].reset_index(drop=True)
    train_end = int(0.8 * len(shuffled_df))
    dev_test_end = int(0.9 * len(shuffled_df))
    return {
        'train': shuffled_df.iloc[:train_end].reset_index(drop=True),
        'val': shuffled_df.iloc[train_end:dev_test_end].reset_index(drop=True),
        'test': shuffled_df.iloc[dev_test_end:].reset_index(drop=True),
    }


def encode_split(df, w2v_model):
    encoded = []
    for row in df.itertuples(index=False):
        smiles, sequence, interaction = row
        atom_feat, adj = mol_features(str(smiles))
        protein_embedding = get_protein_embedding(w2v_model, seq_to_kmers(str(sequence)))
        encoded.append((
            FloatTensor(atom_feat),
            FloatTensor(adj),
            FloatTensor(protein_embedding),
            LongTensor([int(interaction)]),
        ))
    return encoded


def output_name(dataset, split):
    return f'transformercpi_{dataset}_{split}'


if __name__ == '__main__':
    ensure_runtime_dirs()
    if not WORD2VEC_PATH.exists():
        raise FileNotFoundError(f'Word2Vec model not found: {WORD2VEC_PATH}')

    print(f'Loading Word2Vec from {WORD2VEC_PATH}')
    w2v_model = Word2Vec.load(str(WORD2VEC_PATH))

    for dataset in DATASETS:
        raw_path = Path(raw_data_dir(dataset)) / f'{dataset}.txt'
        df = pd.read_table(raw_path, sep=' ', header=None)
        print(f'[{dataset}] total raw rows: {len(df)}')
        splits = split_dataframe(df)
        for split_name, split_df in splits.items():
            print(f'[{dataset}] {split_name}: {len(split_df)}')
            processed_path = processed_file(output_name(dataset, split_name))
            torch.save(encode_split(split_df, w2v_model), processed_path)
            print(f'  saved -> {processed_path}')
