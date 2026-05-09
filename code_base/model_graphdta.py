"""
GraphDTA baseline — dense-graph adaptation for HAG-DTA pipeline.
Architecture matches Nguyen et al. (2021) exactly; uses DenseGNNConv
to stay compatible with HAG-DTA's DenseDataLoader.

Drug:   GNN layers → global max pool → 128d
Protein: Embedding(26→128) → Conv1d(1000,32,8) → Flatten → 128d
Fusion:  concat(256) → 1024 → 256 → 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (DenseGINConv, DenseGCNConv,
                                DenseGATConv, GraphNorm)


# ── Drug GNNs ───────────────────────────────────────────────────────

class DrugGNN_GCN(nn.Module):
    """3 layers: 78→78→156→312, then fc: 312→1024→128."""
    def __init__(self, in_dim=78):
        super().__init__()
        self.conv1 = DenseGCNConv(in_dim, in_dim)
        self.conv2 = DenseGCNConv(in_dim, in_dim * 2)
        self.conv3 = DenseGCNConv(in_dim * 2, in_dim * 4)
        self.fc1 = nn.Linear(in_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj, mask=None):
        x = self.relu(self.conv1(x, adj, mask))
        x = self.relu(self.conv2(x, adj, mask))
        x = self.relu(self.conv3(x, adj, mask))
        x = x.max(dim=1)[0]                        # global max pool
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DrugGNN_GAT(nn.Module):
    """2 GAT layers: 10 heads → 1 head, with ELU + dropout."""
    def __init__(self, in_dim=78):
        super().__init__()
        self.conv1 = DenseGATConv(in_dim, in_dim, heads=10, concat=True)
        self.conv2 = DenseGATConv(in_dim * 10, 128, heads=1, concat=False)
        self.fc = nn.Linear(128, 128)

    def forward(self, x, adj, mask=None):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, adj, mask))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, adj, mask)
        x = F.relu(x)
        x = x.max(dim=1)[0]
        return F.relu(self.fc(x))


class DrugGNN_GIN(nn.Module):
    """5 GIN layers, dim=32 constant, global ADD pool."""
    def __init__(self, in_dim=78, hidden=32):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(5):
            cin = in_dim if i == 0 else hidden
            nn_seq = Sequential(Linear(cin, hidden), ReLU(), Linear(hidden, hidden))
            self.convs.append(DenseGINConv(nn_seq))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.fc = nn.Linear(hidden, 128)

    def forward(self, x, adj, mask=None):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(conv(x, adj, mask))
            x = bn(x)
        x = x.sum(dim=1)                            # global add pool
        return F.relu(self.fc(x))


class DrugGNN_SAGE(nn.Module):
    """3 SAGE layers: 78→78→156→312 (same expansion as GCN)."""
    def __init__(self, in_dim=78):
        super().__init__()
        self.conv1 = DenseSAGEConv(in_dim, in_dim)
        self.conv2 = DenseSAGEConv(in_dim, in_dim * 2)
        self.conv3 = DenseSAGEConv(in_dim * 2, in_dim * 4)
        self.fc1 = nn.Linear(in_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj, mask=None):
        x = self.relu(self.conv1(x, adj, mask))
        x = self.relu(self.conv2(x, adj, mask))
        x = self.relu(self.conv3(x, adj, mask))
        x = x.max(dim=1)[0]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── Protein CNN ─────────────────────────────────────────────────────

class ProteinCNN(nn.Module):
    """Single Conv1d(1000, 32, 8) matching GraphDTA official code."""
    def __init__(self, embed_dim=128, n_filters=32, kernel_size=8, out_dim=128):
        super().__init__()
        self.embed = nn.Embedding(26, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(1000, n_filters, kernel_size)
        self.fc = nn.Linear(n_filters * 121, out_dim)
        self.relu = nn.ReLU()

    def forward(self, target):
        x = self.embed(target)                      # (B, 1000, 128)
        x = self.conv(x)                            # (B, 32, 121)
        x = self.relu(x)
        x = x.view(x.size(0), -1)                   # (B, 3872)
        return self.fc(x)                           # (B, 128)


# ── Full model ──────────────────────────────────────────────────────

class GraphDTA(nn.Module):
    def __init__(self, drug_gnn_class, n_output=1, dropout=0.2):
        super().__init__()
        self.drug_gnn = drug_gnn_class()
        self.protein_cnn = ProteinCNN()

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, adj, target, mask = data.x, data.adj, data.target, data.mask
        target = target.view(-1, 1000)

        drug = self.drug_gnn(x, adj, mask)
        prot = self.protein_cnn(target)

        xc = torch.cat([drug, prot], dim=1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)


class GraphDTA_GIN(GraphDTA):
    def __init__(self, **kw): super().__init__(DrugGNN_GIN, **kw)

class GraphDTA_GCN(GraphDTA):
    def __init__(self, **kw): super().__init__(DrugGNN_GCN, **kw)

class GraphDTA_GAT(GraphDTA):
    def __init__(self, **kw): super().__init__(DrugGNN_GAT, **kw)

class GraphDTA_SAGE(GraphDTA):
    def __init__(self, **kw): super().__init__(DrugGNN_SAGE, **kw)
