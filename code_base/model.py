import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool, DenseGINConv, DenseGCNConv, DenseGATConv, DenseSAGEConv
from torch_geometric.nn import GraphNorm

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = self.softmax(scores)
        attended_values = torch.matmul(attention_weights, value)
        return attended_values
class Attention_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention_Fusion, self).__init__()
        self.project_mol = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_mol_vir = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_target_vir = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_target = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, mol, mol_vir, target_vir, target):
        mol = self.project_mol(mol)
        mol_vir = self.project_mol_vir(mol_vir)
        target_vir = self.project_target_vir(target_vir)
        target = self.project_target(target)
        a = torch.cat((mol, mol_vir, target_vir, target), 1)
        a = torch.softmax(a, dim=1)
        return a
class GNN_GIN_local(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer=1):
        super(GNN_GIN_local, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if layer==1:
            nn2 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
            self.convs.append(DenseGINConv(nn2))
            self.bns.append(GraphNorm(out_channels))
        if layer==2:
            nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(DenseGINConv(nn1))
            self.bns.append(GraphNorm(hidden_channels))
            nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
            self.convs.append(DenseGINConv(nn2))
            self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x


class GNN_GIN_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_GIN_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(DenseGINConv(nn1))
        self.bns.append(GraphNorm(hidden_channels))
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(DenseGINConv(nn2))
        self.bns.append(GraphNorm(hidden_channels))
        nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(DenseGINConv(nn3))
        self.bns.append(GraphNorm(hidden_channels))
        nn4 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(DenseGINConv(nn4))
        self.bns.append(GraphNorm(hidden_channels))
        nn5 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.convs.append(DenseGINConv(nn5))
        self.bns.append(GraphNorm(out_channels))
        self.mol_bias = nn.Parameter(torch.rand(1, 64))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, mol_x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](mol_x, adj, mask))
            x = self.bns[step](x)
            if step == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.fc1(x) + self.fc2(mol_x) + self.mol_bias.expand(mol_x.size(0), mol_x.size(1), mol_x.size(2)))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        return mol_x
class GNN_add_GIN_pool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_add_GIN_pool, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.convs.append(DenseGINConv(nn1))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x

# GINConv model
class Diff_DTA_GIN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25, embed_dim=128, output_dim=128,
                 dropout=0.2,num_nodes_1=4,num_nodes_2 = 2):
        super(Diff_DTA_GIN, self).__init__()
        self.gnn1_pool = GNN_GIN_local(num_features_xd, 64, num_nodes_1, layer=2)
        self.gnn1_embed = GNN_GIN_local(num_features_xd, 64, 64, layer=2)
        self.gnn2_pool = GNN_GIN_local(64, 64, num_nodes_2, layer=1)
        self.gnn2_embed = GNN_GIN_local(64, 64, 64, layer=1)

        self.gnn3_embed = GNN_GIN_local(64, 64, 64, layer=2)

        self.lin1 = torch.nn.Linear(num_nodes_1 * 64, 128)
        self.lin2 = torch.nn.Linear(num_nodes_2 * 64, 128)
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.input_dim = 128
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=(kernel_size, self.input_dim))
            for kernel_size in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1_xt = nn.Linear(len(self.kernel_sizes) * 128, output_dim)
        # combined layers
        self.fc1 = nn.Linear(128 * 4, 1024)
        self.bns1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bns2 = torch.nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

        self.global_gnn_pool = GNN_add_GIN_pool(num_features_xd, 64, 1)
        self.global_gnn = GNN_GIN_global(num_features_xd, 64, 64)
        self.lin3 = torch.nn.Linear(64, 128)
        self.lin4 = torch.nn.Linear(128 * 2, 128)
        self.attention1 = SelfAttention(dim=64)
        self.attention2 = SelfAttention(dim=64)
        self.attention3 = SelfAttention(dim=64)
        self.attention = Attention_Fusion(128)
    def forward(self, data):
        x, adj, target, mask = data.x, data.adj, data.target, data.mask
        target = target.view(-1, 1000)
        s_global = self.global_gnn_pool(x, adj, mask)
        x_global = self.global_gnn(x, adj, mask)
        x_global = self.attention3(x_global)
        x_global, _, _, _ = dense_diff_pool(x_global, adj, s_global, mask)
        x_global = x_global.view(-1, 64)
        x_global = F.relu(self.lin3(x_global))
        x_global = F.dropout(x_global, p=0.2, training=self.training)
        s = self.gnn1_pool(x, adj, mask)
        x_1 = self.gnn1_embed(x, adj, mask)
        x_1, adj, l1, e1 = dense_diff_pool(x_1, adj, s, mask)
        s = self.gnn2_pool(x_1, adj)
        x_2 = self.gnn2_embed(x_1, adj)

        x_2, adj, l2, e2 = dense_diff_pool(x_2, adj, s)

        # attention
        x_1 = self.attention1(x_1)
        x_2 = self.attention2(x_2)

        # flatten
        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_1 = F.relu(self.lin1(x_1))
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_2 = F.relu(self.lin2(x_2))
        x_2 = F.dropout(x_2, p=0.2, training=self.training)

        # x_3 = F.relu(self.lin4(torch.cat([x_1,x_2], dim=1)))
        embedded_xt = self.embedding_xt(target)
        # embedded_xt = self.attention_embed(embedded_xt)
        embedded_xt = embedded_xt.unsqueeze(1)
        embedded_xt = [nn.functional.relu(conv(embedded_xt)).squeeze(3) for conv in self.convs]
        embedded_xt = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in embedded_xt]
        conv_xt = torch.cat(embedded_xt, dim=1)
        xt = conv_xt.reshape(-1, len(self.kernel_sizes) * 128)
        xt = F.dropout(xt, p=0.2, training=self.training)
        # flatten
        # xt = conv_xt.reshape(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        a = self.attention(x_1, x_2, xt, x_global)
        emb = torch.stack([x_1, x_2, xt, x_global], dim=1)
        a = a.unsqueeze(dim=2)
        xc = (a * emb).reshape(-1, 4 * 128)
        # concat
        # xc = torch.cat((x_1, x_2, xt, x_global), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.bns1(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)
        # xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.bns2(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)

        out = self.out(xc)
        return out, l1 + l2, e1 + e2, a, x_2, x_global
# GCN

class GNN_GCN_local(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN_GCN_local, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(in_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x


class GNN_GCN_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_GCN_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(in_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))
        self.mol_bias = nn.Parameter(torch.rand(1, 64))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, mol_x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](mol_x, adj, mask))
            x = self.bns[step](x)
            if step == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.fc1(x) + self.fc2(mol_x) + self.mol_bias.expand(mol_x.size(0), mol_x.size(1), mol_x.size(2)))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        return mol_x
class GNN_add_GCN_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_add_GCN_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(in_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x

class Diff_DTA_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,num_nodes_1 = 4,num_nodes_2 = 2,
                 embed_dim=128, output_dim=128, dropout=0.2):
        super(Diff_DTA_GCN, self).__init__()
        self.gnn1_pool = GNN_GCN_local(num_features_xd, num_nodes_1)
        self.gnn1_embed = GNN_GCN_local(num_features_xd, 64)
        self.gnn2_pool = GNN_GCN_local(64, num_nodes_2)
        self.gnn2_embed = GNN_GCN_local(64, 64)

        self.lin1 = torch.nn.Linear(num_nodes_1 * 64, 128)
        self.lin2 = torch.nn.Linear(num_nodes_2 * 64, 128)
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.input_dim = 128
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=(kernel_size, self.input_dim))
            for kernel_size in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1_xt = nn.Linear(len(self.kernel_sizes) * 128, output_dim)
        # combined layers
        self.fc1 = nn.Linear(128 * 4, 1024)
        self.bns1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bns2 = torch.nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

        self.global_gnn_pool = GNN_add_GCN_global(num_features_xd, 64, 1)
        self.global_gnn = GNN_GCN_global(num_features_xd, 64, 64)
        self.lin3 = torch.nn.Linear(64, 128)

        self.attention1 = SelfAttention(dim=64)
        self.attention2 = SelfAttention(dim=64)
        self.attention3 = SelfAttention(dim=64)
        self.attention = Attention_Fusion(128)
    def forward(self, data):
        x, adj, target, mask = data.x, data.adj, data.target, data.mask
        target = target.view(-1, 1000)
        s_global = self.global_gnn_pool(x, adj, mask)
        x_global = self.global_gnn(x, adj, mask)
        x_global = self.attention3(x_global)
        x_global, _, _, _ = dense_diff_pool(x_global, adj, s_global, mask)
        x_global = x_global.view(-1, 64)
        x_global = F.relu(self.lin3(x_global))
        x_global = F.dropout(x_global, p=0.2, training=self.training)
        s = self.gnn1_pool(x, adj, mask)
        x_1 = self.gnn1_embed(x, adj, mask)
        x_1, adj, l1, e1 = dense_diff_pool(x_1, adj, s, mask)
        s = self.gnn2_pool(x_1, adj)
        x_2 = self.gnn2_embed(x_1, adj)

        x_2, adj, l2, e2 = dense_diff_pool(x_2, adj, s)
        # attention
        x_1 = self.attention1(x_1)
        x_2 = self.attention2(x_2)
        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_1 = F.relu(self.lin1(x_1))
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_2 = F.relu(self.lin2(x_2))
        x_2 = F.dropout(x_2, p=0.2, training=self.training)
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.unsqueeze(1)
        embedded_xt = [nn.functional.relu(conv(embedded_xt)).squeeze(3) for conv in self.convs]
        embedded_xt = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in embedded_xt]
        conv_xt = torch.cat(embedded_xt, dim=1)
        xt = conv_xt.reshape(-1, len(self.kernel_sizes) * 128)
        xt = F.dropout(xt, p=0.2, training=self.training)
        # flatten
        # xt = conv_xt.reshape(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        a = self.attention(x_1, x_2, xt, x_global)
        emb = torch.stack([x_1, x_2, xt, x_global], dim=1)
        a = a.unsqueeze(dim=2)
        xc = (a * emb).reshape(-1, 4 * 128)
        # concat
        # xc = torch.cat((x_1, x_2, xt, x_global), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.bns1(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)
        # xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.bns2(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)

        out = self.out(xc)
        return out, l1 + l2, e1 + e2, a, x_2, x_global

# GAT

class GNN_GAT_local(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN_GAT_local, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGATConv(in_channels,out_channels,concat=False,heads=5))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x


class GNN_GAT_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_GAT_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGATConv(in_channels,hidden_channels,concat=False,heads=3))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGATConv(hidden_channels,hidden_channels,concat=False,heads=3))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGATConv(hidden_channels,hidden_channels,concat=False,heads=3))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGATConv(hidden_channels,hidden_channels,concat=False,heads=3))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseGATConv(hidden_channels,out_channels,concat=False,heads=2))
        self.bns.append(GraphNorm(out_channels))
        self.mol_bias = nn.Parameter(torch.rand(1, 64))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, mol_x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](mol_x, adj, mask))
            x = self.bns[step](x)
            if step == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.fc1(x) + self.fc2(mol_x) + self.mol_bias.expand(mol_x.size(0), mol_x.size(1), mol_x.size(2)))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        return mol_x
class GNN_add_GAT_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_add_GAT_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseGATConv(in_channels,out_channels,concat=False,heads=1))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x

class Diff_DTA_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,num_nodes_1 = 4,num_nodes_2 = 2,
                 embed_dim=128, output_dim=128, dropout=0.2):
        super(Diff_DTA_GAT, self).__init__()
        self.gnn1_pool = GNN_GAT_local(num_features_xd, num_nodes_1)
        self.gnn1_embed = GNN_GAT_local(num_features_xd, 64)
        self.gnn2_pool = GNN_GAT_local(64, num_nodes_2)
        self.gnn2_embed = GNN_GAT_local(64, 64)

        self.lin1 = torch.nn.Linear(num_nodes_1 * 64, 128)
        self.lin2 = torch.nn.Linear(num_nodes_2 * 64, 128)
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.input_dim = 128
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=(kernel_size, self.input_dim))
            for kernel_size in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1_xt = nn.Linear(len(self.kernel_sizes) * 128, output_dim)
        # combined layers
        self.fc1 = nn.Linear(128 * 4, 1024)
        self.bns1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bns2 = torch.nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

        self.global_gnn_pool = GNN_add_GAT_global(num_features_xd, 64, 1)
        self.global_gnn = GNN_GAT_global(num_features_xd, 64, 64)
        self.lin3 = torch.nn.Linear(64, 128)
        self.attention1 = SelfAttention(dim=64)
        self.attention2 = SelfAttention(dim=64)
        self.attention3 = SelfAttention(dim=64)
        self.attention = Attention_Fusion(128)
    def forward(self, data):
        x, adj, target, mask = data.x, data.adj, data.target, data.mask
        target = target.view(-1, 1000)
        s_global = self.global_gnn_pool(x, adj, mask)
        x_global = self.global_gnn(x, adj, mask)
        x_global = self.attention3(x_global)
        x_global, _, _, _ = dense_diff_pool(x_global, adj, s_global, mask)
        x_global = x_global.view(-1, 64)
        x_global = F.relu(self.lin3(x_global))
        x_global = F.dropout(x_global, p=0.2, training=self.training)
        s = self.gnn1_pool(x, adj, mask)
        x_1 = self.gnn1_embed(x, adj, mask)
        x_1, adj, l1, e1 = dense_diff_pool(x_1, adj, s, mask)
        s = self.gnn2_pool(x_1, adj)
        x_2 = self.gnn2_embed(x_1, adj)

        x_2, adj, l2, e2 = dense_diff_pool(x_2, adj, s)
        # attention
        x_1 = self.attention1(x_1)
        x_2 = self.attention2(x_2)
        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_1 = F.relu(self.lin1(x_1))
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_2 = F.relu(self.lin2(x_2))
        x_2 = F.dropout(x_2, p=0.2, training=self.training)
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.unsqueeze(1)
        embedded_xt = [nn.functional.relu(conv(embedded_xt)).squeeze(3) for conv in self.convs]
        embedded_xt = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in embedded_xt]
        conv_xt = torch.cat(embedded_xt, dim=1)
        xt = conv_xt.reshape(-1, len(self.kernel_sizes) * 128)
        xt = F.dropout(xt, p=0.2, training=self.training)
        # flatten
        # xt = conv_xt.reshape(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        a = self.attention(x_1, x_2, xt, x_global)
        emb = torch.stack([x_1, x_2, xt, x_global], dim=1)
        a = a.unsqueeze(dim=2)
        xc = (a * emb).reshape(-1, 4 * 128)
        # concat
        # xc = torch.cat((x_1, x_2, xt, x_global), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.bns1(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)
        # xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.bns2(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)

        out = self.out(xc)
        return out, l1 + l2, e1 + e2, a, x_2, x_global
class GNN_SAGE_local(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN_SAGE_local, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseSAGEConv(in_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x


class GNN_SAGE_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_SAGE_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseSAGEConv(in_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseSAGEConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(hidden_channels))
        self.convs.append(DenseSAGEConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(out_channels))
        self.convs.append(DenseSAGEConv(hidden_channels,hidden_channels))
        self.bns.append(GraphNorm(out_channels))
        self.convs.append(DenseSAGEConv(hidden_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))
        self.mol_bias = nn.Parameter(torch.rand(1, 64))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, mol_x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](mol_x, adj, mask))
            x = self.bns[step](x)
            if step == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.fc1(x) + self.fc2(mol_x) + self.mol_bias.expand(mol_x.size(0), mol_x.size(1), mol_x.size(2)))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        return mol_x
class GNN_add_SAGE_global(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_add_SAGE_global, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(DenseSAGEConv(in_channels,out_channels))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x)
        return x

# GINConv model
class Diff_DTA_SAGE(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25, num_nodes_1=4, num_nodes_2=2,
                 embed_dim=128, output_dim=128, dropout=0.2):
        super(Diff_DTA_SAGE, self).__init__()
        self.gnn1_pool = GNN_SAGE_local(num_features_xd, num_nodes_1)
        self.gnn1_embed = GNN_SAGE_local(num_features_xd, 64)
        self.gnn2_pool = GNN_SAGE_local(64, num_nodes_2)
        self.gnn2_embed = GNN_SAGE_local(64, 64)
        self.lin1 = torch.nn.Linear(num_nodes_1 * 64, 128)
        self.lin2 = torch.nn.Linear(num_nodes_2 * 64, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.input_dim = 128
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=(kernel_size, self.input_dim))
            for kernel_size in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1_xt = nn.Linear(len(self.kernel_sizes) * 128, output_dim)
        # combined layers
        self.fc1 = nn.Linear(128 * 4, 1024)
        self.bns1 = torch.nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bns2 = torch.nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

        self.global_gnn_pool = GNN_add_SAGE_global(num_features_xd, 64, 1)
        self.global_gnn = GNN_SAGE_global(num_features_xd, 64, 64)
        self.lin3 = torch.nn.Linear(64, 128)

        self.lin4 = torch.nn.Linear(128 * 2, 128)
        self.attention1 = SelfAttention(dim=64)
        self.attention2 = SelfAttention(dim=64)
        self.attention3 = SelfAttention(dim=64)
        self.attention = Attention_Fusion(128)
    def forward(self, data):
        x, adj, target, mask = data.x, data.adj, data.target, data.mask
        target = target.view(-1, 1000)
        s_global = self.global_gnn_pool(x, adj, mask)
        x_global = self.global_gnn(x, adj, mask)
        x_global = self.attention3(x_global)
        x_global, _, _, _ = dense_diff_pool(x_global, adj, s_global, mask)
        x_global = x_global.view(-1, 64)
        x_global = F.relu(self.lin3(x_global))
        x_global = F.dropout(x_global, p=0.2, training=self.training)
        s = self.gnn1_pool(x, adj, mask)
        x_1 = self.gnn1_embed(x, adj, mask)
        x_1, adj, l1, e1 = dense_diff_pool(x_1, adj, s, mask)
        s = self.gnn2_pool(x_1, adj)
        x_2 = self.gnn2_embed(x_1, adj)

        x_2, adj, l2, e2 = dense_diff_pool(x_2, adj, s)
        # attention
        x_1 = self.attention1(x_1)
        x_2 = self.attention2(x_2)
        x_1 = x_1.view(x_1.size(0), -1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_1 = F.relu(self.lin1(x_1))
        x_1 = F.dropout(x_1, p=0.2, training=self.training)
        x_2 = F.relu(self.lin2(x_2))
        x_2 = F.dropout(x_2, p=0.2, training=self.training)

        # x_3 = F.relu(self.lin4(torch.cat([x_1, x_2], dim=1)))
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.unsqueeze(1)
        embedded_xt = [nn.functional.relu(conv(embedded_xt)).squeeze(3) for conv in self.convs]
        embedded_xt = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in embedded_xt]
        conv_xt = torch.cat(embedded_xt, dim=1)
        xt = conv_xt.reshape(-1, len(self.kernel_sizes) * 128)
        xt = F.dropout(xt, p=0.2, training=self.training)
        # flatten
        xt = self.fc1_xt(xt)
        a = self.attention(x_1, x_2, xt, x_global)
        emb = torch.stack([x_1, x_2, xt, x_global], dim=1)
        a = a.unsqueeze(dim=2)
        xc = (a * emb).reshape(-1, 4 * 128)
        xc = self.fc1(xc)
        xc = self.bns1(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)
        xc = self.fc2(xc)
        xc = self.bns2(xc)
        xc = self.relu(xc)
        xc = F.dropout(xc, p=0.2, training=self.training)

        out = self.out(xc)
        return out, l1 + l2, e1 + e2, a, x_2, x_global


