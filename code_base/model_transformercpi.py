import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        q = q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        conved = conv_input
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        return conved.permute(0, 2, 1)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x.permute(0, 2, 1)


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ft(trg)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        trg = torch.squeeze(trg, dim=0)
        norm = torch.squeeze(norm, dim=0)
        pooled = torch.zeros((self.hid_dim,), device=src.device)
        for i in range(norm.shape[0]):
            pooled += trg[i] * norm[i]
        pooled = pooled.unsqueeze(0)
        label = F.relu(self.fc_1(pooled))
        return self.fc_2(label)


class TransformerCPI(nn.Module):
    def __init__(
        self,
        protein_dim=100,
        atom_dim=34,
        hid_dim=64,
        n_layers=3,
        n_heads=8,
        pf_dim=256,
        dropout=0.1,
        kernel_size=5,
        device='cpu',
    ):
        super().__init__()
        self.encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
        self.decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, inputs, adj):
        support = torch.mm(inputs, self.weight)
        return torch.mm(adj, support)

    def forward(self, compound, adj, protein):
        compound = self.gcn(compound, adj)
        compound = torch.unsqueeze(compound, dim=0)
        protein = torch.unsqueeze(protein, dim=0)
        enc_src = self.encoder(protein)
        return self.decoder(compound, enc_src)

    def predict_step(self, compound, adj, protein):
        logits = self.forward(compound, adj, protein)
        probs = F.softmax(logits, dim=1)
        predicted_label = int(np.argmax(probs.detach().cpu().numpy()))
        predicted_score = float(probs[0, 1].detach().cpu().item())
        return logits, predicted_label, predicted_score
