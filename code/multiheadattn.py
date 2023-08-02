import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 
 
class MultiheadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, d_model, n_head, output_dim):
        super().__init__()
        self.n_head = n_head

        self.w_qs0 = nn.Conv3d(in_channels=7, out_channels=7, kernel_size=[12,12,1], 
                             stride=[12,12,1], padding=[0,0,0],groups=7, bias=False)
        self.w_ks0 = nn.Conv3d(in_channels=7, out_channels=7, kernel_size=[12,12,1], 
                             stride=[12,12,1], padding=[0,0,0],groups=7, bias=False)
        self.w_vs0 = nn.Linear(d_model, d_model, bias=False)

        self.w_qs1 = nn.Conv3d(in_channels=7, out_channels=7, kernel_size=[12,12,1], 
                             stride=[12,12,1], padding=[0,0,0],groups=7, bias=False)
        self.w_ks1 = nn.Conv3d(in_channels=7, out_channels=7, kernel_size=[12,12,1], 
                             stride=[12,12,1], padding=[0,0,0],groups=7, bias=False)
        self.w_vs1 = nn.Linear(d_model, d_model, bias=False)

        self.temperature = (d_model) ** 0.5
        self.fc = nn.Linear(n_head*d_model, output_dim)
        real_dropout=0.1
        self.dropout = nn.Dropout(real_dropout)
        self.dropout1 = nn.Dropout(real_dropout)
 
    def forward(self, q, k, v, mask=None):
        B, N, T, C, D = v.shape

        q0 = self.w_qs0(q).view(B,N,-1)
        k0 = self.w_ks0(k).view(B,N,-1)
        v0 = self.w_vs0(v)
        attn0 = torch.matmul(q0 / self.temperature, k0.transpose(1, 2))
        if mask is not None:
            attn0 = attn0.masked_fill(mask == 0, -1e9)
        attention0 = self.dropout(F.softmax(attn0, dim=-1))
        out0 = torch.einsum("bst,btnmd->bsnmd", [attention0, v0])

        q1 = self.w_qs1(q).view(B,N,-1)
        k1 = self.w_ks1(k).view(B,N,-1)
        v1 = self.w_vs1(v)
        attn1 = torch.matmul(q1 / self.temperature, k1.transpose(1, 2))
        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, -1e9)
        attention1 = self.dropout1(F.softmax(attn1, dim=-1))
        out1 = torch.einsum("bst,btnmd->bsnmd", [attention1, v1])

        res = self.fc(torch.cat((out0,out1),dim=-1))
        return res
