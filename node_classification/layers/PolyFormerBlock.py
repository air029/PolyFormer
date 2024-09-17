from typing import Callable, Optional, Union
import torch as th
from torch.nn.modules.module import Module, _grad_t
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from utils import cheby, init_temp

    
class PolyAttn(nn.Module):
    def __init__(self, dataset, args):
        super(PolyAttn, self).__init__()
        self.K = args.K + 1
        self.base = args.base
        self.norm = nn.LayerNorm(args.hidden)
        self.n_head = args.n_head
        self.multi = args.multi
        self.d_head = args.hidden // args.n_head
        
        self.token_wise_network = nn.ModuleList([nn.Sequential(
            nn.Linear(args.hidden, int(args.hidden * self.multi)),
            nn.ReLU(),
            nn.Linear(int(args.hidden * self.multi), args.hidden)
        ) for _ in range(self.K)])

        self.W_Q = nn.Linear(args.hidden, self.n_head * self.d_head, bias=False)
        self.W_K = nn.Linear(args.hidden, self.n_head * self.d_head, bias=False)

        self.bias_scale = nn.Parameter(torch.ones(self.n_head, self.K))
        self.bias = torch.tensor([((j+1) ** args.q)**(-1) for j in range(self.K)])

        self.dprate = args.dprate
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.token_wise_network:
            layer[0].reset_parameters()
            layer[2].reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()

    def forward(self, src):
        batch_size = src.shape[0]
        origin_src = src
        src = self.norm(src)
        token = src
        value = src
        token = torch.stack([layer(token[:,idx,:]) for idx, layer in enumerate(self.token_wise_network)],dim=1)
        query = self.W_Q(token)
        key = self.W_K(token)
        q_heads = query.view(query.size(0), query.size(1), self.n_head, self.d_head).transpose(1, 2) # [n,n_head,k,d_head]
        k_heads = key.view(key.size(0), key.size(1), self.n_head, self.d_head).transpose(1, 2)
        v_heads = value.view(value.size(0), value.size(1), self.n_head, -1).transpose(1, 2)
        attention_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head).float())
        attention_scores = torch.tanh(attention_scores)
        attn_mask = torch.einsum('hk,k->hk', self.bias_scale, self.bias.to(src.device))
        attention_scores = torch.einsum('nhdk,hk->nhdk', attention_scores, attn_mask)
        attention_scores = F.dropout(attention_scores, p=self.dprate, training=self.training)
        context_heads = torch.matmul(attention_scores, v_heads)
        context_sequence = context_heads.transpose(1, 2).contiguous().view(batch_size, self.K, -1)
        src = F.dropout(context_sequence, p=self.dprate, training=self.training)
        src = src + origin_src
        return src

class FFNNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FFNNetwork, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(ffn_dim, hidden_dim)
        self.reset_parameters()
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x
     
class FFN(nn.Module):
    def __init__(self, dataset, args):
        super(FFN, self).__init__()
        self.K = args.K + 1
        self.base = args.base
        self.dropout = args.dprate
        self.ffn_norm = nn.LayerNorm(args.hidden)
        self.ffn_net = FFNNetwork(args.hidden, args.d_ffn)

    def forward(self, src):
        origin_src = src
        src = self.ffn_norm(src)
        src = self.ffn_net(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = src + origin_src
        return src
    
class PolyFormerBlock(nn.Module):
    def __init__(self, dataset, args):
        super(PolyFormerBlock, self).__init__()
        self.K = args.K + 1
        self.base = args.base

        self.attnmodule = PolyAttn(dataset, args)
        self.ffnmodule = FFN(dataset, args)

    def forward(self, src):
        src = self.attnmodule(src)
        src = self.ffnmodule(src)
        return src

    
    