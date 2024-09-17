import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
from scipy.fftpack import shift
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.utils.undirected import is_undirected, to_undirected
from layers.PolyFormerBlock import PolyFormerBlock
from utils import init_temp, cheby
import os
    
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



class FeedForwardModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x
    

class PolyFormer(nn.Module): 
    def __init__(self, dataset, args):
        super(PolyFormer,self).__init__()
        self.dropout = args.dropout
        self.nlayers = args.nlayer
        self.dataset = args.dataset
        
        self.attn = nn.ModuleList([PolyFormerBlock(dataset, args) for _ in range(self.nlayers)])
        self.K = args.K + 1
        self.base = args.base

        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.hidden)
        self.lin3 = Linear(args.hidden, args.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x_lis,st=0,end=0):
        input_mat = x_lis[:self.K]
        input_mat = torch.stack(input_mat, dim = 1)[st:end,:] # [N,k,d]
        input_mat = self.lin1(input_mat) # just for common dataset
       
        for block in self.attn:
            input_mat = block(input_mat)
            
        x = torch.sum(input_mat, dim = 1) # [N,d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        if self.dataset == 'genius':
            return x
        return F.log_softmax(x, dim=1)
    
