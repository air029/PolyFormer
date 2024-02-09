
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros
from layers.PolyFormerBlock import PolyFormerBlock


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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

    def forward(self, data):
        input_mat = data.list_mat
        input_mat = torch.stack(input_mat, dim = 1) # [N,k,d]
        input_mat = self.lin1(input_mat) # just for common dataset
       
        for block in self.attn:
            input_mat = block(input_mat)
            
        x = torch.sum(input_mat, dim = 1) # [N,d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return x


