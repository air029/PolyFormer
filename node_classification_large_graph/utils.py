import torch
import math
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.special import comb
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from dataloader.dataset_loader import DataLoader
from torch_geometric.utils import is_undirected, to_undirected, add_self_loops, remove_self_loops, contains_isolated_nodes

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def take_rest(x, y):
    x.sort()
    y.sort()
    res = []
    j, jmax = 0, len(y)
    for i in range(0, len(x)):
        flag = False
        while j < jmax and y[j] <= x[i]:
            if y[j] == x[i]:
                flag = True
            j += 1
        if not flag:
            res.append(x[i])
    return res

def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx=[i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx = take_rest(rest_index, val_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def heter_fixed_splits(dataset, data, idx_run):
    data.train_mask = dataset[0].train_mask.permute(1,0)[idx_run]
    data.val_mask = dataset[0].val_mask.permute(1,0)[idx_run]
    data.test_mask = dataset[0].test_mask.permute(1,0)[idx_run]
    print(data.test_mask.shape)
    return data

def random_splits_miss(data, num_classes, train_prop=.6, valid_prop=.2, seed=42):
    index = np.where(data.y != -1)[0]
    percls_trn  = int(round(0.6*len(index)/num_classes))
    val_lb      = int(round(0.2*len(index)))
    
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx=[i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx = take_rest(rest_index, val_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def fixed_splits(data, num_classes, percls_trn, val_lb, name):
    seed=42
    if name in ["Chameleon","Squirrel", "Actor"]:
        seed = 1941488137
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data

def random_splits_citation(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

def FetchDenseAdj(args, theta, data):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    N = len(data.y)
    coe_tmp=F.relu(torch.tensor(theta))
    coe=coe_tmp.clone()
    for i in range(args.K+1):
        coe[i]=coe_tmp[0]*cheby(i,math.cos((args.K+0.5)*math.pi/(args.K+1)))
        for j in range(1,args.K+1):
            x_j=math.cos((args.K-j+0.5)*math.pi/(args.K+1))
            coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
        coe[i]=2*coe[i]/(args.K+1)

    deg = torch.zeros((N, N)).to(device)
    adj = torch.zeros((N, N)).to(device)
    Lap = torch.zeros((N, N)).to(device)
    out = torch.zeros((N, N)).to(device)
    u, v = data.edge_index[0].to(device), data.edge_index[1].to(device)
    adj[u, v]+=1
    deg = torch.diag_embed(adj.sum(dim=0)**(-1/2))
    #adj = torch.where(torch.isnan(adj), torch.full_like(adj, 0), adj)
    deg = torch.where(torch.isinf(deg), torch.full_like(deg, 0), deg)

    # Lap = L / 2lambdamax - I \approx L - I
    Lap = torch.mm(deg, adj)
    Lap = torch.mm(Lap, deg)
    Lap = - Lap
    T_0 = torch.eye(N).to(device)
    T_1 = Lap.clone()

    out=coe[0]/2*T_0+coe[1]*T_1

    for i in range(2,args.K+1):
        T_2=2*torch.mm(T_1, Lap)-T_0
        out=out+coe[i]*T_2
        T_0, T_1 = T_1, T_2

    print("DenseRate:", (out.abs() > args.sparse_threshold).sum() / (N * N))
    return out

    
    '''
    T_1 = Lap.clone()
    
    for para in theta:
        sum += para * T_0
        T_2 = 2 * torch.mm(T_1, Lap) - T_0
        T_0, T_1 = T_1, T_2
    '''

    '''
    for para in theta:
        sum += para * T_0
        T_0 = torch.mm(T_0, Lap)
    '''
    '''
    adj = torch.mm(torch.mm(deg, adj), deg)
    for para in theta:
        sum += para * T_0
        T_0 = torch.mm(T_0, adj)
    '''
    return sum

def FetchPolyCoefficients(args, theta):
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    coe_tmp=F.relu(torch.tensor(theta))
    coe=coe_tmp.clone()
    for i in range(args.K+1):
        coe[i]=coe_tmp[0]*cheby(i,math.cos((args.K+0.5)*math.pi/(args.K+1)))
        for j in range(1,args.K+1):
            x_j=math.cos((args.K-j+0.5)*math.pi/(args.K+1))
            coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
        coe[i]=2*coe[i]/(args.K+1)
    
    T_0 = torch.zeros(args.K+1).to(device)
    T_0[0] = 1
    T_1 = torch.zeros(args.K+1).to(device)
    T_1[1] = 1
    T_2 = torch.zeros(args.K+1).to(device)
    res = torch.zeros(args.K+1).to(device)
    res[0] += coe[0] / 2
    res[1] += coe[1] * -1
    #print(coe[0], coe[1])
    for i in range(2, args.K+1):
        T_2 = -T_0
        for j in range(0, args.K):
            T_2[j+1] += 2 * T_1[j]
        for j in range(0, args.K+1):
            res[j] += coe[i] * T_2[j] * ((-1) ** j)
        T_0, T_1 = T_1, T_2
    return res

def DataSplit(args, data_smp, data, denseAdj, batch_mask, device):
    batch_size = len(batch_mask)
    data_smp.train_mask = data.train_mask[batch_mask]
    data_smp.val_mask   = data.val_mask[batch_mask]
    data_smp.test_mask  = data.test_mask[batch_mask]
    edge_index_smp = [[], []]
    for i in range(batch_size):
        edge_index_smp[0] += [i] * batch_size
    edge_index_smp[1] = [i for i in range(batch_size)] * batch_size
    edge_index_smp = torch.tensor(edge_index_smp).to(device)
    data_smp.edge_index = edge_index_smp
    data_smp.x = data.x[batch_mask]
    data_smp.y = data.y[batch_mask]
    data_smp.denseAdj = (denseAdj[batch_mask])[:, batch_mask].flatten()

    if args.sparse_threshold > 0:  
        mask_smp = data_smp.denseAdj.abs() > args.sparse_threshold
        data_smp.edge_index = data_smp.edge_index[:, mask_smp]
        data_smp.denseAdj   = data_smp.denseAdj[mask_smp]

    return data_smp

def batch_generator(tsplit, train_mask, N, batchnum):
    res = []
    if not tsplit:
        data = torch.arange(0, N)
    else:
        data = torch.arange(0, N)[train_mask]
        N = len(data)
    
    perm = torch.randperm(N)
    batch_size = N // batchnum
    begins = 0
    ends = begins + batch_size
    while(1):
        if(ends + batch_size / 2 > N):
            res.append(data[perm[begins:]])
            return res
        else:
            res.append(data[perm[begins:ends]])
            begins = ends
            ends = begins + batch_size


def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   #adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def init_temp(base_name, K):  
    if base_name == 'mono':
        bound = np.sqrt(3/(K))
        TEMP = np.random.uniform(-bound, bound, K)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        temp = torch.tensor(TEMP).float()
    elif base_name == 'cheb':
        temp = torch.zeros(K).float()   
        # temp.data[0]=1.0
        temp.data.fill_(1.0)
    else:
        assert False, 'base_name error'
    return temp

import os
import pickle

def load_base(dataname, base_name, K, x, edge_index, edge_attr=None):
    file_path = './bases/' + dataname + '_' + base_name + '_' + str(K) + '.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            list_mat = pickle.load(f)
    else:
        list_mat = get_base(base_name, K, x, edge_index, edge_attr)
        with open(file_path, 'wb') as f:
            pickle.dump(list_mat, f)
    return list_mat

def get_base(base_name, K, x, edge_index, edge_attr=None):
    if base_name == 'mono':
        list_mat = mono_base(K, x, edge_index, edge_attr)
    elif base_name == 'cheb':
        list_mat = cheb_base(K, x, edge_index, edge_attr)
    return list_mat

def mono_base(K, x, edge_index, edge_attr):
    edge_index, norm = gcn_norm(edge_index, edge_attr, num_nodes=x.size(0), dtype=x.dtype)
    adj = to_scipy_sparse_matrix(edge_index, norm, x.size(0))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    device = x.device
    adj = adj.to(device)

    list_mat = []
    list_mat.append(x)
    tmp_mat = x
    for _ in range(K):
        tmp_mat = torch.spmm(adj, tmp_mat)
        list_mat.append(tmp_mat)
    return list_mat

def cheb_base(K, x, edge_index, edge_attr):
    # self.temp.data.fill_(0.0)
    # self.temp.data[0]=1.0
    # coe[i]/i**self.q #The positive constant
    node_dim = 0
    #L=I-D^(-0.5)AD^(-0.5)
    edge_index1, norm1 = get_laplacian(edge_index, edge_attr,normalization='sym', dtype=x.dtype, num_nodes=x.size(node_dim))

    #L_tilde=L-I
    edge_index_tilde, norm_tilde = add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(node_dim))
    L_tilde = to_scipy_sparse_matrix(edge_index_tilde, norm_tilde, x.size(node_dim))
    L_tilde = sparse_mx_to_torch_sparse_tensor(L_tilde)

    device = x.device
    L_tilde = L_tilde.to(device)
    
    list_mat = []
    Tx_0=x
    list_mat.append(Tx_0)
    # Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)
    Tx_1 = torch.spmm(L_tilde, x)
    list_mat.append(Tx_1)

    for i in range(2, K+1):
        # Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
        Tx_2 = 2 * torch.spmm(L_tilde, Tx_1) - Tx_0
        list_mat.append(Tx_2)
        Tx_0, Tx_1 = Tx_1, Tx_2

    return list_mat # K + 1

def get_data_load(args):
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
        args.num_classes = 1
    else:
        args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
    assert is_undirected(data.edge_index)

    data.list_mat = load_base(args.dataset.lower(), args.base, args.K, data.x, data.edge_index, data.edge_attr)
    args.n_channel = args.hidden
    return dataset, data 
