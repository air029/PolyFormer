import numpy as np
import torch
import pickle
import argparse
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import sys
sys.path.append('..')
from dataloader.LINKX_dataset import LINKXDataset


def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
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


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

def get_idx_split(label, split_type='random', train_prop=.5, valid_prop=.25):
    """
    train_prop: The proportion of dataset for train split. Between 0 and 1.
    valid_prop: The proportion of dataset for validation split. Between 0 and 1.
    """
    if split_type == 'random':
        # ignore_negative = False if name == 'ogbn-proteins' else True
        ignore_negative = True
        train_idx, valid_idx, test_idx = rand_train_test_idx(
            label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)

    return train_idx, valid_idx, test_idx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="arxiv", help='datasets.')
parser.add_argument('--K', type=int, default=10, help='propagation steps.')
parser.add_argument('--splits_idx', type=int, default=0, help='splits_idx.')
parser.add_argument('--train_prop', type=float, default=0.5, help='train_prop.')
parser.add_argument('--valid_prop', type=float, default=0.25, help='valid_prop.')
parser.add_argument('--base', type=str, default='mono', help='base.', choices=['mono', 'cheb'])
args = parser.parse_args()
data_path="./data/"
dataset_path="./datasets/"

if args.dataset in ['pokec', 'twitch-gamer']:
    dataset = LINKXDataset(root=dataset_path, name=args.dataset.lower())
    print("Loading completed!")
    data = dataset[0]
    assert 0 <= args.splits_idx < 5
    if 'wiki' in args.dataset:
        train_idx, valid_idx, test_idx = get_idx_split(label=data.y.data, train_prop=args.train_prop, valid_prop=args.valid_prop)
    else:
        train_idx = dataset.train_mask.permute(1,0)[args.splits_idx]
        valid_idx = dataset.val_mask.permute(1,0)[args.splits_idx]
        test_idx = dataset.test_mask.permute(1,0)[args.splits_idx]
elif args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']:
    print("ogbn-data loading...")
    if args.dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=args.dataset, root = dataset_path)
    else:
        dataset = PygNodePropPredDataset(name=args.dataset, root = dataset_path)
    print("Loading completed!")
    data = dataset[0]
    args.splits_idx = 0
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
else:
    raise NotImplementedError

print(args.dataset, "Loading completed!")
edge_index = data.edge_index
N = data.num_nodes

labels = data.y.data
labels_train = labels[train_idx].reshape(-1).long()
labels_valid = labels[valid_idx].reshape(-1).long()
labels_test = labels[test_idx].reshape(-1).long()
with open(data_path+"labels_"+args.dataset+'_split_'+str(args.splits_idx)+".pickle","wb") as fopen:
    pickle.dump([labels_train,labels_valid,labels_test],fopen)
print("Labels have been saved!")


print('Making the graph undirected')
edge_index = to_undirected(edge_index)
# edge_index, _ = to_undirected(edge_index, None, N, None)

#Load edges and create adjacency
row,col = edge_index
row=row.numpy()
col=col.numpy()
adj_mat=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))

del row,col, dataset, edge_index


adj_mat = sys_normalized_adjacency_i(adj_mat)
adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
list_mat_train = []
list_mat_valid = []
list_mat_test = []

T_0_feat = data.x.numpy()
T_0_feat = torch.from_numpy(T_0_feat).float()

del data, labels

list_mat_train.append(T_0_feat[train_idx,:])
list_mat_valid.append(T_0_feat[valid_idx,:])
list_mat_test.append(T_0_feat[test_idx,:])

print("Begining of iteration!")
print("Done:",'0')

for i in range(1,args.K+1):
    T_1_feat = torch.spmm(adj_mat, T_0_feat)
    list_mat_train.append(T_1_feat[train_idx,:])
    list_mat_valid.append(T_1_feat[valid_idx,:])
    list_mat_test.append(T_1_feat[test_idx,:])
    T_0_feat = T_1_feat
    print("Done:",i)

with open(data_path+"training_"+args.dataset+'_split_'+str(args.splits_idx)+'_K_'+str(args.K)+"_mono.pickle","wb") as fopen:
    pickle.dump(list_mat_train,fopen)
with open(data_path+"validation_"+args.dataset+'_split_'+str(args.splits_idx)+'_K_'+str(args.K)+"_mono.pickle","wb") as fopen:
    pickle.dump(list_mat_valid,fopen)
with open(data_path+"test_"+args.dataset+'_split_'+str(args.splits_idx)+'_K_'+str(args.K)+"_mono.pickle","wb") as fopen:
    pickle.dump(list_mat_test,fopen)


print(args.dataset + " on " + args.base + " has been successfully processed")