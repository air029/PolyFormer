from ogb.nodeproppred import PygNodePropPredDataset
import pickle
import torch
import math
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import Coauthor
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz
from scipy import sparse as sp
import dgl
import time

def DataLoader(name):
    root = "./datasets/"
    name = name.lower()
    if name in ['chameleon_filtered', 'squirrel_filtered']:
        print('filtered dataset')
        dataset = HeteroDataset(root=root, name=name)
        if not is_undirected(dataset[0].edge_index):
            print('To undirected ...')
            pre_edge_index = dataset[0].edge_index
            dataset.data.edge_index = to_undirected(pre_edge_index)
        return dataset

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        dataset = HeterophilousGraphDataset(root=root, name=name)
        if name in ["minesweeper", "tolokers", "questions"]:
            # args.num_classes = 1
            pass
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root=root, name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset



class HeteroDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        # return osp.join(self.root, self.name, 'raw')
        return osp.join(self.root)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        data = np.load(self.raw_paths[0])
        x = torch.tensor(data['node_features'])
        y = torch.tensor(data['node_labels'])
        edge_index = torch.tensor(data['edges']).T
        train_mask = torch.tensor(data['train_masks']).to(torch.bool)
        val_mask = torch.tensor(data['val_masks']).to(torch.bool)
        test_mask = torch.tensor(data['test_masks']).to(torch.bool)
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
