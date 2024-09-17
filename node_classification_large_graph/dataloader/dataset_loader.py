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
    name = name.lower()
    root = './datasets/'
    if name in ['citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        dataset = HeterophilousGraphDataset(root=root, name=name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root=root, name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset

