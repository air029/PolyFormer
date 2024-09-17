import argparse
from utils import random_splits, heter_fixed_splits, hetergraph_fixed_split, get_data_load
from mymodels import PolyFormer
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from sklearn.metrics import roc_auc_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import cheby, FetchDenseAdj, FetchPolyCoefficients, DataSplit
from torch_sparse import coalesce
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.utils import to_networkx, add_self_loops
from torch import pca_lowrank, matmul
from torch_geometric.data import Data
from scipy.special import comb
import sys
import os

def RunExp(args, dataset, data, Net):

    args.device = torch.device('cuda:'+str(args.device_idx) if torch.cuda.is_available() else 'cpu')

    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        logits = model(data)[data.train_mask]
        if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), data.y[data.train_mask].to(torch.float))
        else:
            loss = F.cross_entropy(logits, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        del logits
        torch.cuda.empty_cache()

    def test(model, data):
        with torch.no_grad():
            model.eval()
            logits, accs, losses, preds = model(data), [], [], []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
                    pred = (logits[mask].squeeze(-1) > 0).to(torch.long)
                    acc = roc_auc_score(y_true=data.y[mask].cpu().numpy(), y_score=logits[mask].squeeze(-1).cpu().numpy())
                    loss = F.binary_cross_entropy_with_logits(logits[mask].squeeze(-1), data.y[mask].to(torch.float))
                else:
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                    loss = F.cross_entropy(model(data)[mask], data.y[mask])
                preds.append(pred.detach().cpu())
                accs.append(acc)
                losses.append(loss.detach().cpu())
            return accs, preds, losses
    
    # -----------------------------------------------
    # net
    tmp_net = Net(dataset, args)

    # data split
    print('data split seed', args.seed)
    if args.dataset.lower() in ['citeseer', 'pubmed', 'cs','physics']:
        train_rate = 0.6
        val_rate = 0.2
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)            
    elif args.dataset.lower() in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        data = heter_fixed_splits(dataset, data, args.idx_run)
    elif args.dataset.lower() in ["chameleon_filtered", "squirrel_filtered"]:
        data = hetergraph_fixed_split(dataset, data, args.idx_run)
    
    model, data = tmp_net.to(args.device), data.to(args.device) 
    
    parameters = []
    for name, param in model.named_parameters():
        if 'attnmodule' in name: 
            parameters.append({'params': param, 'lr': args.attn_lr, 'weight_decay': args.attn_wd})
        elif 'ffnmodule' in name:  
            parameters.append({'params': param, 'lr': args.lr, 'weight_decay': args.weight_decay})
        else:  
            parameters.append({'params': param, 'lr': args.lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.Adam(parameters)

    best_val_acc = test_acc = 0
    bad_counter = 0

    for epoch in range(args.epochs):
        train(model, optimizer, data)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        print(epoch, 'epochs trained. Current acc:', round(train_acc, 4), round(val_acc, 4), round(tmp_test_acc, 4),\
              'epochs trained. Current loss:', round(float(train_loss), 4), round(float(val_loss), 4), round(float(tmp_test_loss), 4))
        if val_acc > best_val_acc :
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.early_stopping:
            break
    return test_acc, best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=2000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')           
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay.') 
    parser.add_argument('--attn_lr', type=float, default=0.0005, help='learning rate for PolyAttn layer.')
    parser.add_argument('--attn_wd', type=float, default=0.0001, help='weight decay for PolyAttn layer.') 
    parser.add_argument('--early_stopping', type=int, default=250, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for PolyAttn layer.')
    parser.add_argument('--n_head', type=int, default=1, help='number of heads.')
    parser.add_argument('--d_ffn', type=int, default=128, help='hidden dimension of ffn.')
    parser.add_argument('--q', type=float, default=1.0, help='constraint: q.')
    parser.add_argument('--multi', type=float, default=1.0, help='multi: m.')
    parser.add_argument('--K', type=int, default=10, help='truncated order.')
    parser.add_argument('--nlayer', type=int, default=1, help='number of PolyFormer blocks.')
    parser.add_argument('--base', type=str, default='mono')
    parser.add_argument('--dataset', type=str, choices=['citeseer', 'pubmed', 'cs', 'physics', 'chameleon_filtered', 'squirrel_filtered', 'roman-empire', 'minesweeper', 'tolokers', 'questions'], default='tolokers')
    parser.add_argument('--device_idx', type=int, default=0, help='GPU device.')
    parser.add_argument('--net', type=str, default='PolyFormer', help='network name.')
    parser.add_argument('--runs', type=int, default=10, help='runs.')
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    #10 fixed seeds for random splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")
    dataset, data = get_data_load(args)
    gnn_name = args.net
    print("args.num_classes", args.num_classes)
    Net = PolyFormer

    results = []
    for RP in range(args.runs):
        args.idx_run = RP
        print("RP", RP, "Launched...")
        args.seed=SEEDS[RP]
        test_acc, best_val_acc = RunExp(args, dataset, data, Net)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')


    print("--------------------------------------------------")
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
    print(f'{gnn_name} on base {args.base}, dataset {args.dataset}, in {args.runs} repeated experiment: test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    print("---------------------------------------------------")
    
    
