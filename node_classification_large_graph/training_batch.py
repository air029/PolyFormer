from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from mymodels_large import *
import uuid
import pickle
from collections import Counter
import seaborn as sns

def accuracy(output, labels, batch=False):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if batch == True:
        return correct
    return correct / len(labels)

def create_batch(input_data):
    num_sample = input_data[0].shape[0]
    list_bat = []
    for i in range(0,num_sample,batch_size):
        if (i+batch_size)<num_sample:
            list_bat.append((i,i+batch_size))
        else:
            list_bat.append((i,num_sample))
    return list_bat

def train(st,end):
    model.train()
    optimizer.zero_grad()
    output = model(train_data,st,end)
    acc_train = accuracy(output, train_labels[st:end])
    loss_train = F.nll_loss(output, train_labels[st:end])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()

def validate(st,end):
    model.eval()
    with torch.no_grad():
        output = model(valid_data,st,end)
        loss_val = F.nll_loss(output, valid_labels[st:end])
        acc_val = accuracy(output, valid_labels[st:end],batch=True)
        return loss_val.item(),acc_val.item()

def test(st,end):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model(test_data,st,end)
        loss_test = F.nll_loss(output, test_labels[st:end])
        acc_test = accuracy(output, test_labels[st:end],batch=True)
        return loss_test.item(),acc_test.item()
    

def define_model(args):
    gnn_name = args.net
    if gnn_name == 'PolyFormer':
        Net = PolyFormer(None, args)
    else:
        assert False, "Unknown model name"
    return Net

def reset_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=2000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')           
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay.') 
    parser.add_argument('--attn_lr', type=float, default=0.0005, help='learning rate for PolyAttn layer.')
    parser.add_argument('--attn_wd', type=float, default=0.0001, help='weight decay for PolyAttn layer.') 
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for PolyAttn layer.')
    parser.add_argument('--n_head', type=int, default=1, help='number of heads.')
    parser.add_argument('--d_ffn', type=int, default=128, help='hidden dimension of ffn.')
    parser.add_argument('--q', type=float, default=1.0, help='constraint: q.')
    parser.add_argument('--multi', type=float, default=1.0, help='multi: m.')
    parser.add_argument('--K', type=int, default=10, help='truncated order.')
    parser.add_argument('--nlayer', type=int, default=1, help='number of PolyFormer blocks.')
    parser.add_argument('--base', type=str, choices=['mono','cheb'], default='mono')
    parser.add_argument('--dataset', type=str, choices=['ogbn-arxiv', 'twitch-gamer', 'pokec', 'ogbn-papers100M'], default='pokec')
    parser.add_argument('--device_idx', type=int, default=0, help='GPU device.')
    parser.add_argument('--net', type=str, default='PolyFormer', help='network name.')
    parser.add_argument('--runs', type=int, default=1, help='runs.')
    parser.add_argument('--metric', type=str, default='acc', help='metric.')
    parser.add_argument('--test', type=int, default=1, help='test.')
    parser.add_argument('--batch_size', type=int, default=50000, help='batch size.')
        
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_layer = args.nlayer
    batch_size = args.batch_size

    print(f"========================")
    print(args)

    test_acc_list = []
    valid_acc_list = []
    for run_idx in range(args.runs):
        print('--'*50)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        reset_random_seeds(run_idx)
        args.splits_idx = run_idx
        if args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']:
            args.splits_idx = 0
        name = args.dataset
        data_path = './data/'
        args.dev = args.device_idx
        cudaid = "cuda:"+str(args.dev)
        device = torch.device(cudaid)

        if args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']:
            name = args.dataset + '_split_' + str(args.splits_idx)
        elif args.dataset in ['twitch-gamer', 'pokec']:
            name = args.dataset + '_split_' + str(args.splits_idx)
            
        with open(data_path+"labels_"+name+".pickle","rb") as fopen:
            labels = pickle.load(fopen)

        name += '_K_' + str(args.K) + '_' + args.base

        with open(data_path+"training_"+name+".pickle","rb") as fopen:
            train_data = pickle.load(fopen)
        with open(data_path+"validation_"+name+".pickle","rb") as fopen:
            valid_data = pickle.load(fopen)
        with open(data_path+"test_"+name+".pickle","rb") as fopen:
            test_data = pickle.load(fopen)

        train_data = [mat.to(device) for mat in train_data[:args.K+1]]
        valid_data = [mat.to(device) for mat in valid_data[:args.K+1]]
        test_data = [mat.to(device) for mat in test_data[:args.K+1]]

        train_labels = labels[0].reshape(-1).long().to(device)
        valid_labels = labels[1].reshape(-1).long().to(device)
        test_labels = labels[2].reshape(-1).long().to(device)

        num_features = train_data[0].shape[1]
        num_labels = int(train_labels.max()) + 1
        args.num_features = num_features
        args.num_labels = num_labels
        args.num_classes = num_labels

        print("Number of labels for "+name, num_labels)
        checkpt_file = './pretrained/'+uuid.uuid4().hex+'.pt'
        print(cudaid,checkpt_file)

        model = define_model(args)
        model.to(device)

        if args.net in ['PolyFormer']:
            parameters = []
            for name, param in model.named_parameters():
                if 'attnmodule' in name: 
                    parameters.append({'params': param, 'lr': args.attn_lr, 'weight_decay': args.attn_wd})
                elif 'ffnmodule' in name:  
                    parameters.append({'params': param, 'lr': args.lr, 'weight_decay': args.weight_decay})
                else:  
                    parameters.append({'params': param, 'lr': args.lr, 'weight_decay': args.weight_decay})
            optimizer = torch.optim.Adam(parameters)
        else:
            assert False, "Unknown model name"

        list_bat_train = create_batch(train_data)
        list_bat_val = create_batch(valid_data)
        list_bat_test = create_batch(test_data)
        t_total = time.time()

        bad_counter = 0
        best_epoch = 0
        loss_best = 10000000.
        acc_best = 0.
        valid_num = valid_data[0].shape[0]
        test_num = test_data[0].shape[0]
        for epoch in range(args.epochs):
            list_loss = []
            list_acc = []
            random.shuffle(list_bat_train)
            for st,end in list_bat_train:
                loss_tra,acc_tra = train(st,end)
                list_loss.append(loss_tra)
                list_acc.append(acc_tra)
            loss_tra = np.round(np.mean(list_loss),4)
            acc_tra = np.round(np.mean(list_acc),4)

            list_loss_val = []
            list_acc_val = []
            for st,end in list_bat_val:
                loss_val,acc_val = validate(st,end)
                list_loss_val.append(loss_val)
                list_acc_val.append(acc_val)

            loss_val = np.mean(list_loss_val)
            acc_val = (np.sum(list_acc_val))/valid_num

            #Uncomment to see losses
            if(epoch+1)%1 == 0:
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    'acc:{:.2f}'.format(acc_tra*100),
                    '| val',
                    'loss:{:.3f}'.format(loss_val),
                    'acc:{:.2f}'.format(acc_val*100))


            if args.metric == 'loss':
                if loss_val < loss_best:
                    loss_best = loss_val
                    acc_best = acc_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), checkpt_file)
                    bad_counter = 0
                else:
                    bad_counter += 1
            elif args.metric == 'acc':
                if acc_val > acc_best:
                    loss_best = loss_val
                    acc_best = acc_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), checkpt_file)
                    bad_counter = 0
                else:
                    bad_counter += 1

            if bad_counter == args.early_stopping:
                break

        if args.test:
            list_loss_test = []
            list_acc_test = []
            model.load_state_dict(torch.load(checkpt_file))
            for st,end in list_bat_test:
                loss_test,acc_test = test(st,end)
                list_loss_test.append(loss_test)
                list_acc_test.append(acc_test)
            acc_test = (np.sum(list_acc_test))/test_num

        print("Train cost: {:.4f}s".format(time.time() - t_total))
        print('Load {}th epoch'.format(best_epoch))

        if args.test:
            print(f"Valdiation accuracy: {np.round(acc_best*100,2)}")
            print(f"Valdiation loss: {np.round(loss_best,4)}")
            print(f"Test accuracy: {np.round(acc_test*100,2)}")
            test_acc_list.append(acc_test)
        else:
            print(f"Valdiation accuracy: {np.round(acc_best*100,2)}")
            print(f"Valdiation loss: {np.round(loss_best,4)}")
        valid_acc_list.append(acc_best)

    test_acc_mean = np.mean(test_acc_list)
    valid_acc_mean = np.mean(valid_acc_list)
    print('---------------------------------------Results---------------------------------------')
    values=np.asarray(test_acc_list)
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
    print(f'{args.net} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean*100:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {valid_acc_mean*100:.4f}')




