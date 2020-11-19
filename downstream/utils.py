import os
import pickle

import numpy as np
import pandas as pd
import scipy
import torch
from itertools import compress
# import torch.nn.functional as F
# import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import joblib
from gensim.sklearn_api import TfIdfTransformer
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
import fasttext

# Load downstream tasks' data
# For mimic mortality
def load_mimic_mortality(data_dir, data_filename):
    train = pd.read_csv(os.path.join(data_dir, 'train', data_filename), engine='c')
    valid = pd.read_csv(os.path.join(data_dir, 'valid', data_filename), engine='c')
    test = pd.read_csv(os.path.join(data_dir, 'test', data_filename), engine='c')

    train['LABEL_NUM'] = train.HOSPITAL_EXPIRE_FLAG.astype(np.float32)
    valid['LABEL_NUM'] = valid.HOSPITAL_EXPIRE_FLAG.astype(np.float32)
    test['LABEL_NUM'] = test.HOSPITAL_EXPIRE_FLAG.astype(np.float32)

    return train, valid, test

def load_mimic_diagnosis(data_dir, data_filename, diag_to_idx_path):
    train = pd.read_csv(os.path.join(data_dir, 'train', data_filename), engine='c')
    valid = pd.read_csv(os.path.join(data_dir, 'valid', data_filename), engine='c')
    test = pd.read_csv(os.path.join(data_dir, 'test', data_filename), engine='c')
    # with open('../../data/downstream/mimic-diagnosis/train/diag_to_idx.pkl', 'rb') as file:
    with open(diag_to_idx_path, 'rb') as file:
        diag_to_idx = pickle.load(file)

    return train, valid, test, diag_to_idx

def load_model(net, load_path, device='cpu'):
    try:
        pretrained = torch.load(load_path, map_location=device).state_dict()
    except:
        pretrained = torch.load(load_path, map_location=device)
    if os.path.splitext(load_path)[-1] == '.tar':
        pretrained = pretrained['model_state_dict']
    print('pretrained: {}'.format(pretrained.keys()))
    for key, value in pretrained.items():
        new_key = key[len('module.'): ] if key.startswith('module.') else key
        if new_key not in net.state_dict():
            print(new_key, 'not expected')
            continue
        try:
            net.state_dict()[new_key].copy_(value)
        except:
            print(new_key, 'not loaded')
            continue
    return net

def compute_top_k_recall(labels, predictions, k=10):
    try:
        idxs = torch.argsort(predictions, dim=1, descending=True)[:, 0: k]
    except:
        raise Exception(labels.shape)
    return (torch.gather(labels, 1, idxs).sum(1) / labels.sum(1)).mean().item()

def predict(model, loader, dataset, verbose=False):
    preds = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(loader, disable=not verbose):
            sents, labels = dataset[idx]
            # if len(sents) <= 1:
                # continue
            outputs = model(sents)
            preds.append(outputs.round())

    return torch.cat(preds)

def evaluate(model, loader, dataset, criterion, verbose=False, task='mimic-mortality'):
    running_loss = 0.0
    count = 0.
    correct = 0.
    total = 0.
    top_10_recall = 0.
    top_5_recall = 0.
    top_30_recall = 0.
    if task == 'mimic-diagnosis':
        metrics = {}

    model.eval()
    with torch.no_grad():
        for idx in tqdm(loader, disable=not verbose):
            sents, labels = dataset[idx]
            outputs = model(sents)
            if len(sents) <= 1:
                continue
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            if task == 'mimic-mortality':
                correct += torch.sum(outputs.round() == labels).item()
                total += labels.size(0)
            elif task == 'mimic-diagnosis':
                top_10_recall += compute_top_k_recall(labels, outputs)
                top_5_recall += compute_top_k_recall(labels, outputs, k=5)
                top_30_recall += compute_top_k_recall(labels, outputs, k=30)
            count += 1
    
    loss = running_loss / count
    if task == 'mimic-mortality':
        metric = correct / total    # Metric
        return loss, metric
    elif task == 'mimic-diagnosis':
        metrics['top_10_recall'] = top_10_recall / count
        metrics['top_5_recall'] = top_5_recall / count
        metrics['top_30_recall'] = top_30_recall / count
        return loss, metrics

def train_loop(net, optimizer, criterion, train_data, valid_data, n_epochs, batch_size, task='mimic-mortality', save_dir=None, 
                verbose=False, scheduler=None, eval_every=10000, save_every=40, writer=None):
    """
    net: nn.Module we are training
    optimizer: pytorch optimizer object
    criterion: loss function
    train_data: torch.utils.data.Dataset, we will take indices from this
    valid_data: torch.utils.data.Dataset, we will take indices from this
    [...]
    """
    
    logs = {k: [] for k in ['train_loss', 'valid_loss', 'train_metric', 'valid_metric']}
    intermediate_logs = {k: [] for k in ['epoch', 'iteration', 'train_loss', 'valid_loss', 'train_metric', 'valid_metric']}
    if task == 'mimic-diagnosis':
        logs['train_top_5_recall'] = []
        logs['train_top_30_recall'] = []
        logs['valid_top_5_recall'] = []
        logs['valid_top_30_recall'] = []

        intermediate_logs['train_top_5_recall'] = []
        intermediate_logs['train_top_30_recall'] = []
        intermediate_logs['valid_top_5_recall'] = []
        intermediate_logs['valid_top_30_recall'] = []

    break_cnt = 0

    train_loader = DataLoader(
        range(len(train_data)), 
        shuffle=True, 
        batch_size=batch_size
    )
    valid_loader = DataLoader(
        range(len(valid_data)), 
        shuffle=True, 
        batch_size=batch_size
    )
    print("Datasets created:\n")
    print("Training set:", len(train_data), "samples\n")
    print("Validation set:", len(valid_data), "samples\n")
    print("Start training\n")

    for epoch in range(n_epochs):
        running_loss = 0.0
        count = 0.
        correct = 0.
        total = 0.
        top_10_recall = 0.
        top_5_recall = 0.
        top_30_recall = 0.

        net.train()
        for idx in tqdm(train_loader):
            sents, labels = train_data[idx]
            if labels.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            outputs = net(sents)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
            if task == 'mimic-mortality':
                correct += torch.sum(outputs.round() == labels).item()
                total += labels.size(0)
                train_metric = correct / total
            elif task == 'mimic-diagnosis':
                top_10_recall += compute_top_k_recall(labels, outputs)
                top_5_recall += compute_top_k_recall(labels, outputs, k=5)
                top_30_recall += compute_top_k_recall(labels, outputs, k=30)

                train_metric = top_10_recall / count
                train_top_5_recall = top_5_recall / count
                train_top_30_recall = top_30_recall / count

            if count % eval_every == 0 and count > 0:
                net.eval()
                if task == 'mimic-mortality':
                    valid_loss, valid_metric = evaluate(net, valid_loader, valid_data, criterion, verbose=verbose, task=task)
                elif task == 'mimic-diagnosis':
                    valid_loss, valid_metrics = evaluate(net, valid_loader, valid_data, criterion, verbose=verbose, task=task)
                    valid_metric = valid_metrics['top_10_recall']
                net.train()
                if scheduler:
                    scheduler.step(valid_loss)
                
                print(f"End of iteration {count}")
                print(f"Train Loss: {running_loss/count:.4f} \tTrain Metric:{train_metric:.4f}")
                print(f"Valid Loss: {valid_loss:.4f} \tValid Metric:{valid_metric:.4f}")
                if task == 'mimic-diagnosis':
                    print(f"Valid Top 5 Recall: {valid_metrics['top_5_recall']:.4f} \tValid Top 30 Recall:{valid_metrics['top_30_recall']:.4f}")
                print("="*50)
                print()
                intermediate_logs['epoch'].append(epoch)
                intermediate_logs['iteration'].append(count)
                intermediate_logs['train_loss'].append(running_loss/count)
                intermediate_logs['train_metric'].append(train_metric)
                if task == 'mimic-diagnosis':
                    intermediate_logs['train_top_5_recall'].append(train_top_5_recall)
                    intermediate_logs['train_top_30_recall'].append(train_top_30_recall)
                intermediate_logs['valid_loss'].append(valid_loss)
                intermediate_logs['valid_metric'].append(valid_metric)
                if task == 'mimic-diagnosis':
                    intermediate_logs['valid_top_5_recall'].append(valid_metrics['top_5_recall'])
                    intermediate_logs['valid_top_30_recall'].append(valid_metrics['top_30_recall'])
                if not os.path.exists(os.path.join(save_dir)):
                    os.makedirs(os.path.join(save_dir))
                intermediate_log_df = pd.DataFrame(intermediate_logs)
                intermediate_log_df.to_csv(os.path.join(save_dir, 'intermediate_logs.csv'))

        if task == 'mimic-mortality':
            valid_loss, valid_metric = evaluate(net, valid_loader, valid_data, criterion, verbose=verbose, task=task)
        elif task == 'mimic-diagnosis':
            valid_loss, valid_metrics = evaluate(net, valid_loader, valid_data, criterion, verbose=verbose, task=task)
            valid_metric = valid_metrics['top_10_recall']
        if scheduler:
            scheduler.step(valid_loss)

        print(f"End of epoch {epoch}")
        print(f"Train Loss: {running_loss/count:.4f} \tTrain Metric:{train_metric:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} \tValid Metric:{valid_metric:.4f}")
        if task == 'mimic-diagnosis':
            print(f"Valid Top 5 Recall: {valid_metrics['top_5_recall']:.4f} \tValid Top 30 Recall:{valid_metrics['top_30_recall']:.4f}")
        print("="*50)
        print()

        logs['train_loss'].append(running_loss/count)
        logs['train_metric'].append(train_metric)
        if task == 'mimic-diagnosis':
            logs['train_top_5_recall'].append(train_top_5_recall)
            logs['train_top_30_recall'].append(train_top_30_recall)
        logs['valid_loss'].append(valid_loss)
        logs['valid_metric'].append(valid_metric)
        if task == 'mimic-diagnosis':
            logs['valid_top_5_recall'].append(valid_metrics['top_5_recall'])
            logs['valid_top_30_recall'].append(valid_metrics['top_30_recall'])

        # Tensorboard
        if writer:
            for key, values in logs.items():
                writer.add_scalar(key, values[-1], epoch)

        if epoch > 3:
            if logs['valid_metric'][-1] < logs['valid_metric'][-2]:
                break_cnt += 1
                if break_cnt == 3:
                    break
            else:
                break_cnt = 0

        if save_dir and epoch > 0 and (epoch % save_every == 0):
            if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
                os.makedirs(os.path.join(save_dir, 'checkpoints'))
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(save_dir, 'checkpoints', str(epoch) + '.tar'))

            log_df = pd.DataFrame(logs)
            log_df.to_csv(os.path.join(save_dir, 'checkpoints', str(epoch) + '_logs.csv'))
    
    return net,logs