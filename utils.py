import os
import pickle

import numpy as np
import pandas as pd
import scipy
import torch
from itertools import compress

from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import fasttext

def load_dataframes(data_dir, data_filename, adam_path):
    train = pd.read_csv(os.path.join(data_dir, 'train', data_filename), engine='c')
    valid = pd.read_csv(os.path.join(data_dir, 'valid', data_filename), engine='c')
    test = pd.read_csv(os.path.join(data_dir, 'test', data_filename), engine='c')

    adam_df = pd.read_csv(adam_path, sep='\t')
    unique_labels = adam_df.EXPANSION.unique()
    label_to_ix = {label: ix for ix, label in enumerate(unique_labels)}

    train['LABEL_NUM'] = train.LABEL.apply(lambda l: label_to_ix[l])
    valid['LABEL_NUM'] = valid.LABEL.apply(lambda l: label_to_ix[l])
    test['LABEL_NUM'] = test.LABEL.apply(lambda l: label_to_ix[l])
    
    return train, valid, test, label_to_ix

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

def evaluate(model, model_type, loader, dataset, criterion, verbose=False, full=True):
    running_loss = 0.0
    count = 0.
    correct = 0.
    total = 0.

    model.eval()
    with torch.no_grad():
        for batch_idx, idx in tqdm(enumerate(loader), disable=not verbose):
            if not full and batch_idx >= 10000:
                break
            if model_type in ["lr"]:
                sents, labels = dataset[idx]

                outputs = model.forward(sents)
            elif model_type in ["trm", "rnnsoft", "disbert", "electra", "rnn", "clibert", "biobert"]:
                sents, locs, labels = dataset[idx]
                if labels.numel() == 0:
                    continue
                outputs = model(sents, locs)
            elif model_type in ["atetm"]:
                sents, bows, locs, labels = dataset[idx]
                outputs, etm_loss = model(sents, bows, locs)
            else:
                sents, mixtures, locs, labels = dataset[idx]
                outputs = model(sents, mixtures, locs)
        
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += torch.sum(outputs.argmax(dim=-1) == labels).item()
            total += labels.size(0)
            count += 1
    
    accuracy = correct / total
    loss = running_loss / count
    
    return loss, accuracy

def train_loop(net, model_type, optimizer, criterion, train_data, valid_data, n_epochs, batch_size, save_dir=None, 
                verbose=False, scheduler=None, eval_every=10000, save_every=40, clip=0, writer=None, accum_num=1):
    
    logs = {k: [] for k in ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']}
    intermediate_logs = {k: [] for k in ['epoch', 'iteration', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc']}

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


        net.train()
        for idx in tqdm(train_loader):
            sents, locs, labels = train_data[idx]
            # gradient accumulation
            if count > 1 and (count - 1) % accum_num == 0:
                optimizer.zero_grad()
            if labels.numel() == 0:
                continue
            outputs = net(sents, locs)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            # gradient accumulation
            if count > 0 and count % accum_num == 0:
                optimizer.step()

            running_loss += loss.item()
            correct += torch.sum(outputs.argmax(dim=-1) == labels).item()
            total += labels.size(0)
            if count % eval_every == 0 and count > 0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, model_type, valid_loader, valid_data, criterion, verbose=verbose, full=False)
                net.train()
                if scheduler:
                    scheduler.step(valid_loss)
                
                print(f"End of iteration {count}")
                print(f"Train Loss: {running_loss/count:.4f} \tTrain Accuracy:{correct/total:.4f}")
                print(f"Valid Loss: {valid_loss:.4f} \tValid Accuracy:{valid_acc:.4f}")
                print("="*50)
                print()
                intermediate_logs['epoch'].append(epoch)
                intermediate_logs['iteration'].append(count)
                intermediate_logs['train_loss'].append(running_loss/count)
                intermediate_logs['train_acc'].append(correct/total)
                intermediate_logs['valid_loss'].append(valid_loss)
                intermediate_logs['valid_acc'].append(valid_acc)
                if not os.path.exists(os.path.join(save_dir)):
                    os.makedirs(os.path.join(save_dir))
                intermediate_log_df = pd.DataFrame(intermediate_logs)
                intermediate_log_df.to_csv(os.path.join(save_dir, 'intermediate_logs.csv'))
            count += 1

        valid_loss, valid_acc = evaluate(net, model_type, valid_loader, valid_data, criterion, verbose=verbose)
        if scheduler:
            scheduler.step(valid_loss)

        print(f"End of epoch {epoch}")
        print(f"Train Loss: {running_loss/count:.4f} \tTrain Accuracy:{correct/total:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} \tValid Accuracy:{valid_acc:.4f}")
        print("="*50)
        print()

        logs['train_loss'].append(running_loss/count)
        logs['train_acc'].append(correct/total)
        logs['valid_loss'].append(valid_loss)
        logs['valid_acc'].append(valid_acc)

        # Tensorboard
        if writer:
            for key, values in logs.items():
                writer.add_scalar(key, values[-1], epoch)

        if epoch > 3:
            if logs['valid_acc'][-1] < np.sum(logs['valid_acc'][-2]):
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
