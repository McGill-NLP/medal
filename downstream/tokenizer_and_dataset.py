import os
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn

import fasttext

# for LSTM and LSTM SA
class FastTextTokenizer:
    def __init__(self, verbose=False, use_cache=True, 
                 save_cache=True, cache_dir='word_index_cache'):
        self.verbose = verbose
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_dir = cache_dir
        
        self.word_index = {"": 0}
        # self.word_index = {"[MASK]": 0}
        
        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
    
    def build_word_index(self, *args):
        """
        args: pandas series that we will use to build word index
        """
        if self.use_cache:
            filename = os.path.join(self.cache_dir, "word_index.pickle")
            if os.path.exists(filename):
                print("Loading word index from cache...", end=" ")
                self.word_index = pickle.load(open(filename, "rb"))
                # if "[MASK]" not in self.word_index:
                #     self.word_index["[MASK]"] = len(self.word_index)
                print("Done.")
                return
            
            print("Word Index not found!")
        
        print("Generating new word index...", end=" ")
        for df in args:
            for sent in tqdm(df, disable=not self.verbose):
                for word in sent.split():
                    if word not in self.word_index:
                        self.word_index[word] = len(self.word_index)
        print("Done.")
        
        
        if self.save_cache:
            filename = os.path.join(self.cache_dir, "word_index.pickle")
            print("Saving word index...", end=" ")
            pickle.dump(self.word_index, open(filename, 'wb'))
            print("Done.")
        
    def prepare_words(self, words):
        idxs = [self.word_index[w] for w in words]
        return self.embedding_matrix[idxs]
    
    def prepare_sequence(self, seq):
        idxs = [self.word_index[w] for w in seq.split()]
        return torch.tensor(self.embedding_matrix[idxs], dtype=torch.float32)

    
    def build_embedding_matrix(self, path):
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, 300))
        ft_model = fasttext.load_model(path)

        for word, i in tqdm(self.word_index.items(), disable=not self.verbose):
            self.embedding_matrix[i] = ft_model.get_word_vector(word)

        return self.embedding_matrix

def prepare_label(labels, task, output_size):
    if task == 'mimic-mortality':
        return torch.tensor(labels)
    if task == 'mimic-diagnosis':
        output = torch.zeros(labels.shape[0], output_size)
        for sample_idx, label in enumerate(labels):
            for pos_idx in label.split(','):
                output[sample_idx, int(pos_idx)] = 1
        return output

# for LSTM and LSTM SA
class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, task, output_size=None, label_col='LABEL_NUM', text_col='TEXT', 
                 max_len=512, device='cpu'):
        self.df = df
        self.device = device
        self.tokenizer = tokenizer
        self.task = task
        self.output_size = output_size
        self.label_col = label_col
        self.text_col = text_col
        self.max_length = max_len
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        batch_df = self.df.iloc[idxs]
        # idxs = list(compress(idxs, batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()))
        # batch_df = self.df.iloc[idxs]
        
        # labels = torch.tensor(batch_df[self.label_col].values)
        labels = prepare_label(batch_df[self.label_col].values, self.task, self.output_size)
        tokenized = batch_df[self.text_col].apply(self.tokenizer.prepare_sequence).values
        padded = nn.utils.rnn.pad_sequence(tokenized, batch_first=True)
        padded = padded[:, :self.max_length]
        
        # Do not send padded to GPU; this will be done internally by the model!
        # only send labels to the device
        labels = labels.to(self.device)
        
        return padded, labels

# for ELECTRA
class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, task, output_size=None, label_col='LABEL_NUM', text_col='TEXT', 
                max_length=512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df 
        self.label_col = label_col
        self.text_col = text_col
        self.output_size = output_size
        self.task = task

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        batch_df = self.df.iloc[idxs]
        labels = prepare_label(batch_df[self.label_col].values, self.task, self.output_size)
        labels = labels.to(self.device)
        tokenized = self.tokenizer.batch_encode_plus(batch_df[self.text_col].tolist(), max_length=self.max_length, \
                    pad_to_max_length=True)['input_ids']
        return tokenized, labels