import os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import compress

import fasttext

import torch
from torch import nn

# for LSTM and LSTM SA
class FastTextTokenizer:
    def __init__(self, verbose=False, use_cache=True, 
                 save_cache=True, cache_dir='word_index_cache'):
        self.verbose = verbose
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_dir = cache_dir
        
        self.word_index = {}
        
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

# for LSTM and LSTM SA
class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df

    def get_n_features(self):
        return self.tokenizer.embedding_matrix.shape[1]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        batch_df = self.df.iloc[idxs]
        idxs = list(compress(idxs, batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()))
        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        tokenized = batch_df['TEXT'].apply(self.tokenizer.prepare_sequence).values
        padded = nn.utils.rnn.pad_sequence(tokenized, batch_first=True)
        
        labels = torch.tensor(batch_df['LABEL_NUM'].values)
        labels = labels.to(self.device)
        return padded, torch.tensor(locs), labels

# for ELECTRA
class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.df = df 

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idxs):
        batch_df = self.df.iloc[idxs]
        idxs = list(compress(idxs, batch_df['TEXT'].apply(lambda string: len(string.split()) < self.max_length).to_list()))
        batch_df = self.df.iloc[idxs]
        locs = batch_df['LOCATION'].values
        labels = torch.tensor(batch_df['LABEL_NUM'].values)
        labels = labels.to(self.device)
        tokenized = self.tokenizer.batch_encode_plus(batch_df['TEXT'].tolist(), max_length=self.max_length, \
                    pad_to_max_length=True)['input_ids']
        return tokenized, torch.tensor(locs), labels