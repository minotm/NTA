#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import itertools
from math import log10, floor, ceil
import time
from torch.utils.data import WeightedRandomSampler

#===========================                                                            ==========================
#===========================   Categorically Encode ngrams       ==========================
#===========================                                                            ==========================

def encode_ngrams(x_train, x_val, x_test, args, seq_len = 30):
    """
    Converts amino acid or nucleotide sequences to categorically encoded vectors based on a chosen
    encoding approach (ngram vocabulary).    
    
    Parameters
    ----------
    x_train, x_val, x_test: pandas.core.series.Series
        pandas Series containing strings of protein or nucleotide training, validation, and testing sequences
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine correct vocabulary size, output 
        shape, and if a mask should be returned
    seq_len: int
        lenght of input sequence. this parameter is only needed for tri+unigram encoding scheme
    Returns
    -------
    x_train_idx, x_val_idx, x_test_idx: list
        categorically encoded sequences
    vocabulary:
        vocabulary used for ngram encoding. to be passed to dataloaer & collate functions
    """    
    def find_3grams(seq):
        '''
        input: sequence of nucleotides
        output: string of sequential in-frame codons / 3grams
        '''
        str_save = ''
        for i in range(2,len(seq) , 3):
            threeGram = seq[i-2],seq[i-1],seq[i]
            thrgrm_join = ''.join(threeGram)
            str_save += thrgrm_join + ' '
        return str_save
    
    def seq_to_cat(seq_df, word_to_idx_dictionary):
        '''
        input: dataframe of sequences & dictionary containing tokens in vocabulary
        output: out_idx: list of torch.Tensors of categorically encoded (vocab index) ngrams 
        '''
        out_idxs = []
        for i in range(len(seq_df)): out_idxs.append(torch.tensor([word_to_idx_dictionary[w] for w in seq_df.iloc[i] if w != None and w != '' ], dtype=torch.long))
        return out_idxs

    if args.seq_type == 'aa' and args.ngram == 'unigram' and args.data_type != 'thermo':
        vocabulary = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    elif args.seq_type == 'dna' and args.ngram == 'unigram':
        vocabulary = ['A', 'C', 'G', 'T']
    elif args.seq_type == 'dna' and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    elif args.seq_type == 'dna' and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
        
    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    
    if args.ngram == 'tri_unigram':
        x_train_maxlen = max(x_train.str.len())
        x_val_maxlen = max(x_val.str.len())
        x_test_maxlen = max(x_test.str.len())
        
        x_train_unigrams = x_train.str.split('',expand=True).drop(columns=[0, int(x_train_maxlen+1)]) #drop pre & post trailing whitespaces
        x_val_unigrams = x_val.str.split('',expand=True).drop(columns=[0, int(x_val_maxlen+1)])
        x_test_unigrams = x_test.str.split('',expand=True).drop(columns=[0, int(x_test_maxlen+1)])
        
        x_train_trigrams = x_train.apply(lambda x: find_3grams(x)).str.split(expand=True)
        x_val_trigrams = x_val.apply(lambda x: find_3grams(x)).str.split(expand=True)
        x_test_trigrams = x_test.apply(lambda x: find_3grams(x)).str.split(expand=True)
        
        x_train = pd.concat([x_train_trigrams, x_train_unigrams], axis = 1)
        x_val = pd.concat([x_val_trigrams, x_val_unigrams], axis = 1)
        x_test = pd.concat([x_test_trigrams, x_test_unigrams], axis = 1)
        
    elif args.ngram == 'trigram_only':
        x_train = x_train.apply(lambda x: find_3grams(x)).str.split(expand=True)
        x_val = x_val.apply(lambda x: find_3grams(x)).str.split(expand=True)
        x_test = x_test.apply(lambda x: find_3grams(x)).str.split(expand=True)
    
    x_train_idx = seq_to_cat(x_train, word_to_ix)
    x_val_idx = seq_to_cat(x_val, word_to_ix)
    x_test_idx = seq_to_cat(x_test, word_to_ix)
    
    return x_train_idx, x_val_idx, x_test_idx





#===========================                                            ================================
#===========================   DataSet & Loaders         ================================
#===========================                                            ================================   

class CustomTorchDataset(Dataset):
    """
    Converts categorically encoded sequences & labels into a torch Dataset
    
    Parameters
    ----------
    encoded_seqs: pandas.core.series.Series
        categorically encoded protein or nucleotide sequences
    labels: pandas.core.series.Series
        class labels or regression fitness values corresponding to sequences

    Returns
    -------
    tuple of sequences, labels (y)
    """    
    def __init__(self, encoded_seqs, labels, transform=None):
        self.encoded_seqs = encoded_seqs
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels) 
    def __getitem__(self, idx):
        seq = self.encoded_seqs[idx]
        label = self.labels[idx]
        if self.transform:
            seq = self.transform(seq)
        return seq, label
        



#===========================   Collater Fn to Apply Padding & Mask         ====================
class Collater(object):
    """
    Collater function to pad sequences of variable length (AAV data) and calculate padding mask. Fed to 
    torch DataLoader collate_fn.
    
    Parameters
    ----------
    alphabet: str
        vocabulary size (i.e. amino acids, nucleotide ngrams). used for one-hot encoding dimension calculation
    pad_tok: float 
        padding token. zero padding is used as default
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine correct vocabulary size, output 
        shape, and if a mask should be returned

    Returns
    -------
    padded sequences, labels (y), and mask as appropriate
    """    
    #Adapated from:
    #Source: https://github.com/J-SNACKKB/FLIP/blob/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/baselines/cnn.py
    def __init__(self, vocab_length: int, 
                pad_tok=0.,
                args = None):        
        self.vocab_length = vocab_length
        self.pad_tok = pad_tok
        self.args = args
    def __call__(self, batch):
        data = tuple(zip(*batch))
        sequences = data[0]
        y = data[1]
        y = torch.tensor(y).squeeze()
        y = y.type(torch.FloatTensor)

        sequences = [i.view(-1,1) for i in sequences]        
        if self.args.data_type == 'aav' and self.args.seq_type == 'aa' and self.args.ngram == 'unigram': maxlen = 42
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'unigram': maxlen = 126
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'tri_unigram': maxlen = 168
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'trigram_only': maxlen = 42
        else: maxlen = sequences[0].shape[0]
        
        
        padded = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", self.pad_tok) for i in sequences]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)

        if self.args.base_model == 'transformer': padded = padded.squeeze()
        elif self.args.base_model == 'cnn':
            ohe = []
            for i in padded:
                i_onehot = torch.FloatTensor(maxlen, self.vocab_length)
                i_onehot.zero_()
                i_onehot.scatter_(1, i, 1)
                ohe.append(i_onehot)
            padded = torch.stack(ohe)
        
        if self.args.data_type == 'gb1' or self.args.data_type == 'gb1_10' or self.args.data_type == 'her2': mask = None
        return padded, y, mask


#
#===========================   Convert Data to torch.DataLoader        ======================
def data_to_loader(x_train,x_val, x_test, y_train,y_val, y_test, batch_size, args, sampler_weights = None):
    """
    Function for converting categorically encoding sequences + their labels to a torch Dataset and DataLoader
    
    Parameters
    ----------
    x_train, x_val, x_test: list
        categorically encoded protein or nucleotide training, validation, and testing sequences
    y_train, y_val, y_test: pandas.core.series.Series
        class labels or regression fitness values corresponding to training, validation, & testing sequences
    batch_size: int
        batch size to be used for dataloader
    args: argparse.ArgumentParser
        arguments specified by user. used for this program to determine correct vocabulary size, output 
        shape, and if a mask should be returned
    alphabet: list of strings
        vocabulary (i.e. amino acids, nucleotide ngrams). passed to collate_fn and used for one-hot 
        encoding dimension calculation
    Returns
    -------
    torch DataLoader objects for training, validation, and testing sets
    """    
    

    y_train = y_train.tolist()
    y_val = y_val.tolist()
    y_test = y_test.tolist()

    train_data = CustomTorchDataset(x_train, y_train, transform = None)
    
    val_data = CustomTorchDataset(x_val, y_val, transform = None)
    
    test_data = CustomTorchDataset(x_test, y_test, transform = None)
    
    if len(y_val) < batch_size : drop_last_val_bool = False
    else: drop_last_val_bool = True
    
    if len(y_train) < batch_size : drop_last_train_bool = False
    else: drop_last_train_bool = True
    
    if args.ngram == 'trigram_only' and args.seq_type =='dna' : vocab_length = 64
    elif args.ngram == 'tri_unigram' and args.seq_type =='dna' : vocab_length = 68
    elif args.ngram == 'unigram' and args.seq_type =='dna' : vocab_length = 4
    elif args.seq_type =='aa' : vocab_length = 20
    
    if sampler_weights != None:
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement = False)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), 
                                                   drop_last=drop_last_train_bool, sampler=sampler)
        
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                                  collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), 
                                                  drop_last=drop_last_val_bool)
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), 
                                                   drop_last=True)   
        
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_train_bool)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                                  collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_val_bool)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=True)

    return train_loader, val_loader, test_loader
