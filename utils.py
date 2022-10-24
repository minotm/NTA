#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import itertools
import time
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

'''
    This file contains functions for the following:
        - encoding amino acid and nucloetide sequences
        - torch dataloaders, collaters, datasets 
        - online augmentation transformations & dataloaders
'''


#===========================                                                            ==========================
#===========================   Categorically Encode ngrams       ==========================
#===========================                                                            ==========================
def encode_ngrams(x,args, seq_len = 30):
    """
    Converts amino acid or nucleotide sequences to categorically encoded vectors based on a chosen
    encoding approach (ngram vocabulary).    
    
    Parameters
    ----------
    x: pandas.core.series.Series
        pandas Series containing strings of protein or nucleotide training, validation, or testing sequences
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
        vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    elif args.seq_type == 'dna' and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'C', 'G', 'T']
    elif args.seq_type == 'dna' and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    elif args.seq_type == 'dna' and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
        
    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    
    if args.ngram == 'tri_unigram':
        x_maxlen = max(x.str.len())
        x_unigrams = x.str.split('',expand=True).drop(columns=[0, int(x_maxlen+1)]) #drop pre & post trailing whitespaces
        x_trigrams = x.apply(lambda y: find_3grams(y)).str.split(expand=True) 
        x = pd.concat([x_trigrams, x_unigrams], axis = 1)

    elif args.ngram == 'trigram_only':
        x = x.apply(lambda y: find_3grams(y)).str.split(expand=True)
    x_idx = seq_to_cat(x, word_to_ix)
    
    return x_idx



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
    def __init__(self, encoded_seqs, labels, args, transform=None):
        self.encoded_seqs = encoded_seqs
        self.labels = labels
        self.transform = transform
        self.args = args
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
    def __init__(self, vocab_length: int, 
                pad_tok=0,
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

        if self.args.data_type == 'aav' and self.args.seq_type == 'aa' and self.args.ngram == 'unigram': maxlen = 42
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'unigram': maxlen = 126
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'tri_unigram': maxlen = 168
        elif self.args.data_type == 'aav' and self.args.seq_type == 'dna' and self.args.ngram == 'trigram_only': maxlen = 42
        else: maxlen = sequences[0].shape[0]
        
        padded = torch.stack([torch.cat([i, i.new_zeros(maxlen - i.size(0))], 0) for i in sequences],0)
        X_len = torch.LongTensor([len(i) for i in sequences])
        mask = torch.arange(maxlen)[None, :] < X_len[:, None]
        mask = ~mask #torch's transformer src_key_padding mask requires padded tokens = True and non-padded = False

        if self.args.base_model == 'cnn' or self.args.base_model == 'cnn2':
            padded = F.one_hot(padded, num_classes = self.vocab_length)
            mask = torch.ones_like(padded)
            mask[:,:,0] = 0
            if self.args.data_type == 'aav':
                padded = padded * mask
            mask = None

        if self.args.data_type == 'gb1' or self.args.data_type == 'her2': mask = None
        
        return padded, y, mask



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

    train_data = CustomTorchDataset(x_train, y_train,  args, transform = None)
    val_data = CustomTorchDataset(x_val, y_val,  args, transform = None)
    test_data = CustomTorchDataset(x_test, y_test,  args, transform = None)
    
    if len(y_val) < batch_size : drop_last_val_bool = False
    else: drop_last_val_bool = True
    
    if len(y_train) < batch_size : drop_last_train_bool = False
    else: drop_last_train_bool = True
    
    #vocab lenghts are increased by 1 to include UNK token
    if args.ngram == 'trigram_only' and args.seq_type =='dna' : vocab_length = 62
    elif args.ngram == 'tri_unigram' and args.seq_type =='dna' : vocab_length = 66
    elif args.ngram == 'unigram' and args.seq_type =='dna' : vocab_length = 5
    elif args.seq_type =='aa' : vocab_length = 21
    
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

#===========================   Convert Data to torch.DataLoader        ======================
def data_to_loader_single(x,y, batch_size, args, sampler_weights = None, shuffle = True, is_test = True):
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
    y = y.tolist()
    data = CustomTorchDataset(x, y,  args, transform = None)
    
    if len(y) < batch_size : drop_last_bool = False
    elif is_test == True: drop_last_bool = True
    else: drop_last_bool = True
    
    #vocab lenghts are increased by 1 to include UNK token
    if args.ngram == 'trigram_only' and args.seq_type =='dna' : vocab_length = 62
    elif args.ngram == 'tri_unigram' and args.seq_type =='dna' : vocab_length = 66
    elif args.ngram == 'unigram' and args.seq_type =='dna' : vocab_length = 5
    elif args.seq_type =='aa' : vocab_length = 21
    
    if sampler_weights != None:
        
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement = False)
        
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle = shuffle,
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), 
                                                   drop_last=drop_last_bool, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, 
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_bool)
    return data_loader









#============================================================================
#=========================                  Online Augmentaiton   ===========================
#============================================================================
class nta_transform(object):
    #Codon List
     #develop a dictionary of nucleotide codons and their amino acid counterparts
    def __init__(self, args):
        
        self.aug_type = args.aug_type
        self.aug_factor = args.aug_factor
        self.subst_frac = args.subst_frac
        self.args = args

        if self.aug_type == 'online':
            self.nt_synonyms = {
                    #A
                    'GCT': ['GCC', 'GCA', 'GCG'],
                    'GCC': ['GCT', 'GCA', 'GCG'],
                    'GCA': ['GCT', 'GCC', 'GCG'],
                    'GCG': ['GCT', 'GCC', 'GCA'],
                    #R
                    'CGT': ['CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
                    'CGC': ['CGT', 'CGA', 'CGG', 'AGA', 'AGG'],
                    'CGA': ['CGT', 'CGC', 'CGG', 'AGA', 'AGG'],
                    'CGG': ['CGT', 'CGC', 'CGA', 'AGA', 'AGG'],
                    'AGA': ['CGT', 'CGC', 'CGA', 'CGG', 'AGG'],
                    'AGG': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA'],
                    #N
                    'AAT': ['AAC'],
                    'AAC': ['AAT'],
                    #D
                    'GAT': ['GAC'],
                    'GAC': ['GAT'],
                    #C
                    'TGT': ['TGC'],
                    'TGC': ['TGT'],                
                    #Q
                    'CAA': ['CAG'],
                    'CAG': ['CAA'],
                    #E
                    'GAA': ['GAG'],
                    'GAG': ['GAA',],
                    #G
                    'GGT': ['GGC', 'GGA', 'GGG'],
                    'GGC': ['GGT', 'GGA', 'GGG'],
                    'GGA': ['GGT', 'GGC', 'GGG'],
                    'GGG': ['GGT', 'GGC', 'GGA', ],
                    #H
                    'CAT': ['CAC'],
                    'CAC': ['CAT'],
                    #I
                    'ATT': ['ATC', 'ATA'],
                    'ATC': ['ATT', 'ATA'],
                    'ATA': ['ATT', 'ATC',],
                    #L
                    'CTT': ['CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
                    'CTC': ['CTT', 'CTA', 'CTG', 'TTA', 'TTG'],
                    'CTA': ['CTT', 'CTC', 'CTG', 'TTA', 'TTG'],
                    'CTG': ['CTT', 'CTC', 'CTA', 'TTA', 'TTG'],
                    'TTA': ['CTT', 'CTC', 'CTA', 'CTG', 'TTG'],
                    'TTG': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA'],
                    #K
                    'AAA': ['AAG'],
                    'AAG': ['AAA'],
                    #M
                    'ATG': ['ATG'],
                    #F
                    'TTT': ['TTC'],
                    'TTC': ['TTT'],
                    #P
                    'CCT': ['CCC', 'CCA', 'CCG'],
                    'CCC': ['CCT', 'CCA', 'CCG'],
                    'CCA': ['CCT', 'CCC', 'CCG'],
                    'CCG': ['CCT', 'CCC', 'CCA'],
                    #S
                    'TCT': ['TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
                    'TCC': ['TCT', 'TCA', 'TCG', 'AGT', 'AGC'],
                    'TCA': ['TCT', 'TCC', 'TCG', 'AGT', 'AGC'],
                    'TCG': ['TCT', 'TCC', 'TCA', 'AGT', 'AGC'],
                    'AGT': ['TCT', 'TCC', 'TCA', 'TCG', 'AGC'],
                    'AGC': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT'],
                    #T
                    'ACT': ['ACC', 'ACA', 'ACG'],
                    'ACC': ['ACT', 'ACA', 'ACG'],
                    'ACA': ['ACT', 'ACC', 'ACG'],
                    'ACG': ['ACT', 'ACC', 'ACA'],
                    #W
                    'TGG': ['TGG'],
                    #Y
                    'TAT': ['TAC'],
                    'TAC': ['TAT'],
                    #V
                    'GTT': ['GTC', 'GTA', 'GTG'],
                    'GTC': ['GTT', 'GTA', 'GTG'],
                    'GTA': ['GTT', 'GTC', 'GTG'],
                    'GTG': ['GTT', 'GTC', 'GTA'],
                    #*
                    'TAA': ['TGA', 'TAG'],
                    'TGA': ['TAA', 'TAG'],
                    'TAG': ['TAA', 'TGA'],
                }
        elif self.aug_type == 'online_balance':
            #used artificially balanced codon substitution dictionary
            self.nt_synonyms = {
                
                        #'A': 
                        'GCT': ['GCC', 'GCA'],
                        'GCC': ['GCT', 'GCA'],
                        'GCA': ['GCT', 'GCC'],
                        #R
                        'CGT': ['CGC', 'CGA'],
                        'CGC': ['CGT', 'CGA'],
                        'CGA': ['CGT', 'CGC'],
                        #N
                        'AAT': ['AAC',  'GCG'],
                        'AAC': ['AAT', 'GCG'],
                        'GCG': ['AAT', 'AAC'],
                        #D
                        'GAT': ['GAC', 'CGG'],
                        'GAC': ['GAT', 'CGG'],
                        'CGG': ['GAT', 'GAC'],
                        #C
                        'TGT': ['TGC', 'AGA'],
                        'TGC': ['TGT', 'AGA'],
                        'AGA': ['TGT', 'TGC'],
                        #Q
                        'CAA': ['CAG', 'AGG'],
                        'CAG': ['CAA', 'AGG'],
                        'AGG': ['CAA', 'CAG'],
                        #E
                        'GAA': ['GAG', 'GGG'],
                        'GAG': ['GAA', 'GGG'],
                        'GGG': ['GAA', 'GAG'],
                        #G
                        'GGT': ['GGC', 'GGA'],
                        'GGC': ['GGT', 'GGA'],
                        'GGA': ['GGT', 'GGC'],
                        #H
                        'CAT': ['CAC', 'CTG'],
                        'CAC': ['CAT', 'CTG'],
                        'CTG': ['CAT', 'CAC'],
                        #I
                        'ATT': ['ATC', 'ATA'],
                        'ATC': ['ATT', 'ATA'],
                        'ATA': ['ATT', 'ATC'],
                        #L
                        'CTT': ['CTC', 'CTA'],
                        'CTC': ['CTT', 'CTA'],
                        'CTA': ['CTT', 'CTC'],
                        #K
                        'AAA': ['AAG','TTA'],
                        'AAG': ['AAA', 'TTA'],
                        'TTA': ['AAA', 'AAG'],
                        #M
                        'ATG': ['TTG', 'CCG'],
                        'TTG': ['ATG', 'CCG'],
                        'CCG': ['ATG', 'TTG'],
                        #F
                        'TTT': ['TTC', 'TCG'],
                        'TTC': ['TTT', 'TCG'],
                        'TCG': ['TTT', 'TTC'],
                        #P
                        'CCT': ['CCC', 'CCA'],
                        'CCC': ['CCT', 'CCA'],
                        'CCA': ['CCT', 'CCC'],
                        #S
                        'TCT': ['TCC', 'TCA'],
                        'TCC': ['TCT', 'TCA'],
                        'TCA': ['TCT', 'TCC'],
                        #T
                        'ACT': ['ACC', 'ACA'],
                        'ACC': ['ACT', 'ACA'],
                        'ACA': ['ACT', 'ACC'],
                        #W
                        'TGG': ['AGT', 'AGC'],
                        'AGT': ['TGG', 'AGC'],
                        'AGC': ['TGG', 'AGT'],
                        #Y
                        'TAT': ['TAC', 'ACG'],
                        'TAC': ['TAT', 'ACG'],
                        'ACG': ['TAT', 'TAC'],
                         #V
                        'GTT': ['GTC', 'GTA'],
                        'GTC': ['GTT', 'GTA'],
                        'GTA': ['GTT', 'GTC'],
                }
        
        
        elif self.aug_type == 'online_shuffle':
            #used artificially balanced codon substitution dictionary
            self.nt_synonyms = {
              #A
              'GAG': ['CCG', 'CTT', 'TTG'],
              'CCG': ['GAG', 'CTT', 'TTG'],
              'CTT': ['GAG', 'CCG', 'TTG'],
              'TTG': ['GAG', 'CCG', 'CTT'],
              #R
             'GGG': ['GGA', 'AGA', 'ATC', 'CCA', 'CTG'],
             'GGA': ['GGG', 'AGA', 'ATC', 'CCA', 'CTG'],
             'AGA': ['GGG', 'GGA', 'ATC', 'CCA', 'CTG'],
             'ATC': ['GGG', 'GGA', 'AGA', 'CCA', 'CTG'],
             'CCA': ['GGG', 'GGA', 'AGA', 'ATC', 'CTG'],
             'CTG': ['GGG', 'GGA', 'AGA', 'ATC', 'CCA'],
             #N
             'TCG': ['TTT'],
             'TTT': ['TCG'],
             #D
             'ACT': ['CAT'],
             'CAT': ['ACT'],
             #C
             'GAA': ['TGG'],
             'TGG': ['GAA'],
             #Q
             'CGG': ['CTC'],
             'CTC': ['CGG'],
             #E
             'GAC': ['GTA'],
             'GTA': ['GAC'],
             #G
             'TGC': ['TCC', 'AGC', 'ACC'],
             'TCC': ['TGC', 'AGC', 'ACC'],
             'AGC': ['TGC', 'TCC', 'ACC'],
             'ACC': ['TGC', 'TCC', 'AGC'],
             #H
             'TCA': ['GTT'],
             'GTT': ['TCA'],
             #I
             'GTG': ['ACG', 'AAG'],
             'ACG': ['GTG', 'AAG'],
             'AAG': ['GTG', 'ACG'],
             #L
             'ATG': ['ACA', 'CCC', 'GAT', 'TAC', 'TAT'],
             'ACA': ['ATG', 'CCC', 'GAT', 'TAC', 'TAT'],
             'CCC': ['ATG', 'ACA', 'GAT', 'TAC', 'TAT'],
             'GAT': ['ATG', 'ACA', 'CCC', 'TAC', 'TAT'],
             'TAC': ['ATG', 'ACA', 'CCC', 'GAT', 'TAT'],
             'TAT': ['ATG', 'ACA', 'CCC', 'GAT', 'TAC'],
             #K
             'GGT': ['GGT'],
             'GTC': ['GTC'],
             #M
             'CCT': ['CCT'],
             #F
             'CGC': ['TCT'],
             'TCT': ['CGC'],
             #P
             'AAT': ['TTA', 'GGC', 'GCG'],
             'TTA': ['AAT', 'GGC', 'GCG'],
             'GGC': ['AAT', 'TTA', 'GCG'],
             'GCG': ['AAT', 'TTA', 'GGC'],
             #S
             'CAC': ['AGT', 'GCA', 'TTC', 'ATT', 'AAC'],
             'AGT': ['CAC', 'GCA', 'TTC', 'ATT', 'AAC'],
             'GCA': ['CAC', 'AGT', 'TTC', 'ATT', 'AAC'],
             'TTC': ['CAC', 'AGT', 'GCA', 'ATT', 'AAC'],
             'ATT': ['CAC', 'AGT', 'GCA', 'TTC', 'AAC'],
             'AAC': ['CAC', 'AGT', 'GCA', 'TTC', 'ATT'],
             #T
             'CGA': ['TGT', 'CAG', 'GCT'],
             'TGT': ['CGA', 'CAG', 'GCT'],
             'CAG': ['CGA', 'TGT', 'GCT'],
             'GCT': ['CGA', 'TGT', 'CAG'],
             #W
             'ATA': ['ATA'],
             #Y
             'GCC': ['AAA'],
             'AAA': ['GCC'],
             #V
             'AGG': ['CGT', 'CAA', 'CTA'],
             'CGT': ['AGG', 'CAA', 'CTA'],
             'CAA': ['AGG', 'CGT', 'CTA'],
             'CTA': ['AGG', 'CGT', 'CAA']
                }
            
    def find_3grams(self,seq):
        '''
        input: sequence of nucleotides
        output: string of sequential in-frame codons / 3grams
        '''
        #str_save = ''
        out_seq = []
        for i in range(2,len(seq) , 3):
            threeGram = seq[i-2],seq[i-1],seq[i]
            thrgrm_join = ''.join(threeGram)
            #str_save += thrgrm_join + ' '
            out_seq.append(thrgrm_join)
        return out_seq

    def __call__(self, seq):
        
        seq = self.find_3grams(seq)
        
        if self.aug_type.startswith('online'):
            random_idxs = list(range(len(seq)))
            random.shuffle(random_idxs)            
            random_num = random.uniform(0,1)
            
            if random_num >= self.args.prob_of_augmentation:
                
                if self.args.data_type == 'aav': num_substitutions = int( self.subst_frac * len(seq))
                else:
                    if int(len(seq) * random_num) >= 1: num_substitutions = int(len(seq) * random_num)
                    else: num_substitutions = 1
                    
                for k in range(num_substitutions):
                    seq[random_idxs[k]] = random.choice(self.nt_synonyms[seq[random_idxs[k]]])
            
        return seq


class encode_sequence(object):
    #Codon List
     #develop a dictionary of nucleotide codons and their amino acid counterparts
    def __init__(self,args,word_to_ix):
        
        self.ngram = args.ngram
        self.seq_type = args.seq_type
        self.word_to_ix = word_to_ix
        self.args = args

    def find_3grams(self,seq):
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
    
    def seq_to_cat(self, seq_df):
        '''
        input: dataframe of sequences & dictionary containing tokens in vocabulary
        output: out_idx: list of torch.Tensors of categorically encoded (vocab index) ngrams 
        '''
        out_idx = torch.tensor([self.word_to_ix[w] for w in seq_df if w != None and w != '' ], dtype=torch.long)
        return out_idx
    
    def __call__(self, x):
        x  = ''.join(x)
        if self.args.ngram == 'tri_unigram':
            seq_unigrams = [unigram for unigram in x]
            seq_trigrams = str.split(self.find_3grams(x)[:-1])#remove trailing whitespace
            x = seq_trigrams +  seq_unigrams
        elif self.args.ngram == 'trigram_only':
            x = str.split(self.find_3grams(x)[:-1])#remove trailing whitespace
    
        x_idx = self.seq_to_cat(x)
        return x_idx

class ToTensor(object):
    def __init__(self, dtype = torch.long):
        self.dtype = dtype

    def __call__(self, label):
        out_tensor = label.clone().detach()
        return out_tensor
    



#===========================   Convert Data to torch.DataLoader        ======================
def data_to_loader_online_nta(x_train,x_val, x_test, y_train,y_val, y_test, batch_size, args, sampler_weights = None):
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
    x_train = x_train.tolist()
    x_val = x_val.tolist()
    x_test = x_test.tolist()
    
    if args.seq_type == 'aa' and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    if args.seq_type == 'dna' and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'C', 'G', 'T']
    elif args.aug_type.startswith('online') and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'C', 'G', 'T']
    
    if args.seq_type == 'dna' and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    elif args.aug_type.startswith('online') and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    
    if args.seq_type == 'dna' and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
    elif args.aug_type.startswith('online') and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']

    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    
    
    if args.aug_type != 'online_none':
        train_data = CustomTorchDataset(x_train, y_train, args, transform = transforms.Compose([#reverse_translate(args),
                                                                                          nta_transform(args), 
                                                                                          encode_sequence(args,word_to_ix),
                                                                                          ToTensor()]))
    elif args.aug_type == 'online_none':
        train_data = CustomTorchDataset(x_train, y_train, args, transform = transforms.Compose([#reverse_translate(args),
                                                                                          #nta_transform(args), 
                                                                                          encode_sequence(args,word_to_ix),
                                                                                          ToTensor()]))
    
    val_data = CustomTorchDataset(x_val, y_val,  args, transform = transforms.Compose([#reverse_translate(args),
                                                                                      #nta_transform(args), 
                                                                                      encode_sequence(args,word_to_ix),
                                                                                      ToTensor()]))
    
    test_data = CustomTorchDataset(x_test, y_test,  args, transform = transforms.Compose([#reverse_translate(args),
                                                                                      #nta_transform(args), 
                                                                                      encode_sequence(args,word_to_ix),
                                                                                      ToTensor()]))
    
    if len(y_val) < batch_size : drop_last_val_bool = False
    else: drop_last_val_bool = True
    
    if len(y_train) < batch_size : drop_last_train_bool = False
    else: drop_last_train_bool = True
    
    #vocab lenghts are increased by 1 to include UNK token
    if args.ngram == 'trigram_only' and args.seq_type =='dna' : vocab_length = 62
    elif args.ngram == 'tri_unigram' and args.seq_type =='dna' : vocab_length = 66
    elif args.ngram == 'unigram' and args.seq_type =='dna' : vocab_length = 5
    elif args.seq_type =='aa' : vocab_length = 21
    
    if sampler_weights != None:
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement = True)
        
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








#===========================   Convert Data to torch.DataLoader        ======================
def data_to_loader_online_nta_single(x, y, batch_size, args, sampler_weights = None, shuffle = True, nta=False):
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
    

    y = y.tolist()
    x = x.tolist()
    
    if args.seq_type == 'aa' and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    if args.seq_type == 'dna' and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'C', 'G', 'T']
    elif args.aug_type.startswith('online') and args.ngram == 'unigram':
        vocabulary = ['UNK', 'A', 'C', 'G', 'T']
    
    if args.seq_type == 'dna' and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    elif args.aug_type.startswith('online') and args.ngram == 'tri_unigram':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
    
    if args.seq_type == 'dna' and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
    elif args.aug_type.startswith('online') and args.ngram == 'trigram_only':
        vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
        
    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    
    if nta == True:
        data = CustomTorchDataset(x, y, args, transform = transforms.Compose([nta_transform(args), 
                                                                                          encode_sequence(args,word_to_ix),
                                                                                          ToTensor()]))
    elif nta == False:
        data = CustomTorchDataset(x, y,  args, transform = transforms.Compose([encode_sequence(args,word_to_ix),
                                                                                          ToTensor()]))
        
    if len(y) < batch_size : drop_last_bool = False
    else: drop_last_bool = True

    if args.ngram == 'trigram_only' and args.seq_type =='dna' : vocab_length = 62
    elif args.ngram == 'tri_unigram' and args.seq_type =='dna' : vocab_length = 66
    elif args.ngram == 'unigram' and args.seq_type =='dna' : vocab_length = 5
    elif args.seq_type =='aa' : vocab_length = 21
    
    if sampler_weights != None:
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement = True)
        
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), 
                                                   drop_last=drop_last_bool, sampler=sampler, shuffle = shuffle)
    else:
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, 
                                                   collate_fn=Collater(vocab_length = vocab_length, pad_tok=0., args=args), drop_last=drop_last_bool)
        
    return data_loader
