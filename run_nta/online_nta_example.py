#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import random
import torch
from torch.utils.data import Dataset
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--aug_type', type=str, default='random',
                        help = 'Augmentation Options: online, none')
    parser.add_argument('--subst_frac', type=int, default=0.5,
                        help='desired length of final augmented data set')           
    parser.add_argument('--t_aug', type=int, default=0.5,
                        help='threshold for p_aug [0,1], above which one or more codons will be substituted for a given nucleotide sequence')      
    parser.add_argument('--ngram', type=str, default='tri_unigram',
                        help='DNA ngram encoding options: unigram, trigram_only, tri_unigram')      
    parser.add_argument('--seq_type', type=str, default='dna',
                        help='Sequence type options: dna, aa')      
    parser.set_defaults(augment=True)
    return parser


parser = create_parser()
args = parser.parse_args()

#============================================================================
#=========================     Online Augmentaiton Transforms =========================
#============================================================================
'''
    The following code details the transforms used for online nucleotide augmentation via codon substitution
    Relevant args are provided in this file as reference and should be modified for personal use
    
    INPUT:
        x: list of nucleotide sequences
        
        y: list of labels/targets
        
        word_to_idx: a mapping of input strings (amino acids or DNA nucleotides / codons) to categorical values
            example creation of word_to_idx:
                vocabulary = ['UNK', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                word_to_ix = {word: i for i, word in enumerate(vocabulary)}


    To use online NTA, one would create a Torch Dataset (here we use our CustomTorchDataSet) with the 
    following transforms:
        data = CustomTorchDataset(x, y, args, transform = transforms.Compose([nta_transform(args), 
                                                                                          encode_sequence(args,word_to_ix),
                                                                                          ToTensor()]))
'''





class nta_transform(object):
    #Codon List
     #develop a dictionary of nucleotide codons and their amino acid counterparts
    def __init__(self, args):
        
        self.aug_type = args.aug_type
        self.subst_frac = args.subst_frac
        self.args = args

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
        
    def find_3grams(self,seq):
        '''
        input: sequence of nucleotides
        output: string of sequential in-frame codons / 3grams
        '''
        out_seq = []
        for i in range(2,len(seq) , 3):
            threeGram = seq[i-2],seq[i-1],seq[i]
            thrgrm_join = ''.join(threeGram)
            out_seq.append(thrgrm_join)
        return out_seq

    def __call__(self, seq):
        
        seq = self.find_3grams(seq)
        
        if self.aug_type == 'online':
            random_idxs = list(range(len(seq)))
            random.shuffle(random_idxs)            
            random_num = random.uniform(0,1)
            
            if random_num >= self.args.t_aug:
                num_substitutions = int( self.subst_frac * len(seq))
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
   