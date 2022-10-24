#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import random
import numpy as np
import pandas as pd
import functools, itertools, operator

def get_nt_aa_dict(aug_type):
    #assign nucleotide to amino acid dictionary appropriate for augmentation type
    if aug_type == 'iterative' or aug_type == 'random' or aug_type == 'online':
        nt_aa_dict = nt_aa_dict_std
        #print('standard nt to aa dict')
    elif 'shuffle' in aug_type:
        nt_aa_dict = nt_aa_dict_codon_shuffle
        #print('shuffle nt to aa dict')
    elif 'balance' in aug_type:
        nt_aa_dict = nt_aa_dict_codon_balance
        #print('balance nt to aa dict')
    return nt_aa_dict


#=====================    Standard NT to AA ========================================
#standard nucleotide to amino acid relationship corresponding to codon degeneracy found in nature
nt_aa_dict_std = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'N': ['AAT', 'AAC'],
        'D': ['GAT', 'GAC'],
        'C': ['TGT', 'TGC'],
        'Q': ['CAA', 'CAG'],
        'E': ['GAA', 'GAG'],
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
        'K': ['AAA', 'AAG'],
        'M': ['ATG'],
        'F': ['TTT', 'TTC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        #'*': ['TAA', 'TGA', 'TAG']
        }


#=====================    Codon Shuffle NT to AA ====================================
#arbitrariliy shuffled codon to amino acid relationship
#shuffled from standard dictionary via the following: 
'''
codon_list = list(nt_aa_dict_std.values())
flattened_codon_list = [x for xs in codon_list for x in xs]
random.shuffle(flattened_codon_list) #shuffle codons randomly
codon_qty_per_aa = {}
for k, v in nt_aa_dict_std.items():
    codon_qty_per_aa[k] = len(v)
    
new = {}
counter = 0
for key in nt_aa_dict_std:
    tmp_list = []
    for i in range(len(nt_aa_dict_std[key])): 
        tmp_list.append(flattened_codon_list[counter])
        counter+=1
    new[key] = tmp_list

nt_aa_dict_codon_shuffle = new.copy()                 
'''
nt_aa_dict_codon_shuffle = {
  'A': ['GAG', 'CCG', 'CTT', 'TTG'],
 'R': ['GGG', 'GGA', 'AGA', 'ATC', 'CCA', 'CTG'],
 'N': ['TCG', 'TTT'],
 'D': ['ACT', 'CAT'],
 'C': ['GAA', 'TGG'],
 'Q': ['CGG', 'CTC'],
 'E': ['GAC', 'GTA'],
 'G': ['TGC', 'TCC', 'AGC', 'ACC'],
 'H': ['TCA', 'GTT'],
 'I': ['GTG', 'ACG', 'AAG'],
 'L': ['ATG', 'ACA', 'CCC', 'GAT', 'TAC', 'TAT'],
 'K': ['GGT', 'GTC'],
 'M': ['CCT'],
 'F': ['CGC', 'TCT'],
 'P': ['AAT', 'TTA', 'GGC', 'GCG'],
 'S': ['CAC', 'AGT', 'GCA', 'TTC', 'ATT', 'AAC'],
 'T': ['CGA', 'TGT', 'CAG', 'GCT'],
 'W': ['ATA'],
 'Y': ['GCC', 'AAA'],
 'V': ['AGG', 'CGT', 'CAA', 'CTA']
 }


#======================  Balanced Codon NT to AA ===================================
#artificially modified the nt to aa relationship such that each amino acid contains 3 possible codons
nt_aa_dict_codon_balance = {
        'A': ['GCT', 'GCC', 'GCA'],
        'R': ['CGT', 'CGC', 'CGA'],
        'N': ['AAT', 'AAC',  'GCG'],
        'D': ['GAT', 'GAC', 'CGG'],
        'C': ['TGT', 'TGC', 'AGA'],
        'Q': ['CAA', 'CAG', 'AGG'],
        'E': ['GAA', 'GAG', 'GGG'],
        'G': ['GGT', 'GGC', 'GGA'],
        'H': ['CAT', 'CAC', 'CTG'],
        'I': ['ATT', 'ATC', 'ATA'],
        'L': ['CTT', 'CTC', 'CTA'],
        'K': ['AAA', 'AAG','TTA'],
        'M': ['ATG', 'TTG', 'CCG'],
        'F': ['TTT', 'TTC', 'TCG'],
        'P': ['CCT', 'CCC', 'CCA'],
        'S': ['TCT', 'TCC', 'TCA'],
        'T': ['ACT', 'ACC', 'ACA'],
        'W': ['TGG', 'AGT', 'AGC'],
        'Y': ['TAT', 'TAC', 'ACG'],
        'V': ['GTT', 'GTC', 'GTA'],
        #'*': ['TAA', 'TGA', 'TAG']
        }




#=============   Iterative Augmentation - Sample Iteratively From Codon Space =================
def aa_to_nt_iterative(seq_list, target_list, aug_factor, nt_aa_dict):
    '''
    Function which produces aug_factor unique nucleotide sequences for each amino acid sequence in 
    seq_list. Nucleotide codons are iteratively sampled from codon space. The appropriate targets 
    (labels) are maintained for the augmented sequences.

    Parameters
    ----------
    seq_list : list or pandas.core.series.Series
        list or series of amino acid sequences
    target_list : list or pandas.core.series.Series
        list or series of 'targets' or class labels corresponding to the sequences
    aug_factor : int 
        the augmentation factor. the number of unique nucleotide sequences to create per protein sequence

    Returns
    -------
    out_df : pandas.core.frame.DataFrame
        pandas dataframe containing augmented nucleotide sequences

    '''
    seq_dict = {}
    target_dict = {}
    for k in range(len(seq_list)):
        seq_dict[k] = []
        nt_codons_per_residue = {}
        for i in range(len(seq_list[k])):
            #determine possible nt codons per aa position
            nt_codons_per_residue[str(i)] = nt_aa_dict[seq_list[k][i]]
            
        #use itertools product function to create a list of all possible combinations of nt codons  for given aa seq    
        nucleotides = list(itertools.islice(itertools.product(*nt_codons_per_residue.values()), aug_factor))
        #convert list of tuples to list of strings
        nucleotides = list(map(''.join,nucleotides))
        tmp_target_list = []
        for j in range(len(nucleotides)):  
            tmp_target_list.append(target_list[k])
        seq_dict[k] = (nucleotides)
        target_dict[k] = tmp_target_list
        
    return seq_dict, target_dict

#=============   Random Augmentation - Sample Randomly From Codon Space =================
#reference: https://stackoverflow.com/questions/48686767/how-to-sample-from-cartesian-product-without-repetition
def samples(list_of_sets):
        list_of_lists = list(map(list, list_of_sets))  # choice only works on sequences
        seen = set()  # keep track of seen samples
        while True:
            x = tuple(map(random.choice, list_of_lists))  # tuple is hashable
            if x not in seen:
                seen.add(x)
                yield x

def aa_to_nt_random(seq_list, target_list, aug_factor, nt_aa_dict):

    seq_out_dict = {}
    target_out_dict = {}

    for i in range(len(seq_list)):
        tmp_aug_factor = aug_factor
        seq = seq_list[i]
        list_of_possible_nts = []
        counter = int(0)
        for j in range(len(seq)):
            #determine possible nt codons per aa position
            list_of_possible_nts.append(nt_aa_dict[seq[j]])
        
        num_possible_seq_augs = functools.reduce(operator.mul, map(len, list_of_possible_nts), 1) #determine the number of possible augmentations available 
        
        gen = samples(list_of_possible_nts)        
        tmp_target_list = []
        tmp_seq_list = []
        
        #restrict augmentation loop to number of possible sequence combinations, if limiting
        if num_possible_seq_augs < aug_factor: tmp_aug_factor = num_possible_seq_augs
        
        for k in range(tmp_aug_factor):
            counter+=int(1)
            tmp_seq = next(gen)
            tmp_seq_list.append(''.join(tmp_seq))            
            tmp_target_list.append(target_list[i])
        seq_out_dict[i] = tmp_seq_list
        target_out_dict[i] = tmp_target_list
                
    return seq_out_dict, target_out_dict




#=====================        Offline Nucleotide Augmentation     ===========================
def nt_augmentation(input_seqs, args, aug_factor = 1, is_val_set = False):    
    '''
    Wrapper function to setup nucleotide augmentation based on a desired augmented data length. If
    is_val_set = True, then sequences will be backtranslated (from amino acids to nucleotides) without
    augmentation
    
    Parameters
    ----------
    input_seqs : list or pandas.core.series.Series
        list or series of amino acid sequences
    final_data_len : int
        desired length of final data set
    is_val_set : bool 
        whether or not input_seqs is a validation set. If is_val_set = True, backtranslation without
        augmentation is performed.
    
    Returns
    -------
    out_df : pandas.core.frame.DataFrame
        pandas dataframe containing augmented nucleotide sequences
    '''

    aug_type = args.aug_type
    


    data_len = len(input_seqs)
    if is_val_set == True:
        aug_factor = 1
    
    final_data_len = int(data_len * aug_factor)
    data = input_seqs.copy()
    aa_seq_list = data['aaseq'].to_list()
    target_list = data['target'].to_list()
    
    if isinstance(aug_factor, float): #if aug factor is float, round up to int and perform downsampling per end of algorithm
        aug_factor = int(np.ceil(aug_factor ))
    
    if 'iterative' in args.aug_type: 
        aa_to_nt = aa_to_nt_iterative
        #print('setting iterative augmentation')
    elif 'random' in args.aug_type: 
        aa_to_nt = aa_to_nt_random
        #print('setting random augmentation')
    elif 'online' in args.aug_type: 
        aa_to_nt = aa_to_nt_random
        #print('setting random online augmentation')
    else: 
        aa_to_nt = aa_to_nt_iterative
        #print('setting default iterative augmentation')
    
    codon_dictionary = get_nt_aa_dict(aug_type)
    
    seq_dict, target_dict = aa_to_nt(aa_seq_list, target_list = target_list, 
                                     aug_factor = aug_factor, nt_aa_dict = codon_dictionary)

    #randomly downsample augmented data set to desired length 
    if is_val_set == False:
        len_seq_dict = sum([len(x) for x in seq_dict.values()]) #number of total nucleotide sequences in dictionary
        
        #downsample augmented sequences by iteratively dropping one augmented nt seq from each 
            #aa seq until desired data size is reached
        if final_data_len < len_seq_dict:
            num_seqs_to_drop = int(len_seq_dict - final_data_len) 
            for i in range(num_seqs_to_drop): 
                seq_dict[i] =  np.random.choice(seq_dict[i], len(seq_dict[i]) -1, replace=False)
                target_dict[i] =  np.random.choice(target_dict[i], len(target_dict[i]) -1, replace=False)
    
    seq_out_list = []
    target_out_list = []
    
    for key in seq_dict:
        for seq_entry in seq_dict[key]:
            seq_out_list.append(seq_entry)
        for target_entry in target_dict[key]:
            target_out_list.append(target_entry)
    
    out_df = pd.DataFrame(seq_out_list)
    out_df.columns = ['dnaseq']
    out_df['target'] = target_out_list
    
    return out_df



