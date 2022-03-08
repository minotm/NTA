#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""


from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from nt_augment import *
import re

"""
This script serves to apply NTA to the various truncated trastuzumab (her2) datasets 

Inputs
----------
csv files containing truncated trastuzumab (her2) data
csv files containing full aav three vs rest train, validation, and test data

Outputs
-------
nucleotide augmented (or non-augmented, but backtranslated) csv files for pre-specified augmentation
quantities
"""    


print('Nucleotide Augmentation of Trastuzumab Data Set...')

#================================== Load Data  ===================================

her2_path = '../data/her2/'
out_file_str_test = her2_path + 'her2_test_dna.csv'
test = pd.read_csv(her2_path + 'her2_test.csv')


#================================== Augment Test Set =========================
aug_factor = 1
augmented_data_len_test = int(np.round(len(test['aaseq'])*aug_factor))

test_aug = nt_augmentation(input_seqs = test, 
                            final_data_len = augmented_data_len_test, is_val_set=True)

test_aug = test_aug.drop_duplicates(subset=['dnaseq'], keep='first')
test_aug.to_csv(out_file_str_test, index=False)


#================================== Augment Train & Val Sets ========================

def combine_df_list_and_shuffle(df_list, keep = False):
    frames = df_list
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    combined_df = pd.concat([df[common_cols] for df in frames], ignore_index=True).drop_duplicates(subset='dnaseq', keep=keep)
    np.random.seed(0)
    combined_df = combined_df.reindex(np.random.permutation(combined_df.index))
    return combined_df



class_imbalance_qty_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0] 
aug_factor_list = [2, 5, 10, 'none']

for aug_factor in aug_factor_list:
    for imbal_qty in class_imbalance_qty_list:
        
        train_file_str = her2_path + 'her2_train_imbal_' + str(imbal_qty) + '.csv'
        val_file_str = her2_path + 'her2_val_imbal_' + str(imbal_qty) + '.csv'
        
        train_out_str = her2_path + 'her2_train_imbal_'  + str(imbal_qty) + '_aug_' + str(aug_factor) +  '_dna.csv'
        val_out_str = her2_path + 'her2_val_imbal_' + str(imbal_qty) + '_aug_' + str(aug_factor) + '_dna.csv'
        
        train = pd.read_csv(train_file_str)
        val = pd.read_csv(val_file_str)
        
        train_pos = train[train['target'] == 1]
        train_neg = train[train['target'] == 0]
        val_pos = val[val['target'] == 1]
        val_neg = val[val['target'] == 0]
        
        if aug_factor != 'none' and aug_factor != 'double_neg' and aug_factor != 'test':            
            
            final_data_len_train_per_class = int(np.round( aug_factor * len(train_pos) ) )
            
            if len(train_pos) < final_data_len_train_per_class:
                train_pos_aug = nt_augmentation(input_seqs = train_pos, 
                                            final_data_len = final_data_len_train_per_class, is_val_set=False)
            else:
                train_pos_aug = nt_augmentation(input_seqs = train_pos, 
                                            final_data_len = len(train_pos), is_val_set=True)

            train_neg_aug = nt_augmentation(input_seqs = train_neg, 
                                        final_data_len = len(train_neg),  is_val_set=True)
            
            val_pos_aug = nt_augmentation(input_seqs = val_pos, 
                                        final_data_len = len(val_pos), is_val_set=True)
            val_neg_aug = nt_augmentation(input_seqs = val_neg, 
                                        final_data_len = len(val_neg), is_val_set=True)

            
        elif aug_factor == 'none':
            train_pos_aug = nt_augmentation(input_seqs = train_pos, 
                                        final_data_len = len(train_pos), is_val_set=True)
            train_neg_aug = nt_augmentation(input_seqs = train_neg, 
                                        final_data_len = len(train_neg), is_val_set=True)
                        
            val_pos_aug = nt_augmentation(input_seqs = val_pos, 
                                        final_data_len = len(val_pos), is_val_set=True)
            val_neg_aug = nt_augmentation(input_seqs = val_neg, 
                                        final_data_len = len(val_neg), is_val_set=True)
            
        train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
        val_out_df = combine_df_list_and_shuffle([val_pos_aug, val_neg_aug], keep=False)
        

        train_out_df.to_csv(train_out_str, index=False)
        val_out_df.to_csv(val_out_str, index=False)

