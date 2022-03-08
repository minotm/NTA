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


"""
This script serves to apply NTA to the various truncated gb1 'three vs rest' datasets 

Inputs
----------
csv files containing truncated gb1 three vs rest data
csv files containing full aav three vs rest train, validation, and test data

Outputs
-------
nucleotide augmented (or non-augmented, but backtranslated) csv files for pre-specified augmentation
quantities
"""    


print('Nucleotide Augmentation of GB1 Data Set...')

#================================== Load Data  ===================================
def rename_columns(df):
    df = df.copy()
    df = df.rename(columns = {'AASeq': 'aaseq', 'AgClass': 'target'})
    return df

aug_factor = 1
data_path = '../data/gb1/'
out_file_str_test = data_path + 'gb1_three_vs_rest_test_dna.csv'

data = pd.read_csv(data_path + 'three_vs_rest_truncated_seqs.csv')
train_full = data[data['set'] == 'train']
val_full = train_full[train_full['validation'] == True]
test_full = data[data['set'] == 'test']

train_full = rename_columns(train_full)
val_full = rename_columns(val_full)
test_full = rename_columns(test_full)

#Function to drop duplicates from two dataframes
def drop_test_seqs(train_df, test_df, seq_name):
    train_df = train_df.copy()
    train_df['df'] = 'train'
    test_df = test_df.copy()
    test_df['df'] = 'test'
    frames = [train_df.copy(),test_df.copy()]
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    concat_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    concat_df = concat_df.drop_duplicates(subset=[seq_name],keep=False)
    out_df = concat_df[concat_df['df'] == 'train']
    out_df.drop(columns = ['df'])
    return out_df


train_full = drop_test_seqs(train_full, test_full, 'aaseq')
train_full = drop_test_seqs(train_full, val_full, 'aaseq')




#================================== Augment Test Set =========================

augmented_data_len_test = int(np.round(len(test_full['aaseq'])*aug_factor))

test_aug = nt_augmentation(input_seqs = test_full, 
                            final_data_len = augmented_data_len_test, is_val_set=True)

test_aug = test_aug.drop_duplicates(subset=['dnaseq'], keep='first')
test_aug.to_csv(out_file_str_test, index=False)




#================================== Augment Train Sets =============================

data_str_list = ['three_vs_rest']
final_data_len_list = [300, 3000, 30000, 300000, 3000000]

aug_dict = {}
aug_dict[0.01] = [300, 1500, 3000]
aug_dict[0.05] = [300, 3000, 30000]
aug_dict[0.1] = [3000, 30000, 300000]
aug_dict[0.25] = [3000, 30000, 300000]
aug_dict[0.5] = [6000, 30000, 300000]
aug_dict[0.75] = [6000, 30000, 300000]
aug_dict[1.0] = [6000, 30000, 300000]

for data_str in data_str_list:
    for key in aug_dict:
        truncate_factor = key
        for i in range(len(aug_dict[key])):
            final_data_len = aug_dict[key][i]
            
            file_str_train = data_path + 'gb1_' + data_str + '_train_truncated_' + str(truncate_factor) + '.csv'
            file_str_val = data_path + 'gb1_' + data_str + '_val_truncated_' + str(truncate_factor) + '.csv'

            outpath = '../data/gb1/'
            out_file_str_train = outpath + 'gb1_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{i+1}.csv'
            out_file_str_val = outpath + 'gb1_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{i+1}.csv'
            
            train = pd.read_csv(file_str_train)
            val = pd.read_csv(file_str_val)            
            augmented_data_len = final_data_len
            
            if truncate_factor != 'none':
                train_aug = nt_augmentation(input_seqs = train, 
                                            final_data_len = augmented_data_len, is_val_set=False)
                
                val_aug = nt_augmentation(input_seqs = val, 
                                            final_data_len = len(val), is_val_set=True)
            
            if truncate_factor == 'none':
                augmented_data_len = len(train)
                augmented_data_len_val = len(val)
                
                train_aug = nt_augmentation(input_seqs = train, 
                                final_data_len = augmented_data_len, is_val_set=True)
                
                val_aug = nt_augmentation(input_seqs = val, 
                            final_data_len = augmented_data_len_val, is_val_set=True)
                out_file_str_train = outpath + 'gb1_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna.csv'
                out_file_str_val = outpath + 'gb1_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna.csv'
            
            train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
            train_aug.to_csv(out_file_str_train, index=False)
    
            val_aug = val_aug.drop_duplicates(subset=['dnaseq'], keep='first')
            val_aug.to_csv(out_file_str_val, index=False)



#==========================Translate Truncated Train Sets Without Augmentation ============

data_str_list = ['three_vs_rest']
truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
final_data_len_list = ['none']

for data_str in data_str_list:
    for truncate_factor in truncate_factor_list:
        for final_data_len in final_data_len_list:

            file_str_train = data_path + 'gb1_' + data_str + '_train_truncated_' + str(truncate_factor) + '.csv'
            file_str_val = data_path + 'gb1_' + data_str + '_val_truncated_' + str(truncate_factor) + '.csv'

            outpath = data_path
            out_file_str_train = outpath + 'gb1_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{final_data_len}.csv'
            out_file_str_val = outpath + 'gb1_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{final_data_len}.csv'
            
            train = pd.read_csv(file_str_train)
            val = pd.read_csv(file_str_val)
            
            augmented_data_len = len(train)
            augmented_data_len_val = len(val)
            
            train_aug = nt_augmentation(input_seqs = train, 
                            final_data_len = augmented_data_len, is_val_set=True)
            
            val_aug = nt_augmentation(input_seqs = val, 
                        final_data_len = augmented_data_len_val, is_val_set=True)

            train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
            train_aug.to_csv(out_file_str_train, index=False)
    
            val_aug = val_aug.drop_duplicates(subset=['dnaseq'], keep='first')
            val_aug.to_csv(out_file_str_val, index=False)
