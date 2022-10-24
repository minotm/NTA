#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
from offline_nta_functions import *
import argparse

"""
This script serves to apply NTA to the various truncated aav 'seven vs rest' datasets 

Inputs
----------
csv files containing truncated aav seven vs rest data
csv files containing full aav seven vs rest train, validation, and test data

Outputs
-------
nucleotide augmented (or non-augmented, but backtranslated) csv files for pre-specified augmentation
quantities
"""    

def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--aug_type', type=str, default='iterative',
                        help = 'Augmentation Options: random, iterative, codon_shuffle, codon_balance, online')
    parser.set_defaults(augment=True)
    return parser


parser = create_parser()
args = parser.parse_args()

print(f'Nucleotide Augmentation of AAV Data Set with Augmentation Type = {args.aug_type}')

#================================== Load Data  ===================================
def rename_columns(df):
    df = df.copy()
    df = df.rename(columns = {'AASeq': 'aaseq', 'AgClass': 'target'})
    return df

data_path = '../data/aav/'
#data_path = 'cluster/scratch/mminot/data/aav'

data = pd.read_csv(data_path + 'aav_seven_vs_rest.csv')
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
if 'random' in args.aug_type or 'online' in args.aug_type:
    for i in range(5):
        out_file_str_test = data_path + f'aav_seven_vs_rest_test_dna_{args.aug_type}_{i}.csv'
        test_aug = nt_augmentation(input_seqs = test_full, args = args,
                                    aug_factor = 1, is_val_set=True)
    
        test_aug = test_aug.drop_duplicates(subset=['dnaseq'], keep='first')
        test_aug.to_csv(out_file_str_test, index=False)
else:
    out_file_str_test = data_path + f'aav_seven_vs_rest_test_dna_{args.aug_type}.csv'
    test_aug = nt_augmentation(input_seqs = test_full, args = args,
                                aug_factor = 1, is_val_set=True)

    test_aug = test_aug.drop_duplicates(subset=['dnaseq'], keep='first')
    test_aug.to_csv(out_file_str_test, index=False)

#================================== Augment Train Sets =============================
if not args.aug_type.startswith('online'):
    data_str_list = ['seven_vs_rest']
    
    aug_dict = {}
    aug_array =[2, 5, 10, 25, 50]
    aug_dict[0.005] = aug_array
    aug_dict[0.01] = aug_array
    aug_dict[0.05] = aug_array
    aug_dict[0.1] = aug_array
    aug_dict[0.25] = aug_array
    aug_dict[0.5] = aug_array
    aug_dict[0.75] = aug_array
    aug_dict[1.0] = aug_array
    
    
    for data_str in data_str_list:
        for key in aug_dict:
            truncate_factor = key
            for i in range(len(aug_dict[key])):
                aug_factor = aug_dict[key][i]
    
                file_str_train = data_path + 'aav_' + data_str + '_train_truncated_' + str(truncate_factor) + '.csv'
                outpath =  data_path
                
                if 'random' in args.aug_type:
                    for k in range(5):
                        out_file_str_train = outpath + 'aav_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{i+1}_{args.aug_type}_{k}.csv'
                        
                        train = pd.read_csv(file_str_train)
                        train_aug = nt_augmentation(input_seqs = train, args = args,
                                                    aug_factor = aug_factor, is_val_set=False)
        
                        train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                        train_aug.to_csv(out_file_str_train, index=False)
                else:
                    file_str_val = data_path + 'aav_' + data_str + '_val_truncated_' + str(truncate_factor) + '.csv'
                    out_file_str_train = outpath + 'aav_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{i+1}_{args.aug_type}.csv'
                    out_file_str_val = outpath + 'aav_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_{i+1}_{args.aug_type}.csv'
                    
                    train = pd.read_csv(file_str_train)
                    val = pd.read_csv(file_str_val)
                    
                    train_aug = nt_augmentation(input_seqs = train, args = args,
                                                aug_factor = aug_factor, is_val_set=False)
                    
                    val_aug = nt_augmentation(input_seqs = val, args = args,
                                                aug_factor = 1, is_val_set=True)
                    
                    train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                    train_aug.to_csv(out_file_str_train, index=False)
            
                    val_aug = val_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                    val_aug.to_csv(out_file_str_val, index=False)


#==========================Translate Truncated Train Sets Without Augmentation ============

data_str_list = ['seven_vs_rest']
truncate_factor_list = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
final_data_len_list = ['none']

for data_str in data_str_list:
    for truncate_factor in truncate_factor_list:
        for final_data_len in final_data_len_list:

            file_str_train = data_path + 'aav_' + data_str + '_train_truncated_' + str(truncate_factor) + '.csv'
            file_str_val = data_path + 'aav_' + data_str + '_val_truncated_' + str(truncate_factor) + '.csv'

            outpath = data_path
            
            train = pd.read_csv(file_str_train)
            val = pd.read_csv(file_str_val)
            
            aug_factor = 1
            aug_factor_val = 1
            
            if 'random' in args.aug_type or args.aug_type.startswith('online'):
                for i in range(5):
                    
                    out_file_str_train = outpath + 'aav_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_none_{args.aug_type}_{i}.csv'
                    out_file_str_val = outpath + 'aav_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_none_{args.aug_type}_{i}.csv'
    
                    train_aug = nt_augmentation(input_seqs = train, args = args,
                                    aug_factor = aug_factor, is_val_set=True)
                    
                    val_aug = nt_augmentation(input_seqs = val, args = args,
                                aug_factor = aug_factor_val, is_val_set=True)
    
                    train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                    train_aug.to_csv(out_file_str_train, index=False)
            
                    val_aug = val_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                    val_aug.to_csv(out_file_str_val, index=False)
            else:
                out_file_str_train = outpath + 'aav_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_none_{args.aug_type}.csv'
                out_file_str_val = outpath + 'aav_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + f'_dna_aug_none_{args.aug_type}.csv'
                
                train = pd.read_csv(file_str_train)
                val = pd.read_csv(file_str_val)
                
                train_aug = nt_augmentation(input_seqs = train, args = args,
                                aug_factor = aug_factor, is_val_set=True)
                
                val_aug = nt_augmentation(input_seqs = val, args = args,
                            aug_factor = aug_factor_val, is_val_set=True)

                train_aug = train_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                train_aug.to_csv(out_file_str_train, index=False)
        
                val_aug = val_aug.drop_duplicates(subset=['dnaseq'], keep='first')
                val_aug.to_csv(out_file_str_val, index=False)
