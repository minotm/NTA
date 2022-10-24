#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
import numpy as np
from offline_nta_functions import *
import argparse

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


def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--aug_type', type=str, default='iterative',
                        help = 'Augmentation Options: random, iterative, codon_shuffle, codon_balance, online')
    parser.set_defaults(augment=True)
    return parser


parser = create_parser()
args = parser.parse_args()

print(f'Nucleotide Augmentation of Trastuzumab Data Set with Augmentation Type = {args.aug_type}')

#================================== Load Data  ===================================

her2_path = '../data/her2/'
test = pd.read_csv(her2_path + 'her2_seven_vs_rest_test.csv')


#================================== Augment Test Set =========================
if 'random' in args.aug_type or 'online' in args.aug_type:
    for i in range(5):
        out_file_str_test = her2_path + f'her2_seven_vs_rest_test_dna_{args.aug_type}_{i}.csv'
        test_aug = nt_augmentation(input_seqs = test, args = args,
                                    aug_factor = 1, is_val_set=True)
    
        test_aug = test_aug.drop_duplicates(subset=['dnaseq'], keep='first')
        test_aug.to_csv(out_file_str_test, index=False)
else:
    out_file_str_test = her2_path + f'her2_seven_vs_rest_test_dna_{args.aug_type}.csv'
    test_aug = nt_augmentation(input_seqs = test, args = args,
                                aug_factor = 1, is_val_set=True)

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
aug_factor_list = [2, 5, 10, 25, 'none']

for i in range(len(aug_factor_list)):
    for imbal_qty in class_imbalance_qty_list:
        
        aug_factor = aug_factor_list[i]
        train_file_str = her2_path + 'her2_seven_vs_rest_train_imbal_' + str(imbal_qty) + '.csv'
        val_file_str = her2_path + 'her2_seven_vs_rest_val_imbal_' + str(imbal_qty) + '.csv'
        
        train = pd.read_csv(train_file_str)
        val = pd.read_csv(val_file_str)
        
        train_pos = train[train['target'] == 1]
        train_neg = train[train['target'] == 0]
        val_pos = val[val['target'] == 1]
        val_neg = val[val['target'] == 0]
        
        
        if aug_factor != 'none':            
            if not args.aug_type.startswith('online'):    
                if 'random' in args.aug_type:
                    for k in range(5):
                        train_out_str = her2_path + 'her2_seven_vs_rest_train_imbal_'  + str(imbal_qty) + f'_dna_aug_{i + 1}_{args.aug_type}_{k}.csv'
                        final_data_len_train_per_class = int(np.round( aug_factor * len(train_pos) ) )
                        
                        if len(train_pos) < final_data_len_train_per_class:
                            train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                        aug_factor = aug_factor, is_val_set=False)
                        else:
                            train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                        aug_factor = 1, is_val_set=True)
        
                        train_neg_aug = nt_augmentation(input_seqs = train_neg, args = args,
                                                    aug_factor = 1,  is_val_set=True)
        
                        train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
                        train_out_df.to_csv(train_out_str, index=False)
                
                else:
                    final_data_len_train_per_class = int(np.round( aug_factor * len(train_pos) ) )
                    
                    if len(train_pos) < final_data_len_train_per_class:
                        train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                    aug_factor = aug_factor, is_val_set=False)
                    else:
                        train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                    aug_factor = 1, is_val_set=True)
        
                    train_neg_aug = nt_augmentation(input_seqs = train_neg, args = args,
                                                aug_factor = 1,  is_val_set=True)
                    
                    val_pos_aug = nt_augmentation(input_seqs = val_pos, args = args,
                                                aug_factor = 1, is_val_set=True)
                    val_neg_aug = nt_augmentation(input_seqs = val_neg, args = args,
                                                aug_factor = 1, is_val_set=True)
        
                    train_out_str = her2_path + 'her2_seven_vs_rest_train_imbal_'  + str(imbal_qty) + f'_dna_aug_{i + 1}_{args.aug_type}.csv'
                    val_out_str = her2_path + 'her2_seven_vs_rest_val_imbal_' + str(imbal_qty) + f'_dna_aug_{i + 1}_{args.aug_type}.csv'
                    
                    train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
                    val_out_df = combine_df_list_and_shuffle([val_pos_aug, val_neg_aug], keep=False)
                    
                    train_out_df.to_csv(train_out_str, index=False)
                    val_out_df.to_csv(val_out_str, index=False)

            
        elif aug_factor == 'none':
            if 'random' in args.aug_type:# or 'online' in args.aug_type:
                for k in range(5):
                    train_out_str = her2_path + 'her2_seven_vs_rest_train_imbal_'  + str(imbal_qty) + f'_dna_aug_none_{args.aug_type}_{k}.csv'
                    val_out_str = her2_path + 'her2_seven_vs_rest_val_imbal_' + str(imbal_qty) +  f'_dna_aug_none_{args.aug_type}_{k}.csv'
                
                    train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                aug_factor = 1, is_val_set=True)
                    train_neg_aug = nt_augmentation(input_seqs = train_neg, args = args,
                                                aug_factor = 1, is_val_set=True)
                                
                    val_pos_aug = nt_augmentation(input_seqs = val_pos, args = args,
                                                aug_factor = 1, is_val_set=True)
                    val_neg_aug = nt_augmentation(input_seqs = val_neg, args = args,
                                                aug_factor = 1, is_val_set=True)
                    
                    train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
                    val_out_df = combine_df_list_and_shuffle([val_pos_aug, val_neg_aug], keep=False)
                
                    train_out_df.to_csv(train_out_str, index=False)
                    val_out_df.to_csv(val_out_str, index=False)
            
            elif 'online' in args.aug_type:
                aug_factor = len(train_neg) /  len(train_pos)
                for k in range(5):
                    train_out_str = her2_path + 'her2_seven_vs_rest_train_imbal_'  + str(imbal_qty) + f'_dna_aug_none_{args.aug_type}_{k}.csv'
                    val_out_str = her2_path + 'her2_seven_vs_rest_val_imbal_' + str(imbal_qty) +  f'_dna_aug_none_{args.aug_type}_{k}.csv'
                    
                    train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                                aug_factor = aug_factor, is_val_set=False)
                    train_neg_aug = nt_augmentation(input_seqs = train_neg, args = args,
                                                aug_factor = 1, is_val_set=True)
                                
                    val_pos_aug = nt_augmentation(input_seqs = val_pos, args = args,
                                                aug_factor = 1, is_val_set=True)
                    val_neg_aug = nt_augmentation(input_seqs = val_neg, args = args,
                                                aug_factor = 1, is_val_set=True)
                    
                    train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
                    val_out_df = combine_df_list_and_shuffle([val_pos_aug, val_neg_aug], keep=False)
                
                    train_out_df.to_csv(train_out_str, index=False)
                    val_out_df.to_csv(val_out_str, index=False)
            else:
                train_pos_aug = nt_augmentation(input_seqs = train_pos, args = args,
                                            aug_factor = 1, is_val_set=True)
                train_neg_aug = nt_augmentation(input_seqs = train_neg, args = args,
                                            aug_factor = 1, is_val_set=True)
                            
                val_pos_aug = nt_augmentation(input_seqs = val_pos, args = args,
                                            aug_factor = 1, is_val_set=True)
                val_neg_aug = nt_augmentation(input_seqs = val_neg, args = args,
                                            aug_factor = 1, is_val_set=True)
                
                train_out_str = her2_path + 'her2_seven_vs_rest_train_imbal_'  + str(imbal_qty) + f'_dna_aug_none_{args.aug_type}.csv'
                val_out_str = her2_path + 'her2_seven_vs_rest_val_imbal_' + str(imbal_qty) + f'_dna_aug_none_{args.aug_type}.csv'
                
                train_out_df = combine_df_list_and_shuffle([train_pos_aug, train_neg_aug], keep=False)
                val_out_df = combine_df_list_and_shuffle([val_pos_aug, val_neg_aug], keep=False)
                
                train_out_df.to_csv(train_out_str, index=False)
                val_out_df.to_csv(val_out_str, index=False)

