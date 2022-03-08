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
from preprocessing.nt_augment import *
import argparse

"""
This script serves to apply NTA to a user defined dataset saved as a CSV file in the same folder

Inputs
----------
input_file:  type - csv
    csv file containing protein sequence data in one column and lables in another. Train, Val, & Test 
    sets should be in seperate csv files

rev_translate_only: type - str / bool
    if `True`, sequences will only be reverse translated, without augmentation (i.e. for test/val sets). 
    `False` if augmentation desired
    
aug_size: type - int
    desired length of final augmented data set (number of nucleotide sequences)

outfile_prefix: type - str 
        pefix of output nta data file, omitting .csv, i.e. `data_nta`. If file is only reverse translated, 
        `_reverse_translated` will be added to prefix. `nta` will be added to prefix if file is augmented

Outputs
-------
nucleotide augmented (train) or non-augmented, but reverse translated (validation, test) csv files for pre-specified augmentation
quantities
"""    


def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--rev_translate_only', type=str, default='False',
                        help='Whether data set is validation or test set, in which case, aa sequences will be reverse translated and not augmented. Options: True / False')
    parser.add_argument('--outfile_prefix', type=str, default='data_nta',
                        help='name of output file')
    parser.add_argument('--input_file', type=str, default='data.csv',
                        help='name of input csv file')
    parser.add_argument('--aug_size', type=int, default=1000,
                        help='desired length of final augmented data set')         
    parser.set_defaults(augment=True)
    
    return parser


parser = create_parser()
args = parser.parse_args()


if args.rev_translate_only == 'False': is_val = False
elif args.rev_translate_only == 'True': is_val = True

print('Nucleotide Augmentation of Data Set...')
#================================== Load Data  ===================================
outfile_prefix = args.outfile_prefix
data = pd.read_csv(args.input_file)
data['aaseq'] = data['aaseq'].str.upper()
#====================    Reverse Translate Data if Val or Test Set =========================
if is_val == True:
    
    augmented_data_len_test = int(np.round(len(data['aaseq'])*aug_factor))
    out_df = nt_augmentation(input_seqs = data, 
                                final_data_len = augmented_data_len_test, is_val_set=True)
    out_df = out_df.drop_duplicates(subset=['dnaseq'], keep='first')
    out_df.to_csv(outfile_prefix + '_reverse_translated.csv' , index=False)


#================================== Augment Train Sets =============================
elif is_val == False:

    data_aug = nt_augmentation(input_seqs = data, 
                                final_data_len = args.aug_size, is_val_set=False)
    
    data_aug = data_aug.drop_duplicates(subset=['dnaseq'], keep='first')
    data_aug.to_csv(outfile_prefix + f'_to_{args.aug_size}_seqs.csv', index=False)
