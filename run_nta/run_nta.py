#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
from offline_nta_functions import *
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--aug_type', type=str, default='iterative',
                        help = 'Augmentation Options: random, iterative, codon_shuffle, codon_balance, online')
    parser.add_argument('--outfile_prefix', type=str, default='data_nta',
                        help='name of output file')
    parser.add_argument('--input_file', type=str, default='../data/Example_NTA_Input_GB1.csv',
                        help='name of input csv file')
    parser.add_argument('--aug_factor', type=int, default=10,
                        help='desired length of final augmented data set')         
    parser.add_argument('--rev_translate_only', type=str, default='False',
                        help='To reverse translate without augmentation. Options: True, False')         
    parser.set_defaults(augment=True)
    
    return parser


parser = create_parser()
args = parser.parse_args()

if args.rev_translate_only == 'False': is_val = False
elif args.rev_translate_only == 'True': is_val = True

if args.aug_type == 'online': is_val = True

#================================== Load Data  ===================================
outfile_prefix = args.outfile_prefix
data = pd.read_csv(args.input_file)
data['aaseq'] = data['aaseq'].str.upper()

if is_val == True: outfile_suffix = '_reverse_translated.csv'
elif is_val == False: outfile_suffix = f'_{args.aug_type}_augmented_aug_factor_{args.aug_factor}.csv'

if is_val == True and args.aug_type == 'online': outfile_suffix = f'_reverse_translated_for_online_nta.csv'

#======================   Run NTA & Save to File =====================================
out_df = nt_augmentation(input_seqs = data, args= args, aug_factor = args.aug_factor, is_val_set=is_val)
out_df = out_df.drop_duplicates(subset=['dnaseq'], keep='first')
out_df.to_csv(outfile_prefix + outfile_suffix, index=False)





