#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import torch
import torch.nn.parallel
import torch.optim
import pandas as pd
from utils import *
from train_routines  import *
import numpy as np

def batch_train_val_test_sets(args, trunc_factor=0.1):
    print(f'Batching {args.data_type} Data')
    print(f'Batching Task {args.data_set}')
    
    gb1_path = 'data/gb1/'
    aav_path = 'data/aav/'
    her2_path = 'data/her2/'
    
    if args.data_type == 'gb1':  data_path = gb1_path                
    elif args.data_type == 'aav' :  data_path = aav_path                                
    elif args.data_type == 'her2': data_path = her2_path                
    
    if args.data_type == 'her2': trunc_or_imbal = 'imbal'
    elif args.data_type == 'gb1' or args.data_type == 'aav': trunc_or_imbal = 'truncated'
    
    def drop_val_seqs(train_df, val_df, seq_str):
        train_df = train_df.copy()
        train_df['df'] = 'train'
        val_df = val_df.copy()
        val_df['df'] = 'val'
        frames = [train_df, val_df]
        common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
        concat_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
        concat_df = concat_df.drop_duplicates(subset=[seq_str],keep=False)
        out_df = concat_df[concat_df['df'] == 'train']
        return out_df
    
    if args.seq_file_type == 'aa' and not args.aug_type.startswith('online'):
        print('Using AA Data Files')
    
        train_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_train_{trunc_or_imbal}_{str(trunc_factor)}.csv')
        val_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_val_{trunc_or_imbal}_{str(trunc_factor)}.csv')
        test_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test.csv')
        
        train_df = drop_val_seqs(train_df, val_df, seq_str = 'aaseq')                
        train_df = train_df.reindex(np.random.permutation(train_df.index))
    
    elif args.seq_file_type =='dna' and not args.aug_type.startswith('online'):
        print('\nUsing DNA Data')
                        
        if 'random' in args.aug_type:
            train_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_train_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_{args.aug_factor}_{args.aug_type}_{args.seed}.csv')
            test_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test_dna_{args.aug_type}_{args.seed}.csv')
            val_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_val_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_none_{args.aug_type}_{args.seed}.csv')
            
            test_df_list = []
            for i in range(5):
                test_df_list.append(pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test_dna_{args.aug_type}_{i}.csv'))
            
        else: 
            train_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_train_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_{args.aug_factor}_{args.aug_type}.csv')
            val_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_val_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_{args.aug_factor}_{args.aug_type}.csv')
            test_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test_dna_{args.aug_type}.csv')
        
        train_df = train_df.reindex(np.random.permutation(train_df.index))
        train_df = drop_val_seqs(train_df, val_df, seq_str = 'dnaseq')
    
    elif args.aug_type.startswith('online'):

        test_df_list = []
        test_df = pd.DataFrame()
        use_spec_aug_type = args.aug_type
        if args.aug_type == 'online_none': args.aug_type ='online'
        
        train_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_train_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_none_{args.aug_type}_{args.seed}.csv')
        test_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test_dna_{args.aug_type}_{args.seed}.csv')
        val_df = pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_val_{trunc_or_imbal}_{str(trunc_factor)}_dna_aug_none_{args.aug_type}_{args.seed}.csv')
        
        test_df_list = []
        for i in range(5):
            test_df_list.append(pd.read_csv(data_path + f'{args.data_type}_{args.data_set}_test_dna_{args.aug_type}_{i}.csv'))
        
        args.aug_type = use_spec_aug_type
        
    def x_y_split(df, args):
        if 'dnaseq' in df.columns: df = df.rename(columns = {'dnaseq': 'seq'})
        elif 'aaseq' in df.columns: df= df.rename(columns = {'aaseq': 'seq'})
        x,y = df['seq'], df['target']
        return x, y
        
    x_train, y_train = x_y_split(train_df.copy(), args)
    x_val, y_val = x_y_split(val_df.copy(), args)    
    x_test, y_test = x_y_split(test_df.copy(), args)       
    
    #====================            Input Encoding            ============================
    if args.data_type == 'gb1' and  args.seq_type  == 'dna': seq_len = 12
    elif args.data_type == 'aav' and  args.seq_type  == 'dna': seq_len = 126
    elif args.data_type == 'her2' and  args.seq_type  == 'dna': seq_len = 30
    else: seq_len = None

    if not args.aug_type.startswith('online'):
        x_train = encode_ngrams(x_train, args, seq_len)
        x_val = encode_ngrams(x_val, args, seq_len)
        x_test = encode_ngrams(x_test, args, seq_len)
        
    if args.data_type == 'her2':
        class_sample_count = np.array( [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        weights_for_sampler = np.array([weight[t] for t in y_train])
        weights_for_sampler = torch.from_numpy(weights_for_sampler)
        weights_for_sampler = weights_for_sampler.double()
        
    else: weights_for_sampler = None
        
    if not args.aug_type.startswith('online'):
        train_loader, val_loader, test_loader = data_to_loader(x_train, x_val, x_test, y_train, y_val, y_test, batch_size = args.batch_size, 
                                                                args = args, sampler_weights = weights_for_sampler)
    else:
        
        train_loader, val_loader, test_loader = data_to_loader_online_nta(x_train, x_val, x_test, y_train, y_val, y_test, batch_size = args.batch_size, 
                                                                args = args, sampler_weights = weights_for_sampler)                                                                        
    
    if 'random' in args.aug_type: #encode multiple test sets
        test_loader_list = []
        for i in range(5):
            x_test_tmp, y_test_tmp = x_y_split(test_df_list[i].copy(), args)
            x_test_tmp = encode_ngrams(x_test_tmp, args, seq_len)                       
            test_loader_list.append(data_to_loader_single(x_test_tmp,y_test_tmp, batch_size = args.batch_size, 
                                                 args = args, sampler_weights = None, shuffle = False, is_test = True))
            
    elif args.aug_type.startswith('online'): #encode multiple test sets
        test_loader_list =  []
        for i in range(5):
            x_test_tmp, y_test_tmp = x_y_split(test_df_list[i].copy(), args)
            test_loader_list.append(data_to_loader_online_nta_single(x_test_tmp,y_test_tmp, batch_size = args.batch_size, 
                                                 args = args, sampler_weights = None, shuffle = False))            
        
    if 'random' in args.aug_type or args.aug_type.startswith('online'):  return train_loader, val_loader, test_loader_list, y_train, y_val, y_test
    else: return train_loader, val_loader, test_loader, y_train, y_val, y_test
    
    
    

















