#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from nt_augment import *

print('Now Executing GB1 Train/Val/Test Splitting...')
gb1_path = '../data/gb1/'

#truncate full gb1 3 vs Rest data set to just the 4 variable amino acids

three_v_rest_full = pd.read_csv(gb1_path + 'gb1_three_vs_rest_full_data.csv')
#GB1 varied residues: V39, D40, G41 and V54
#   therefore, indices = 38, 39, 40, 53
def return_df_truncated_seqs(input_df):
    trunc_seq_list, seq_class_list, set_list, val_list = [], [],[],[]
    
    for i in range(len(input_df)):
        tmp_seq = input_df['sequence'].iloc[i]
        tmp_class = input_df['target'].iloc[i]
        tmp_set = input_df['set'].iloc[i]
        tmp_val = input_df['validation'].iloc[i]
        tmp_seq_new = tmp_seq[38] + tmp_seq[39] + tmp_seq[40] + tmp_seq[53]
        
        trunc_seq_list.append(tmp_seq_new)
        seq_class_list.append(tmp_class)
        set_list.append(tmp_set)
        val_list.append(tmp_val)
        
    out_df = pd.DataFrame()
    out_df['aaseq'] = trunc_seq_list
    out_df['target'] = seq_class_list
    out_df['set'] = set_list
    out_df['validation'] = val_list
    return out_df

three_vr_trunc = return_df_truncated_seqs(three_v_rest_full)
three_vr_trunc.to_csv(gb1_path + 'three_vs_rest_truncated_seqs.csv', index=False)

three_vr = pd.read_csv(gb1_path + 'three_vs_rest_truncated_seqs.csv')

def drop_val_seqs(df, val_df):
    train = df.copy()
    val = val_df.copy()
    train['df'] = 'train'
    val['df'] = 'val'
    frames = [train,val]
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    concat_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    df = concat_df.drop_duplicates(subset=['aaseq'],keep=False)
    df = df[df['df'] == 'train']
    return df

def bin_targets_to_classes_gb1(input_df):
    df = input_df.copy()
    class_list = []

    for i in range(len(df)):
        if df['target'].iloc[i] <= 0.7:
            class_list.append(0)
        elif df['target'].iloc[i] > 0.7 and df['target'].iloc[i] <= 1.5:
            class_list.append(1)
        elif df['target'].iloc[i] > 1.5:
            class_list.append(2)
    df['class'] = class_list
    return df

def bin_targets_to_classes_aav(input_df):
    df = input_df.copy()
    class_list = []

    for i in range(len(df)):
        if df['target'].iloc[i] <= -4:
            class_list.append(0)
        elif df['target'].iloc[i] > -4 and df['target'].iloc[i] <= -2:
            class_list.append(1)
        elif df['target'].iloc[i] > -2 and df['target'].iloc[i] <= 0:
            class_list.append(2)
        elif df['target'].iloc[i] > 0 and df['target'].iloc[i] <= 2:
            class_list.append(3)
        elif df['target'].iloc[i] > 2:
            class_list.append(4)
    df['class'] = class_list
    return df


truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

for truncate_factor in truncate_factor_list:

    data = three_vr.copy()
            
    train = data[data['set'] == 'train']
    val = data[data['validation'] == True]
    test = data[data['set'] == 'test']
    
    train = drop_val_seqs(train, val)
   
    
    train = bin_targets_to_classes_gb1(train)
    val = bin_targets_to_classes_gb1(val)
    test = bin_targets_to_classes_gb1(test)
    
    #train = bin_targets_to_classes_aav(train)
    #val = bin_targets_to_classes_aav(val)
    #test = bin_targets_to_classes_aav(test)
    
    
    if truncate_factor != 1.0 and truncate_factor != 0.01:
        train_truncated, x_discard, y_train, y_discard = train_test_split(train, train['target'], test_size = 1 - truncate_factor,
                                                                  random_state = 1, shuffle = True, stratify = train['class'])            
        val_truncated, x_discard, y_val, y_discard = train_test_split(val, val['target'], test_size = 1 - truncate_factor,
                                                          random_state = 1, shuffle = True, stratify = val['class'])
    elif truncate_factor == 0.01:
        #the smallest validation set has to manually be separated. sequence length limitations cauase sklearn train_test_split to throw an error
        val_0 = val[val['class'] == 0].sample(n=1, random_state = 1)
        val_1 = val[val['class'] == 1].sample(n=1, random_state = 1)
        val_2 = val[val['class'] == 2].sample(n=1, random_state = 1)
        val_truncated = val_0.append(val_1, ignore_index = True)
        val_truncated = val_truncated.append(val_2, ignore_index = True)
        
        train_truncated, x_discard, y_train, y_discard = train_test_split(train, train['target'], test_size = 1 - truncate_factor,
                                                                  random_state = 1, shuffle = True, stratify = train['class'])      
    elif truncate_factor == 1.0:
        train_truncated = train
        val_truncated = val
        
    train_truncated = train_truncated.drop(columns = ['df'])
    
    out_path = '../data/gb1/'
    data_str = 'three_vs_rest'
    
    
    out_str = out_path  + 'gb1_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + '.csv'
    train_truncated.to_csv(out_str, index=False)
    
    out_str = out_path + 'gb1_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + '.csv'
    val_truncated.to_csv(out_str, index=False)
