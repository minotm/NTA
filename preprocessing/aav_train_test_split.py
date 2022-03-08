#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nt_augment import *

print('Now Executing AAV Train/Val/Test Splitting...')
aav_path = '../data/aav/'

#load full aav data set & truncate amino acids to variable region only
full_data= pd.read_csv(aav_path + 'aav_seven_vs_many_full_data.csv')
seq_list = full_data['sequence'].to_list()
out_list = []

#match the variable region of the sequence and append the constant 7 aas to it at the end QAATADVN
for seq in seq_list:
    m = re.search(r'KTNVDIEKVMIT(.*)QAATADVNTQGV',seq)
    out_list.append(m.group(1))
    
full_data = full_data.drop(columns = ['sequence'])
full_data['aaseq'] = out_list
full_data.to_csv(aav_path + 'aav_seven_vs_rest.csv', index = False)

seven_vr = pd.read_csv(aav_path + 'aav_seven_vs_rest.csv')
test = seven_vr[seven_vr['set'] == 'test']
test.to_csv(aav_path + 'aav_seven_vs_rest_test.csv')

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

truncate_factor_list = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

for truncate_factor in truncate_factor_list:    
    data = seven_vr.copy()
            
    train = data[data['set'] == 'train']
    val = data[data['validation'] == True]
    test = data[data['set'] == 'test']
    
    train = drop_val_seqs(train, val)

    train = bin_targets_to_classes_aav(train)
    val = bin_targets_to_classes_aav(val)
    test = bin_targets_to_classes_aav(test)
    
    
    if truncate_factor != 1.0:
        train_truncated, x_discard, y_train, y_discard = train_test_split(train, train['target'], test_size = 1 - truncate_factor,
                                                                  random_state = 1, shuffle = True, stratify = train['class'])            
        val_truncated, x_discard, y_val, y_discard = train_test_split(val, val['target'], test_size = 1 - truncate_factor,
                                                          random_state = 1, shuffle = True, stratify = val['class'])
    elif truncate_factor == 1.0:
        train_truncated = train
        val_truncated = val

    train_truncated = train_truncated.drop(columns = ['df'])
    
    out_path = '../data/aav/'
    data_str = 'seven_vs_rest'
    out_str = out_path  + '/aav_' + data_str + '_train_' + 'truncated_' + str(truncate_factor) + '.csv'
    train_truncated.to_csv(out_str, index=False)
    
    out_str = out_path + '/aav_' + data_str + '_val_' + 'truncated_' + str(truncate_factor) + '.csv'
    val_truncated.to_csv(out_str, index=False)
