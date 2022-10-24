#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


print('Now Executing Trastuzumab Train/Val/Test Splitting...')

"""
This script serves to split the trastuzumab (her2) data into training, validation, and testing sets. The train
and validation sets are selected to be an edit distance of 7 or less from the wild type trastuzumab. The 
test set is an edit distance of 8 or greater from the wild type

Inputs
----------
labeled trastuzumab data from Mason et. al 2021 github repo: https://github.com/dahjan/DMS_opt
mHER_H3_AgPos.csv
mHER_H3_AgNeg.csv

Outputs
-------
csv files containing training, validation, and testing sets
"""    

def add_LD_to_df(antigen_ID, data_frame):
    '''
    Function to add Edit Distance (Levenshtein Distance) from wt for each sequence to dataframe
    
    Parameters
    ----------
    antigen_ID : str
        corresponds to the antigen identity. in this case, her2
    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data

    Returns
    -------
    data_frame : pandas.DataFrame
        input dataframe with an added column containing the Levenshtein Distance from the wild type
        for each sequence in the dataframe
    '''
    
    if antigen_ID == 'her2':
        wt_str = 'WGGDGFYAMK'
    LD_arr = []
    for i in range(len(data_frame)):
        LD_arr.append( levenshtein_distance(wt_str, data_frame['AASeq'].iloc[i]) )
    data_frame['LD'] = LD_arr
    
    return data_frame

def class_balance_binary(data_frame):
    '''
    Function to class balance dataset
    
    Parameters
    ----------

    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data

    Returns
    -------
    data_frame : pandas.DataFrame
        class balanced dataframe. number of positive examples is equal to the number of negatives
    '''
    positives = data_frame[data_frame['AgClass'] == 1].copy()
    negatives = data_frame[data_frame['AgClass'] == 0].copy()
    min_list = min([len(ls) for ls in [positives, negatives]])
    positives = positives[: int(np.round(min_list))] 
    negatives = negatives[: int(np.round(min_list))] 
    return positives, negatives


her2_path_local = '../data/her2/'

pos = pd.read_csv(her2_path_local + 'mHER_H3_AgPos.csv')
neg = pd.read_csv(her2_path_local + 'mHER_H3_AgNeg.csv')

def combine_df_list_and_shuffle(df_list, keep = False):
    '''
    combines two dataframes, drops duplicates, & shuffles
    
    Parameters
    ----------

    data_frame : pandas.DataFrame
        dataframe containing all sequence & label data
    keep: bool
        whether or not to keep duplicates

    Returns
    -------
    data_frame : pandas.DataFrame
        combined, shuffled dataframe
    '''
    frames = df_list
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    combined_df = pd.concat([df[common_cols] for df in frames], ignore_index=True).drop_duplicates(subset='AASeq', keep=keep)
    np.random.seed(0)
    combined_df = combined_df.reindex(np.random.permutation(combined_df.index))
    return combined_df

all_data_frames = [pos.copy(), neg.copy()]
data_frame = combine_df_list_and_shuffle(all_data_frames, keep = False)
data_frame = add_LD_to_df('her2', data_frame)


selected_LD_split = 7
train_df = data_frame[data_frame['LD'] <= selected_LD_split]
test_df_initial = data_frame[data_frame['LD'] > selected_LD_split]

#Function to drop duplicates from two dataframes
def drop_test_seqs(train_df, test_df, seq_name):
    '''
    Function serves as a check to prevent dataleakage between training & test or training & val sets

    Parameters
    ----------
    train_df : pandas.DataFrame
        train dataframe
    test_df : pandas.DataFrame
        test dataframe
    seq_name : str
        corresponds to the dataframe column name containing sequences.

    Returns
    -------
    out_df : TYPE
        train dataframe without test sequences
    '''
    train_df = train_df.copy()
    train_df['df'] = 'train'
    test_df_copy = test_df.copy()
    test_df_copy['df'] = 'test'
    frames = [train_df.copy(),test_df_copy]
    common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
    concat_df = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    concat_df = concat_df.drop_duplicates(subset=[seq_name],keep=False)
    out_df = concat_df[concat_df['df'] == 'train']
    return out_df

train_df = drop_test_seqs(train_df, test_df_initial, 'AASeq')

def drop_and_rename_columns(df):
    df = df.copy()
    df = df.rename(columns = {'AASeq': 'aaseq', 'AgClass': 'target'})
    df = df.drop(columns = ['Unnamed: 0', 'Fraction', 'NucSeq', 'Count', 'df'])
    return df
    
#Balance test set & save to csv 
test_df = test_df_initial.copy()
test_df['df'] = 'test' #add to df to facilitate using the function below
test_df = drop_and_rename_columns(test_df)
test_positives = test_df[test_df['target'] == 1]
test_negs = test_df[test_df['target'] == 0].sample(n = int(len(test_positives)), random_state = 1)
test_df = test_positives.append(test_negs,ignore_index = True)
test_df = test_df.reindex(np.random.permutation(test_df.index))

out_path = '../data/her2/'
test_df.to_csv(out_path + 'her2_seven_vs_rest_test.csv', index=False)


train_df = drop_and_rename_columns(train_df)

#Class balance full training data set & shuffle dataframe
initial_train_pos = train_df[train_df['target'] == 1]
initial_train_neg = train_df[train_df['target'] == 0]

initial_train_neg = initial_train_neg[initial_train_neg['LD'] != 3] #drop the single LD 3 seq from df. required for sklearn train test stratifying later in script
initial_train_pos = initial_train_pos[initial_train_pos['LD'] != 2] #drop the two LD 2 seq from df. required for sklearn train test stratifying later in script

minlen = min([len(initial_train_pos),len(initial_train_neg) ])
initial_train_pos = initial_train_pos.sample(n = minlen, random_state = 1)
initial_train_neg = initial_train_neg.sample(n = minlen, random_state = 1)


train_df = pd.DataFrame()
train_df = initial_train_pos.append(initial_train_neg, ignore_index = True)
train_df = train_df.sample(n = int(len(train_df)), random_state = 1)

#Batch training & val sets with different quantities of class imbalance using positives as the minority class
class_imbalance_qty_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0] 


train_df_master_copy = train_df.copy()
for imbal_qty in class_imbalance_qty_list:
    #artificially increase class imbalance in training set by downsampling positives
    train_positives = train_df_master_copy[train_df_master_copy['target'] == 1]
    train_negs = train_df_master_copy[train_df_master_copy['target'] == 0]
    
    #new downsampling method using sklearn & edit distance
    if imbal_qty != 1.0:
        train_positives, x_discard, y_train, y_discard = train_test_split(train_positives, train_positives['target'], test_size = 1 - imbal_qty,
                                                                  random_state = 1, shuffle = True, stratify = train_positives['LD'])            
    elif imbal_qty == 1.0:
        train_truncated = train_positives    
    
    #split val set from training & maintain LD distribution per class
    train_positives, val_positives, y_train, y_val = train_test_split(train_positives, train_positives['target'], test_size = 1 - 0.8,
                                                              random_state = 1, shuffle = True, stratify = train_positives['LD'])
    
    train_negs, val_negs, y_train, y_val = train_test_split(train_negs, train_negs['target'], test_size = 1 - 0.8,
                                                          random_state = 1, shuffle = True, stratify = train_negs['LD'])

    train_df = train_positives.append(train_negs,ignore_index = True)
    train_df = train_df.reindex(np.random.permutation(train_df.index))

    val_df = val_positives.append(val_negs,ignore_index = True)
    val_df = val_df.reindex(np.random.permutation(val_df.index))
    

    train_df = drop_test_seqs(train_df, val_df, 'aaseq')
    train_df = train_df.drop(columns = ['df'])
    train_df = train_df.drop(columns = ['LD'])
    val_df = val_df.drop(columns = ['LD'])
    
    out_str_train = out_path + 'her2_seven_vs_rest_train_imbal_' +  str(imbal_qty) + '.csv'
    out_str_val = out_path + 'her2_seven_vs_rest_val_imbal_' +  str(imbal_qty) + '.csv'
    train_df.to_csv(out_str_train, index=False)
    val_df.to_csv(out_str_val, index=False)
