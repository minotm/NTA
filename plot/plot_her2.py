#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: mminot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

def add_metric_to_df(df, is_float, metric_str):
    data = df.copy()
    modified_metric = []
    
    if not is_float:
        for entry in data['output/final_test_' + metric_str]:
            m = re.search(r'[\-]*0\.[0-9]{4}',entry)
            try:
                modified_metric.append(float(m.group(0)))
            except:
                if entry.startswith('nan'):
                    modified_metric.append(np.nan)
                else:
                    modified_metric.append(np.nan)
        data['best_' + metric_str] = modified_metric
    else:
        data['best_' + metric_str] = data['output/final_test_' + metric_str]
        
    for entry in data['best_' + metric_str]:
        entry = float(entry)
    return data

path =  '../results/'
cnn = 'her2_seven_vs_rest_cnn.csv'
transformer = 'her2_seven_vs_rest_transformer.csv'

model_list = [cnn, transformer]
model_str_list = ['cnn', 'transformer']

full_data = pd.DataFrame()
for model, model_str in zip(model_list, model_str_list):
    tmp_data = pd.read_csv(path + model)    
    full_data = full_data.append(tmp_data)
full_data = full_data.rename(columns = {'basemodel': 'model'})
full_data.loc[full_data['ngram'] == 'trigram_only', 'ngram'] = 'trigram'
full_data = full_data.sort_values(by=['ngram','model','aug_factor'],ascending=False)

aug_type_list = list(full_data['aug_type'].unique())

aug_factor_dict = {'1': '2', '2': '5', '3': '10', '4':'25'} #rename augmentation datasets to their true n_aug values

for model in model_str_list:
    data = full_data.copy()
    data = data[data['model'] == model]
    data = data[data['seq_type'] != 'aa']
    data['style_col'] = 'NT Augmented'
    #Parse Metric from DataFrame
    data['aug_factor'] = data['aug_factor'].replace(aug_factor_dict)
    data = add_metric_to_df(data, is_float = False, metric_str = 'mcc')

    none_seq = data[(data['aug_factor'] == 'none') & (data['seq_type'] == 'dna')]
    data = pd.merge(data,none_seq, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    none_seq['aug_factor'] = none_seq['aug_factor'].replace({'none': 'DNA Baseline'})

    indexer = none_seq[none_seq.aug_factor == 'none'].index
    none_seq.loc[indexer, 'aug_factor'] = 'DNA Baseline'
    none_seq.loc[indexer, 'style_col'] = 'DNA Baseline'
    
    none_seq['style_col'] = 'DNA Baseline'
    data = data.append(none_seq)
    
    aug_type_list_out = ['iterative', 'codon_balance', 'codon_shuffle', 'random']
    
    for aug_type in aug_type_list_out:
        if aug_type.startswith('Online'): on_off = 'on'
        else: on_off = 'off'
        aa_data = full_data[(full_data['seq_type'] == 'aa') & (full_data['on_off'] == on_off)].copy()
        
        aa_data['aug_factor'] = 'AA Baseline'
        aa_data = add_metric_to_df(aa_data, is_float = False, metric_str = 'mcc')
        aa_data_rename = aa_data.copy()
        aa_data_rename['aug_type'] = aug_type
        aa_data_rename['style_col'] = 'AA Baseline'
    
        aa_trigram = aa_data_rename.copy()
        aa_trigram['ngram'] = 'trigram'
        aa_tri_unigram = aa_data_rename.copy()
        aa_tri_unigram['ngram'] = 'tri_unigram'
        
        data = data.append(aa_trigram, ignore_index = True)
        data = data.append(aa_tri_unigram, ignore_index = True)
        data = data.append(aa_data_rename, ignore_index = True)

    data = data[data['aug_factor'] != '25']
    
    data['aug_type'] = data['aug_type'].replace({'codon_shuffle': 'Codon Shuffle', 'iterative': 'Iterative',
                                                 'random': 'Random', 'codon_balance': 'Codon Balance'})
    
    data['ngram'] = data['ngram'].replace({'tri_unigram': 'tri+unigram'})
    
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(7,5)})
    plt.figure()
    
    cb = sns.color_palette('tab10')
    aug_color = '#6CA2E6'
    dna_b = cb[2]
    aa_b = '#E6A63C'
    
    palette_dict = {'AA Baseline': aa_b, 'DNA Baseline': dna_b, 
                    '2': cb[0], '5': cb[1], '10': cb[3], '50': cb[4], '100': cb[5],
                    'online_partial': cb[9], 'Online Codon Balance': cb[7], 'Online': cb[4], 
                    'Online No Aug. Baseline': dna_b, 'Online Codon Shuffle': cb[0]}
    
    marker_dict = {'cnn': 'o', 'transformer': 'o'}
    dashes_dict = {'cnn': '-', 'transformer': '-'}
    results = sns.relplot(data= data, kind = 'line', x="truncate_factor", y="best_mcc",hue='aug_factor', 
                          col = "ngram", row = "aug_type", marker = "o", palette=palette_dict, legend = 'full', 
                          style = 'style_col', row_order = ['Iterative', 'Random', 'Codon Balance', 'Codon Shuffle'])
    
    (results.set_axis_labels("Fraction Total Data", "Test MCC")
      .set_titles("{row_name}: {col_name}")
      .tight_layout(w_pad=0))
    
    results.legend.remove()
    results.fig.legend(handles=results.legend.legendHandles[1:6], loc=7, frameon= False, title = 'Augmentation Factor \n($n_{aug}$)')
    
    plt.subplots_adjust(wspace=0.1)
    data_type = 'her2'
    fig_name_str = f'{data_type}_{model}'
    plt.suptitle(fig_name_str, fontsize='large', x = 0.5, y = 1.025)
    plt.savefig(f'{fig_name_str}.pdf')