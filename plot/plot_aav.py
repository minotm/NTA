#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created 2022

@author: mminot
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_metric_to_df(df, is_float, metric_str):
    data = df.copy()
    out_list = []
    for entry in data['output/final_test_' + metric_str]:
        out_list.append(float(entry))
    data['best_' + metric_str] = out_list
    
    return data

path = f'../results/'
cnn = 'aav_seven_vs_rest_cnn2.csv'
transformer = 'aav_seven_vs_rest_t2.csv'

model_list = [cnn, transformer]
model_str_list = ['cnn2', 't2']

full_data = pd.DataFrame()
for model, model_str in zip(model_list, model_str_list):
    tmp_data = pd.read_csv(path + model)    
    full_data = full_data.append(tmp_data)
full_data = full_data.rename(columns = {'basemodel': 'model'})
full_data.loc[full_data['ngram'] == 'trigram_only', 'ngram'] = 'trigram'
full_data = full_data.sort_values(by=['ngram','model','aug_factor'],ascending=False)

aug_type_list = list(full_data['aug_type'].unique())
aug_factor_dict = {'1': '2', '2': '5', '3': '10', '4':'25', '5': '50'} #raname augmentation datasets to their true n_aug values


aug_type_list = aug_type_list[1: ]
for model in model_str_list:
    
    data = full_data.copy()
    data = data[data['model'] == model]
    data = data[data['seq_type'] != 'aa']
    data['aug_factor'] = data['aug_factor'].replace(aug_factor_dict)

    data['style_col'] = 'NT Augmented'
    #Parse Metric from DataFrame
    data = add_metric_to_df(data, is_float = True, metric_str = 'spearman')
    data.astype({'best_spearman': float})
    
    none_seq = data[(data['aug_factor'] == 'none') & (data['seq_type'] == 'dna')]
    data = pd.merge(data,none_seq, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    none_seq['aug_factor'] = none_seq['aug_factor'].replace({'none': 'DNA Baseline'})
    
    indexer = none_seq[none_seq.aug_type  == 'online'].index
    none_seq.loc[indexer, 'aug_factor'] = 'Online'
    none_seq.loc[indexer, 'style_col'] = 'NT Augmented'
    
    indexer = none_seq[none_seq.aug_type  == 'online_balance'].index
    none_seq.loc[indexer, 'aug_factor'] = 'Online Codon Balance'
    none_seq.loc[indexer, 'aug_type'] = 'Online'
    none_seq.loc[indexer, 'style_col'] = 'NT Augmented'
    
    indexer = none_seq[none_seq.aug_type  == 'online_none'].index
    none_seq.loc[indexer, 'aug_factor'] = 'Online No Aug. Baseline'
    none_seq.loc[indexer, 'aug_type'] = 'Online'
    none_seq.loc[indexer, 'style_col'] = 'NT Augmented'
    
    indexer = none_seq[none_seq.aug_type  == 'online_shuffle'].index
    none_seq.loc[indexer, 'aug_factor'] = 'Online Codon Shuffle'
    none_seq.loc[indexer, 'aug_type'] = 'Online'
    none_seq.loc[indexer, 'style_col'] = 'NT Augmented'
    
    indexer = none_seq[none_seq.aug_factor == 'none'].index
    none_seq.loc[indexer, 'aug_factor'] = 'DNA Baseline'
    none_seq.loc[indexer, 'style_col'] = 'DNA Baseline'
    
    
    none_seq['style_col'] = 'DNA Baseline'
    data = data.append(none_seq)

    aug_type_list_out = ['iterative', 'codon_balance', 'codon_shuffle', 'Online', 'random']
    
    for aug_type in aug_type_list_out:
        
        if aug_type.startswith('Online'): on_off = 'on'
        else: on_off = 'off'
        aa_data = full_data[(full_data['seq_type'] == 'aa') & (full_data['on_off'] == on_off)].copy()
        
        aa_data['aug_factor'] = 'AA Baseline'
        aa_data = add_metric_to_df(aa_data, is_float = True, metric_str = 'spearman')
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
    
    data['aug_type'] = data['aug_type'].replace({'codon_shuffle': 'Codon Shuffle', 'iterative': 'Iterative',
                                                 'random': 'Random', 'codon_balance': 'Codon Balance', 'online': 'Online'})
    
    data['ngram'] = data['ngram'].replace({'tri_unigram': 'tri+unigram'})
    
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(7,5)})
    plt.figure()
    
    cb = sns.color_palette('tab10')
    aug_color = '#6CA2E6'
    dna_b = cb[2]
    aa_b = '#E6A63C'
    
    palette_dict = {'AA Baseline': aa_b, 'DNA Baseline': dna_b, 
                    '2': cb[0], '5': cb[1], '10': cb[3], '25': cb[4], '50': cb[5],
                    'online_partial': cb[9], 'Online Codon Balance': cb[7], 'Online': cb[4], 
                    'Online No Aug. Baseline': dna_b, 'Online Codon Shuffle': cb[0]}
    
    
    data = data[(data['aug_factor'] != '50')]
    
    
    results = sns.relplot(data= data, kind = 'line', x="truncate_factor", y="best_spearman",hue='aug_factor', 
                          col = "ngram", row = "aug_type", marker = "o", palette=palette_dict, legend = 'full', 
                          style = 'style_col', row_order = ['Online', 'Iterative', 'Random', 'Codon Balance', 'Codon Shuffle'])
    
    (results.set_axis_labels("Fraction Total Data", "Test Rho")
      .set_titles("{row_name}: {col_name}")
      .tight_layout(w_pad=0))
    
    results.legend.remove()
    results.fig.legend(handles=results.legend.legendHandles[1:10], loc=7, frameon= False, title = "Augmentation Factor ($n_{aug}$)")
    
    results.set(xscale = 'log')
    plt.subplots_adjust(wspace=0.1)
    data_type = 'aav'
    fig_name_str = f'{data_type}_{model}'
    plt.suptitle(fig_name_str, fontsize='large', x = 0.5, y = 1.025)
    plt.savefig(f'{fig_name_str}.pdf')