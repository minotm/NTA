#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: mminot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
import pickle
from matplotlib.backends.backend_pdf import PdfPages
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
    return data



def get_rho_by_variable(df,variable_str, metric_str):
    
        df = df[['best_' + metric_str,variable_str,'seed']].copy()
        s1,s2,s3, s4, s5 = df[df['seed'] == 1].copy(),  df[df['seed'] == 2].copy(), df[df['seed'] == 3].copy(), df[df['seed'] == 4].copy(), df[df['seed'] == 5].copy()
  
        variable_arr = df[variable_str].drop_duplicates().values
        variable_arr.sort()
        df = df.drop_duplicates([variable_str,'seed'],keep='first')
        df = df.sort_values([variable_str])
        
        
        for var in variable_arr:
            if var not in s1[variable_str].values:    
                df = df.append({'best_' + metric_str: np.nan, variable_str: var, 'seed': 1}, ignore_index=True)
            if var not in s2[variable_str].values:
                df = df.append({'best_' + metric_str: np.nan, variable_str: var, 'seed': 2}, ignore_index=True)
            if var not in s3[variable_str].values:
                df = df.append({'best_' + metric_str: np.nan, variable_str: var, 'seed': 3}, ignore_index=True)
            if var not in s4[variable_str].values:
                df = df.append({'best_' + metric_str: np.nan, variable_str: var, 'seed': 4}, ignore_index=True)
            if var not in s5[variable_str].values:
                df = df.append({'best_' + metric_str: np.nan, variable_str: var, 'seed': 5}, ignore_index=True)

        df = df.sort_values(by=[variable_str])
        test_rho = [list (x) for x in (zip(df[df['seed'] == 1]['best_' + metric_str],
                                          df[df['seed'] == 2]['best_' + metric_str],
                                          df[df['seed'] == 3]['best_' + metric_str],
                                          df[df['seed'] == 4]['best_' + metric_str],
                                          df[df['seed'] == 5]['best_' + metric_str],
                                          ))]
        return test_rho



data_type = 'her2'
path = f'../results/'
cnn = 'her2_cnn.csv'
transformer = 'her2_transformer.csv'

model_list = [cnn, transformer]
model_str_list = ['cnn', 'transformer']


for model, model_str in zip(model_list, model_str_list):
    data = pd.read_csv(path + model)    
    data['style_col'] = 'NT Augmented'
    #Parse Metric from DataFrame
    data = add_metric_to_df(data, is_float = False, metric_str = 'mcc')
    data = data.rename(columns = {'truncate_factor': 'training pos/neg ratio'})


    data_2 = data[data['aug_factor'] == '2']
    data_2_rename = data_2.copy()
    #rename dna aug fraction 1 to DNA 
    data_2_rename['aug_factor'] = '2'
    data_2_rename['style_col'] = '2'
    
    #convert aug_factor type from str to int
    data_5 = data[data['aug_factor'] == '5']
    data_5_rename = data_5.copy()
    #rename dna aug fraction 1 to DNA 
    data_5_rename['aug_factor'] = '5'
    data_5_rename['style_col'] = '5'
    
    data_10 = data[data['aug_factor'] == '10']
    data_10_rename = data_10.copy()
    #rename dna aug fraction 1 to DNA 
    data_10_rename['aug_factor'] = '10'
    data_10_rename['style_col'] = '10'
    
    data = data.append(data_2_rename, ignore_index = True)
    data = pd.merge(data,data_2, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    data = data.append(data_5_rename, ignore_index = True)
    data = pd.merge(data,data_5, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    data = data.append(data_10_rename, ignore_index = True)
    data = pd.merge(data,data_10, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)


    #store dna aug fraction 1 in separate df & drop from original dna df
    none_seq = data[data['aug_factor'] == 'none']
    
    data = pd.merge(data,none_seq, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    none_seq['aug_factor'] = 'DNA Baseline'
    none_seq['style_col'] = 'DNA Baseline'
    data = data.append(none_seq)
    
    aa_data = data[data['seq_type'] == 'aa']
    aa_data_rename = aa_data.copy()
    #rename dna aug fraction 1 to DNA 
    aa_data_rename['aug_factor'] = 'AA Baseline'
    aa_data_rename['style_col'] = 'AA Baseline'
    
    data = data.append(aa_data_rename, ignore_index = True)
    data = pd.merge(data,aa_data, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    for ngram in ['unigram', 'trigram_only', 'tri_unigram']: 
    
        plot_df = pd.DataFrame()
        plot_df = data[data['ngram'] == ngram]
        plot_df = plot_df.append(aa_data_rename,ignore_index=True)
        
        
        sns.set_theme(style="darkgrid")
        sns.set(rc={'figure.figsize':(7,5)})
        plt.figure()
        
        t10 = sns.color_palette('tab10')
        t20c = sns.color_palette('tab20c')
        t20b = sns.color_palette('tab20b')
        cb = sns.color_palette('colorblind')
        dna_b = cb[2]
        aa_b = '#E6A63C'
                
        palette_dict = {'AA Baseline': aa_b, 'DNA Baseline': dna_b,  '2': t20c[1], '5': t10[3], '10': t10[4]}
        marker_dict = {'AA Baseline': 'o', 'DNA Baseline': 'D','2': 'X', '5': 'X', '10': 'X'}
        dash_list = sns._core.unique_dashes(data['style_col'].unique().size+1)
        style = {key:value for key,value in zip(data['style_col'].unique(), dash_list[1:])}
        style['2'] = ''
        style['5'] = ''
        style['10'] = ''
        
        
        #legend code adapted from:
        #https://stackoverflow.com/questions/68591271/how-can-i-combine-hue-and-style-groups-in-a-seaborn-legend
        results = sns.lineplot(data=plot_df, x='training pos/neg ratio', y="best_mcc", hue='aug_factor',
                               #hue_order = [2,5,10,'DNA Baseline', 'AA Baseline'],palette=palette_dict, 
                               palette=palette_dict, style='style_col', dashes = style, markers = marker_dict)


        handles, labels = results.get_legend_handles_labels()
        index_item_title = labels.index('style_col')
        color_dict = {label: handle.get_color()
                      for handle, label in zip(handles[1:index_item_title], labels[1:index_item_title])}
        
        # loop through the items, assign color via the subscale of the item idem
        for handle, label in zip(handles[index_item_title + 1:], labels[index_item_title + 1:]):
            handle.set_color(color_dict[label])

        handles[-2].set_color(color_dict['DNA Baseline'])
        handles[-1].set_color(color_dict['AA Baseline'])
        
        handle_legend = []
        handle_legend.append(handles[1])
        handle_legend.append(handles[2])
        handle_legend.append(handles[3])
        handle_legend.append(handles[-2])
        handle_legend.append(handles[-1])
        
        results.legend(handles[index_item_title + 1:], labels[index_item_title + 1:], title='Augmentation Factor',
                  fontsize = 11, loc = 'lower right')

        results.set_ylabel('Test MCC Score')
        results.set_xlabel('Positive to Negative Balance Ratio')
        results.set(ylim=(0, 0.6))
        fig_name_str = f'{data_type}_{model_str}_{ngram}_mcc'
        plt.suptitle(fig_name_str, fontsize='large', x = 0.4)
        plt.savefig(f'{fig_name_str}.png', dpi=300)
