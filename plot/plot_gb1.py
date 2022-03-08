#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
    
def add_rho_to_df(df, is_float):
    data = df.copy()
    modified_rho = []
    
    if not is_float:
        for entry in data['output/final_test_spearman']:
            m = re.search(r'^[\-]*0\.[0-9]{4}',entry)
            try:
                modified_rho.append(float(m.group(0)))
            except:
                if entry.startswith('nan'):
                    modified_rho.append(np.nan)
                else:
                    modified_rho.append(np.nan)
        data['best_rho'] = modified_rho    
    else:
        data['best_rho'] = data['output/final_test_spearman']
    return data



def get_rho_by_variable(df,variable_str):
    
        df = df[['best_rho',variable_str,'seed']].copy()
        s1,s2,s3, s4, s5 = df[df['seed'] == 1].copy(),  df[df['seed'] == 2].copy(), df[df['seed'] == 3].copy(), df[df['seed'] == 4].copy(), df[df['seed'] == 5].copy()
  
        variable_arr = df[variable_str].drop_duplicates().values
        variable_arr.sort()
        df = df.drop_duplicates([variable_str,'seed'],keep='first')
        df = df.sort_values([variable_str])
        
        
        for var in variable_arr:
            if var not in s1[variable_str].values:    
                df = df.append({'best_rho': np.nan, variable_str: var, 'seed': 1}, ignore_index=True)
            if var not in s2[variable_str].values:
                df = df.append({'best_rho': np.nan, variable_str: var, 'seed': 2}, ignore_index=True)
            if var not in s3[variable_str].values:
                df = df.append({'best_rho': np.nan, variable_str: var, 'seed': 3}, ignore_index=True)
            if var not in s4[variable_str].values:
                df = df.append({'best_rho': np.nan, variable_str: var, 'seed': 4}, ignore_index=True)
            if var not in s5[variable_str].values:
                df = df.append({'best_rho': np.nan, variable_str: var, 'seed': 5}, ignore_index=True)

        df = df.sort_values(by=[variable_str])
        test_rho = [list (x) for x in (zip(df[df['seed'] == 1]['best_rho'],
                                          df[df['seed'] == 2]['best_rho'], 
                                          df[df['seed'] == 3]['best_rho'],
                                          df[df['seed'] == 4]['best_rho'],
                                          df[df['seed'] == 5]['best_rho']
                                          ))]
        return test_rho



data_type = 'GB1'
plot_dict = {}

path = f'../results/'
cnn = 'gb1_three_vs_rest_cnn.csv'
transformer = 'gb1_three_vs_rest_transformer.csv'

model_list = [cnn, transformer]
model_str_list = ['cnn', 'transformer']
baseline_list = ['AA Baseline', 'DNA Baseline']


for model, model_str in zip(model_list, model_str_list):
    data = pd.DataFrame()
    data = pd.read_csv(path + model)
    data = add_rho_to_df(data, is_float = True)
    
    #store dna aug fraction none in separate df & drop from original dna df
    dna_data_none= data[data['aug_factor'] == 'none']
    #rename dna aug fraction 1 to DNA 
    dna_data_rename = dna_data_none.copy()
    dna_data_rename['aug_factor'] = 'DNA Baseline'
    data = data.append(dna_data_rename, ignore_index = True)
    
    data = data.append(dna_data_rename, ignore_index = True)
    data = pd.merge(data,dna_data_none, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    #store dna aug fraction 1 in separate df & drop from original dna df
    aa_data = data[data['seq_type'] == 'aa']
    aa_data_rename = aa_data.copy()
    #rename dna aug fraction 1 to DNA 
    aa_data_rename['aug_factor'] = 'AA Baseline'
    
    data = data.append(aa_data_rename, ignore_index = True)
    data = pd.merge(data,aa_data, how='outer', indicator=True)
    data = data[data._merge.ne('both')].drop('_merge',1)
    
    
    aa_baseline = data[(data['ngram'] == 'unigram') & (data['seq_type'] == 'aa')]
    
        
    for ngram in ['unigram', 'trigram_only', 'tri_unigram']: 
        
        nta_df = pd.DataFrame()
        data2 = data.copy()
        nta_df = data2[(data2['ngram'] == ngram) & (data2['seq_type'] == 'dna')]
        
        
        #create mask & drop all baeline entries
        dna_baseline = nta_df[(nta_df['aug_factor'] == 'DNA Baseline')]
        rgx = r'Baseline'
        mask = nta_df['aug_factor'].str.contains(rgx, na=False, flags=re.IGNORECASE, regex=True, case=False)            
        nta_df = nta_df[~mask]
        

        def entries_with_max_mean_to_df(input_df):
            df = input_df.copy()
            pre_df = df.groupby(['truncate_factor', 'aug_factor']).agg([np.mean])
            pre_df = pre_df.pivot_table(index=['aug_factor'], columns='truncate_factor', values='best_rho')

            max_dict  = {}
            truncate_factors = df['truncate_factor'].unique()
            for trunc_factor in truncate_factors:
                max_dict[trunc_factor] = pre_df['mean'][trunc_factor].idxmax()
            
            out_df = pd.DataFrame()
            for trunc_factor in truncate_factors:
                out_df = out_df.append(df[(df['truncate_factor'] == trunc_factor) 
                                      & (df['aug_factor'] == str(max_dict[trunc_factor]) ) ])            
            return out_df
        
        nta_df = entries_with_max_mean_to_df(nta_df)
        aa_base_df = entries_with_max_mean_to_df(aa_baseline)
        dna_base_df = entries_with_max_mean_to_df(dna_baseline)
        
        nta_df['data_type'] = 'NT Augmented'
        aa_base_df['data_type'] = 'AA Baseline'
        dna_base_df['data_type'] = 'DNA Baseline'
        
        
        plot_df = nta_df.append(aa_base_df,ignore_index=True)
        plot_df = plot_df.append(dna_base_df,ignore_index=True)
        
        
        sns.set(rc={'figure.figsize':(7,5)})
        plt.figure()
        
        cb = sns.color_palette('colorblind')
        aug_color = '#6CA2E6'
        dna_b = cb[2]
        aa_b = '#E6A63C'
                
        palette_dict = {'AA Baseline': aa_b, 'DNA Baseline': dna_b, 'NT Augmented': aug_color}
        marker_dict = {'AA Baseline': 'o', 'DNA Baseline': 'D', 'NT Augmented': 'X'}
        results = sns.lineplot(data=plot_df, x='truncate_factor', y="best_rho", hue='data_type', 
                               palette=palette_dict, style = 'data_type', markers = marker_dict)
            
        results.set(xscale='log')
        results.set_ylabel('Test Rho')
        results.set_xlabel('% Total Data')
        results.set(ylim=(0, 0.9))
        sns.despine(ax = results)
        legend = plt.legend(title='Data Type', loc='lower right', prop={'size': 11})
        fig_name_str = f'{data_type}_{model_str}_{ngram}'
        plt.suptitle(fig_name_str, fontsize='large', x = 0.4)
        plt.savefig(f'{fig_name_str}.png', dpi=300)