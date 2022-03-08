#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import torch
import torch.nn.parallel
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import datetime
import argparse
from utils import *
from train_routines  import *
import numpy as np
import re
import os



def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', default='1', type=int,
                        help='number of class labels in data')
    parser.add_argument('--truncate_factor', default='0.5', type=float,
                        help='training data truncate factor (float) between 0 and 1')
    parser.add_argument('--base_model', default='cnn', type=str,
                        help='base model to use, cnn or transformer')
    parser.add_argument('--data_type', default='gb1', type=str,
                        help='gb1, aav, her2')
    parser.add_argument('--learn_rate', default=5e-4, type=float,
                       help='initial optimizer learning rate. only used if learn rate scheduler turned off')
    parser.add_argument('--lr_scheduler', default=False, type=bool,
                       help='include learn rate scheduler')
    parser.add_argument('--dropout', default=0.3, type=float,
                       help='dropout fraction for model base_d')
    parser.add_argument('--conv_filters', default=64, type=int,
                        help='number convolutional filters')
    parser.add_argument('--opt_id', default='sgd', type=str,
                        help='options sgd, adam')
    parser.add_argument('--ngram', default='unigram', type=str,
                        help='unigram, trigram_only, trigram')
    parser.add_argument('--seq_type', default='aa', type=str,
                        help='aa,dna')
    parser.add_argument('--data_set', default='three_vs_rest', type=str,
                       help='options: three_vs_rest if using gb1, seven_vs_rest if using aav, irrelevant if using her2')
    parser.add_argument('--aug_factor', default='1', type=str,
                       help='data augmentation factor')
    parser.add_argument('--kernel', default=5, type=int,
                       help='size of conv1d kernel')
    parser.add_argument('--big_data', default='False', type=str,
                       help='runs model for only a single truncated data set. purpose: to optimize GPU use on cluster. options: True, False')
    parser.add_argument('--trunc', default=0.05, type=float,
                       help='True, False')
    parser.add_argument('--seq_file_type', default='aa', type=str,
                       help='aa, dna')
    parser.set_defaults(augment=True)
    
    return parser


def main(args):
    
    if torch.cuda.is_available():
         use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    #=============== Preset Experimental Parameters ========== =============================    
    #args.data_type = 'aav'
    #args.learn_rate = 5e-6
    #args.lr_scheduler = False
    #args.opt_id = 'adam'
    #args.big_data = 'True'
    #args.aug_factor = 'neg_ds_explicit'
    #args.aug_factor = 2
    #args.aug_factor = 'none'
    #args.seq_file_type = 'dna'
    #args.seq_type = 'dna'
    #args.trunc= 0.5
    #args.full_dataset = 'True'
    #args.base_model = 'transformer'
    #args.ngram = 'trigram_only'
    #args.ngram = 'tri_unigram'
    #args.dropout = 0.3
    
        
    if args.base_model == 'transformer': args.batch_size = 32
    elif args.base_model == 'cnn': args.batch_size = 256
    
    if args.seq_type == 'aa': args.ngram = 'unigram'
    if args.data_type == 'gb1': args.data_set = 'three_vs_rest'
    elif args.data_type == 'aav': args.data_set = 'seven_vs_rest'
    
    model_outpath = 'saved_models/'
    
    
    data_type = args.data_type
    base_model_list = [args.base_model]
    seed_list = [1,2,3,4,5]
    
    for seed_entry in seed_list:
        torch.manual_seed(seed_entry)
        torch.manual_seed(seed_entry)
        torch.cuda.manual_seed(seed_entry)
        np.random.seed(seed_entry)
        
    
        print(f'SEED {seed_entry}')
        
        
        #====================          Instantiate Model & Optimizer                    ======================
        for base_model in base_model_list:
        
            
            params = {'data': args.data_type, 'model_name': base_model, 'lr': args.learn_rate, 'lr_scheduler': args.lr_scheduler,
                      'p_dropout': args.dropout, 'ngram': args.ngram, 'seq_type': args.seq_type,
                      'conv_filters': args.conv_filters, 'opt': args.opt_id, 'kernel': args.kernel, 'batch_size': args.batch_size}
            
            l_r = args.learn_rate #for tensorboard writer
            model_dict = {'basic': BaseModel(params,device)}
    
            batch_size = args.batch_size
            #====================            Data Fractionation Params                        ====================
            
            if args.data_type == 'gb1':
                truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                
            if args.data_type == 'aav':
                truncate_factor_list = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                
            #NOTE: when using her2 data, 'truncate factor' refers to train imbalance ratio (minority class examples / majority class examples)
            if args.data_type == 'her2':
                truncate_factor_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]          
    
            #if the final data set size is >= 1e5, limit experiment to 5 seeds for single provided truncate factor
                #rationale: limit GPU runtime on cluster
            if args.big_data == 'True':
                truncate_factor_list = [args.trunc]
            
            
            #====================            Restrict Train Size  & Load Data===========================
            for trunc_factor in truncate_factor_list:
                print(f'Batching {args.data_type} Data')
                print(f'Batching Task {args.data_set}')
                
                gb1_path = 'data/gb1/'
                aav_path = 'data/aav/'
                her2_path = 'data/her2/'
                
                if args.data_type == 'gb1':  data_path = gb1_path                
                elif args.data_type == 'aav' :  data_path = aav_path                                
                elif args.data_type == 'her2': data_path = her2_path                
                
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
                
                if args.seq_file_type == 'aa':
                    print('Using AA Data Files')
                    
                    if args.data_type == 'aav':
                        train_df = pd.read_csv(data_path + f'aav_{args.data_set}_train_truncated_{str(trunc_factor)}.csv')
                        val_df = pd.read_csv(data_path + f'aav_{args.data_set}_val_truncated_{str(trunc_factor)}.csv')
                        test_df = pd.read_csv(data_path + 'aav_seven_vs_rest_test.csv')
    
    
                    elif args.data_type == 'gb1':
                        train_df = pd.read_csv(data_path + f'gb1_{args.data_set}_train_truncated_{str(trunc_factor)}.csv')
                        val_df = pd.read_csv(data_path + f'gb1_{args.data_set}_val_truncated_{str(trunc_factor)}.csv')
                        data = pd.read_csv(data_path + args.data_set + '_truncated_seqs.csv')
                        test_df = data[data['set'] == 'test']
                
                    elif args.data_type == 'her2':
                        train_df = pd.read_csv(data_path + f'her2_train_imbal_{str(trunc_factor)}.csv')
                        val_df = pd.read_csv(data_path + f'her2_val_imbal_{str(trunc_factor)}.csv')
                        test_df = pd.read_csv(data_path + 'her2_test.csv')
                    
                    train_df = drop_val_seqs(train_df, val_df, seq_str = 'aaseq')                
                    train_df = train_df.reindex(np.random.permutation(train_df.index))
                    
                    def x_y_split_aa(df):
                        x = df['aaseq']
                        y = df['target']
                        return x, y
                    x_train, y_train = x_y_split_aa(train_df.copy())
                    x_val, y_val = x_y_split_aa(val_df.copy())       
                    x_test, y_test = x_y_split_aa(test_df.copy())       
                    
                    
                elif args.seq_file_type =='dna':
                    print('Using DNA Data')
                    
                    if args.data_type == 'aav':
                        train_df = pd.read_csv(data_path + f'aav_{args.data_set}_train_truncated_{str(trunc_factor)}_dna_aug_{args.aug_factor}.csv')
                        val_df = pd.read_csv(data_path + f'aav_{args.data_set}_val_truncated_{str(trunc_factor)}_dna_aug_{args.aug_factor}.csv')
                        test_df = pd.read_csv(data_path + f'aav_{args.data_set}_test_dna.csv')
    
                    
                    elif args.data_type == 'gb1':
                        train_df = pd.read_csv(data_path + f'gb1_{args.data_set}_train_truncated_{str(trunc_factor)}_dna_aug_{args.aug_factor}.csv')
                        val_df = pd.read_csv(data_path + f'gb1_{args.data_set}_val_truncated_{str(trunc_factor)}_dna_aug_{args.aug_factor}.csv')
                        test_df = pd.read_csv(data_path + f'gb1_{args.data_set}_test_dna.csv')
    
                    elif args.data_type == 'her2':
                        train_df = pd.read_csv(data_path + f'her2_train_imbal_{str(trunc_factor)}_aug_{str(args.aug_factor)}_dna.csv')
                        val_df = pd.read_csv(data_path + f'her2_val_imbal_{str(trunc_factor)}_aug_{str(args.aug_factor)}_dna.csv')
                        test_df = pd.read_csv(data_path + 'her2_test_dna.csv')
                    
                    train_df = train_df.reindex(np.random.permutation(train_df.index))
                    train_df = drop_val_seqs(train_df, val_df, seq_str = 'dnaseq')
    
                    def x_y_split_dna(df):
                            x = df['dnaseq']
                            y = df['target']
                            return x, y
                        
                    x_train, y_train = x_y_split_dna(train_df.copy())
                    x_val, y_val = x_y_split_dna(val_df.copy())    
                    x_test, y_test = x_y_split_dna(test_df.copy())       
    
                #====================            Input Encoding            ============================
                if args.data_type == 'gb1' and  args.seq_type  == 'dna': seq_len = 12
                elif args.data_type == 'aav' and  args.seq_type  == 'dna': seq_len = 126
                elif args.data_type == 'her2' and  args.seq_type  == 'dna': seq_len = 30
                else: seq_len = None
               
                
    
                model_list = [model_dict['basic']]
                
            
                if args.seq_type == 'aa' and args.ngram == 'unigram' and args.data_type != 'thermo':
                    vocabulary = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
                elif args.seq_type == 'dna' and args.ngram == 'unigram':
                    vocabulary = ['A', 'C', 'G', 'T']
                elif args.seq_type == 'dna' and args.ngram == 'tri_unigram':
                    vocabulary = pd.read_csv('data/ngram_vocabularies/nt_vocabulary.csv')['gram']        
                elif args.seq_type == 'dna' and args.ngram == 'trigram_only':
                    vocabulary = pd.read_csv('data/ngram_vocabularies/nt_trigram_vocabulary.csv')['gram']
                
                word_to_ix = {word: i for i, word in enumerate(vocabulary)}
                
                x_train, x_val, x_test,vocabulary = encode_ngrams(x_train, x_val, x_test, args, seq_len)
                if args.data_type == 'her2':# and args.aug_factor != 'none':
                    class_sample_count = np.array( [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
                    weight = 1. / class_sample_count
                    weights_for_sampler = np.array([weight[t] for t in y_train])
                    weights_for_sampler = torch.from_numpy(weights_for_sampler)
                    weights_for_sampler = weights_for_sampler.double()
                    
                else: weights_for_sampler = None
                
                
                train_loader, val_loader, test_loader = data_to_loader_online_nta(x_train, x_val, x_test, y_train, y_val, y_test, batch_size = args.batch_size, 
                                                                            args = args, alphabet = vocabulary, word_to_idx_conversion = word_to_ix, sampler_weights = weights_for_sampler)
            
                train_length = len(y_train)
                
                print(f'\n Seq Type: {args.seq_type}')
                print(f'ngram: {args.ngram}')
                print(f'Trunc Factor: {trunc_factor}')
                print(f'\nNumber Training Samples: {len(y_train)}')
                print(f'Number Validation Samples: {len(y_val)}')
                
                #====================           Initialize Model                              ======================
                for i in range(len(model_list)):
                    
                    model = model_list[i].to(device)
                    print(f'\nNow Training with base_model {base_model}')
                    print(f'learn rate = {args.learn_rate}')
                    print(f'learn rate scheduler = {args.lr_scheduler}')
                    print(f'optimizer = {args.opt_id}')
                    
                    #====================== Tensorboard Initialization ===================================
                    current_date = str(datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
                    run_name = f'runs/{args.data_type}_{args.seq_type}_ngram_{args.ngram}_model_{args.base_model}_aug_{args.aug_factor}_trunc_{trunc_factor}_{args.data_set}_seed_{seed_entry}_{current_date}'
                    writer = SummaryWriter(run_name)
                    writer.flush()
                                        
                    hparams = {'data_type': data_type, 'seq_type': args.seq_type,'basemodel': base_model, 
                               'truncate_factor': trunc_factor, 'train_len': train_length, 
                               'seed': seed_entry, 'optimizer': args.opt_id, 'learn_rate': args.learn_rate, 'ngram': args.ngram,
                               'data_set': args.data_set, 'dropout': args.dropout, 'aug_factor': args.aug_factor}
                    
                    if args.base_model == 'transformer': args.epochs =  250
                    if args.base_model == 'cnn': args.epochs = 250
                    patience = 25
                    patience_counter = 0
                    if args.data_type == 'aav' or args.data_type == 'gb1':                                    
                        best_spearman, best_spearman_epoch, best_mse, best_mse_epoch = -10,0,100,0
                        
                        #======================     Train & Eval Cycle          ===================================
                        bestmodel_name = f'{args.data_type}_{args.seq_type}_{base_model}_{args.ngram}_aug_{args.aug_factor}_trunc_{trunc_factor}_{args.data_set}_seed_{seed_entry}_{current_date}'
                        for epoch in range(args.epochs):
                            
                            model.train_step(train_loader,epoch,writer,args.batch_size)
                            val_spearman, val_mse,  = model.test_step(val_loader, epoch, writer)
                            #record best performance
                            if val_mse < best_mse:
                                best_mse = val_mse
                                best_mse_epoch = epoch
                            if val_spearman > best_spearman:
                                best_spearman = val_spearman
                                best_spearman_epoch = epoch
                                patience_counter = 0
                                torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': model.optimizer.state_dict()}, 
                                           model_outpath + bestmodel_name)
                            else:
                                patience_counter += 1
                            print(f'best Spearman = {best_spearman}')                        
                            print(f'patience_counter = {patience_counter}')
                            if epoch >= 40 and patience_counter >= patience:
                                print('Met Early Stopping Patience')
                                break
                            
                        #Run Model Evaluation After Training Is Complete
                        #If best Rho does not beat initial Rho (i.e. low data sets), then use most recent model for test eval
                        try:
                            sd = torch.load(model_outpath + bestmodel_name)
                            model.load_state_dict(sd['model_state_dict'])   
                        except:
                            pass 
                        
                        test_spearman, test_mse,  = model.test_step(test_loader, epoch, writer)
                        metric_dict = {'output/best_val_spearman': best_spearman, 'output/best_val_mse': best_mse, 
                                       'output/best_spearman_epoch': best_spearman_epoch, 'output/best_mse_epoch': best_mse_epoch,
                                       'output/final_test_spearman': test_spearman, 'output/final_test_mse': test_mse,
                                       'final_epoch': epoch}
                        
                        for key, value in metric_dict.items():
                            if key == 'output/best_val_spearman' or key == 'output/best_val_mse' or key == 'output/final_test_spearman' or key == 'output/final_test_mse':
                                m = re.search(r'\((.*)\)', str(value))                                
                                try:
                                    metric_dict[key] = m.group(1)
                                except:
                                    pass
                        output_dict = {**hparams, **metric_dict}                        
                        output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
                        output_dict['time'] = output_time
                        filename = f'results/{data_type}_{args.data_set}_{base_model}.csv'
    
                        df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
                        print(f'SpearmanR on Test Set = {test_spearman}')
                        print(f'MSE on Test Set = {test_mse}')                
                        print('Now Deleting Model')
                        del model
                        try:
                            os.remove(model_outpath + bestmodel_name)
                        except:
                            pass
                        #reinstantiate model option
                        model_dict = {'basic': BaseModel(params,device)}
        
                    elif args.data_type == 'her2':
                        if args.base_model == 'transformer': args.epochs =  400
                        if args.base_model == 'cnn': args.epochs = 400
                        patience = 40
                        patience_counter = 0    
    
                        best_f1, best_f1_epoch, best_auroc, best_auroc_epoch, best_loss, best_loss_epoch, best_precision, best_recall, best_precision_epoch, best_recall_epoch = 0,0,0,0,100,0,0,0,0,0
                        test_auroc, test_recall, test_precision, test_mcc = 0,0,0, -10
                        val_mcc, best_mcc = 0,-10
                        
                        #======================     Train & Eval Cycle          ===================================
                        bestmodel_name = f'{args.data_type}_{args.seq_type}_{base_model}_{args.ngram}_aug_{args.aug_factor}_trunc_{trunc_factor}_{args.data_set}_seed_{seed_entry}_{current_date}'
                        for epoch in range(args.epochs):
                            
                            model.train_step_classifier(train_loader,epoch,writer,args.batch_size)
                            val_f1, val_loss, val_precision, val_recall, val_mcc = model.test_step_classifier(val_loader, epoch, writer)                        
                            
                            #record best performance
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_loss_epoch = epoch
                            if val_precision > best_precision:
                                best_precision = val_precision
                                best_precision_epoch = epoch
                            if val_recall > best_recall:
                                best_recall = val_recall
                                best_recall_epoch = epoch
                            if val_f1 > best_f1:
                                best_f1 = val_f1
                                best_f1_epoch = epoch
                            if val_mcc > best_mcc:
                                best_mcc = val_mcc
                                best_mcc_epoch = epoch
                                patience_counter = 0
                                
                                torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': model.optimizer.state_dict()}, 
                                           model_outpath + bestmodel_name)
                            else:
                                patience_counter += 1                        
                            print(f'best MCC = {best_mcc}')                        
                            print(f'patience_counter = {patience_counter}')
                            if epoch >= 40 and patience_counter >= patience:
                                print('Met Early Stopping Patience')
                                break
    
                        #Run Model Evaluation After Training Is Complete
                        #If best MCC does not beat initial MCC (i.e. low data sets & nan values), then use most recent model for test eval
                        try:
                            sd = torch.load(model_outpath + bestmodel_name)
                            model.load_state_dict(sd['model_state_dict'])
                        except:
                            pass
                        
                        test_f1, test_loss, test_precision, test_recall,  test_mcc = model.test_step_classifier(test_loader, epoch, writer)
    
                        metric_dict = {'output/best_val_f1': best_f1, 'output/best_val_loss': best_loss, 
                                       'output/best_f1_epoch': best_f1_epoch, 'output/best_loss_epoch': best_loss_epoch,
                                       'output/final_test_f1': test_f1, 'output/final_test_auroc': test_auroc, 
                                       'output/final_test_loss': test_loss,'final_epoch': epoch,
                                       'output/best_val_auroc': best_auroc, 
                                       'output/best_val_auroc_epoch': best_auroc_epoch, 
                                       'output/best_val_precision': best_precision,
                                       'output/best_val_precision_epoch': best_precision_epoch,
                                       'output/best_val_recall': best_recall,
                                       'output/best_val_recall_epoch': best_recall_epoch,
                                       'output/final_test_precision': test_precision,
                                       'output/final_test_recall': test_recall,
                                       'output/final_test_mcc': test_mcc}
        
                        for key, value in metric_dict.items():
                            if key == 'output/best_val_f1' or key == 'output/best_val_loss' or key == 'output/final_test_f1' or key == 'output/final_test_loss':
                                m = re.search(r'\((.*)\)', str(value))                                
                                try:
                                    metric_dict[key] = m.group(1)
                                except:
                                    pass
                                
                        output_dict = {**hparams, **metric_dict}
                        output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
                        output_dict['time'] = output_time                    
                        filename = f'results/{data_type}_{base_model}.csv'
                        df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))    
                        print(f'Test F1 = {test_f1}')
                        print(f'Test MCC = {test_mcc}')
                        print(f'Test Precision = {test_precision}')
                        print(f'Test Recall = {test_recall}')            
                        print(f'Test Loss= {test_loss}')              
                        print('Now Deleting Model')
                        del model
                        try:
                            os.remove(model_outpath + bestmodel_name)    
                        except:
                            pass
                        #reinstantiate model option
                        model_dict = {'basic': BaseModel(params,device)}


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)


