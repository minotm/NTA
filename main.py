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
from data_helper  import *
import numpy as np
import re
import os


def create_parser():
    parser = argparse.ArgumentParser(description='Nucleotide Augmentation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
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
    parser.add_argument('--opt_id', default='sgd', type=str,
                        help='options sgd, adam')
    parser.add_argument('--ngram', default='unigram', type=str,
                        help='unigram, trigram_only, trigram')
    parser.add_argument('--seq_type', default='aa', type=str,
                        help='aa,dna')
    parser.add_argument('--data_set', default='three_vs_rest', type=str,
                       help='options: three_vs_rest if using gb1, seven_vs_rest if using aav, irrelevant if using her2')
    parser.add_argument('--aug_factor', default='none', type=str,
                       help='data augmentation factor')
    parser.add_argument('--kernel', default=5, type=int,
                       help='size of conv1d kernel')
    parser.add_argument('--single_data_truncation', default='False', type=str,
                       help='runs model for only a single truncated data set. options: True, False')
    parser.add_argument('--trunc', default=0.05, type=float,
                       help='True, False')
    parser.add_argument('--seq_file_type', default='aa', type=str,
                       help='aa, dna')
    parser.add_argument('--aug_type', default='none', type=str,
                       help='iterative, random, codon_shuffle, codon_balance, online, online_balance')
    parser.add_argument('--rand_set_num', default=-1, type=int,
                       help='number of random augmentation data set (int)')
    parser.add_argument('--num_epochs', default=1200, type=int,
                       help='number of epochs')
    parser.add_argument('--subst_frac', default=0.25, type=float,
                       help='fraction of sequence to substitute for online NTA for AAV data')
    parser.add_argument('--prob_of_augmentation', default=0.5, type=float,
                       help='Probability of performing codon synonym swap')
    parser.add_argument('--eval_every_n_epochs', default=10, type=int,
                       help='number of epochs to evaluate test set & save model')
    parser.add_argument('--on_off', default='off', type=str,
                       help='online or offline training approach')
    parser.set_defaults(augment=True)
    
    
    return parser



def main(args):
        
    if torch.cuda.is_available():
         use_cuda = True
    else:
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
        
    if args.base_model == 't2'  or args.base_model == 'transformer': args.batch_size = 32
    elif args.base_model == 'cnn' or args.base_model == 'cnn2': args.batch_size = 256
    
    if args.data_type == 'aav':
        args.learn_rate, args.kernel, args.batch_size = 1e-3, 3, 256
        if args.base_model == 't2': 
            args.learn_rate, args.batch_size= 5e-4, 32
             
            
        
    if args.seq_file_type == 'aa' and not args.aug_type.startswith('online'): args.ngram = 'unigram'
    if args.data_type == 'gb1': args.data_set = 'three_vs_rest'
    elif args.data_type == 'aav' or args.data_type == 'her2': args.data_set = 'seven_vs_rest'
    
    if args.aug_type.startswith('online'):        
        args.seq_type = 'dna'
        args.on_off = 'on'
        if args.aug_type == 'online_none': args.subst_frac = 0.0
        
    base_model_list = [args.base_model]
    if args.seed == 0: seed_list = [1,2,3]
    else: seed_list = [args.seed]
    
    for seed_entry in seed_list:
        torch.manual_seed(seed_entry)
        torch.manual_seed(seed_entry)
        torch.cuda.manual_seed(seed_entry)
        np.random.seed(seed_entry)
        
        args.seed = seed_entry
    
        print(f'SEED {seed_entry}')
        
        
        #====================          Instantiate Model & Optimizer                    ======================
        for base_model in base_model_list:
        
            
            params = {'data': args.data_type, 'model_name': base_model, 'lr': args.learn_rate, 'lr_scheduler': args.lr_scheduler,
                      'p_dropout': args.dropout, 'ngram': args.ngram, 'seq_type': args.seq_type,
                      'opt': args.opt_id, 'kernel': args.kernel, 'batch_size': args.batch_size}
            
            model_dict = {'basic': BaseModel(params,device)}
    
            #====================            Data Fractionation Params                        ====================
            
            if args.data_type == 'gb1':
                truncate_factor_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                
            if args.data_type == 'aav':
                truncate_factor_list = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                
            #NOTE: when using her2 data, 'truncate factor' refers to train imbalance ratio (minority class examples / majority class examples)
            if args.data_type == 'her2':
                truncate_factor_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]          
    
            if args.single_data_truncation == 'True':
                truncate_factor_list = [args.trunc]
            
            #====================            Restrict Train Size  & Load Data===========================
            for trunc_factor in truncate_factor_list:
                
                model_list = [model_dict['basic']]
            
                train_loader, val_loader, test_loader, y_train, y_val, y_test = batch_train_val_test_sets(args, trunc_factor = trunc_factor)
                
                if type(test_loader) == list:
                    test_loader3,test_loader4 = test_loader[3],test_loader[4]
                    test_loader1,test_loader2,test_loader = test_loader[1],test_loader[2],test_loader[0]
                
                train_length = len(y_train)
                print(f'\n Seq Type: {args.seq_type}')
                print(f'ngram: {args.ngram}')
                print(f'Trunc Factor: {trunc_factor}')
                print(f'\nNumber Training Samples: {len(y_train)}')
                print(f'Number Validation Samples: {len(y_val)}')
                print(f'Number Test Samples: {len(y_test)}')
                print(f'\nAugmentation Type = {args.aug_type}')
                print(f'Aug Factor  = {args.aug_factor}')
                
                #====================           Initialize Model                              ======================
                for i in range(len(model_list)):
                    
                    model = model_list[i].to(device)
                    print(f'\nNow Training with base_model {base_model}')
                    print(f'learn rate = {args.learn_rate}')
                    print(f'learn rate scheduler = {args.lr_scheduler}')
                    print(f'optimizer = {args.opt_id}')
                    if args.aug_type.startswith('online'): print(f'subst frac = {args.subst_frac}')
                    
                    #====================== Tensorboard Initialization ===================================
                    current_date = str(datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
                    run_name = f'runs/{args.data_type}_{args.seq_type}_ngram_{args.ngram}_model_{args.base_model}_aug_{args.aug_type}_{args.aug_factor}_trunc_{trunc_factor}_{args.data_set}_seed_{seed_entry}_{current_date}'
                    writer = SummaryWriter(run_name)
                    writer.flush()
                                        
                    hparams = {'data_type': args.data_type, 'seq_type': args.seq_type,'basemodel': args.base_model, 
                               'truncate_factor': trunc_factor, 'train_len': train_length, 
                               'seed': seed_entry, 'optimizer': args.opt_id, 'learn_rate': args.learn_rate, 'ngram': args.ngram,
                               'data_set': args.data_set, 'dropout': args.dropout, 'aug_factor': args.aug_factor, 
                               'aug_type': args.aug_type, 'rand_train_set_num':args.rand_set_num, 'subst_frac': args.subst_frac,
                               'on_off': args.on_off}
                    
                    if args.data_type == 'gb1':
                        if args.on_off == 'on' : args.epochs = 5000
                        else:  args.epochs = 1200
                            
                    elif args.data_type == 'aav': 
                        if args.on_off == 'off': args.epochs = 1200
                        else: args.epochs = 2000
                    
                    patience = 200
                    patience_counter = 0
                    
                    if args.data_type == 'aav' or args.data_type == 'gb1':                                    
                        best_spearman, best_spearman_epoch, best_mse, best_mse_epoch = -10,0,100,0
                        
                        #======================     Train & Eval Cycle          ===================================
                        
                        for epoch in range(args.epochs):
                            
                            model.train_step(train_loader,epoch,writer,args.batch_size)
                            if args.data_type == 'gb1' and 'online' not in args.aug_type: 
                                val_spearman, val_mse,  = model.test_step(val_loader, epoch, writer) 
                            
                            elif epoch % args.eval_every_n_epochs == 0:
                                val_spearman, val_mse,  = model.test_step(val_loader, epoch, writer)
                                
                            if val_mse < best_mse:
                                best_mse = val_mse
                                best_mse_epoch = epoch
                            if val_spearman > best_spearman:
                                best_spearman = val_spearman
                                best_spearman_epoch = epoch
                                patience_counter = 0
                            
                            else:
                                patience_counter += 1
                            print(f'best Spearman = {best_spearman}')                        
                            
                            if epoch >= 200 and patience_counter >= patience and args.on_off != 'on':
                                print('Met Early Stopping Patience')
                                break
                            
                        test_spearman, test_mse  = model.test_step(test_loader, epoch, writer)
                        
                        #the following test metrics are used only for probabilistic / random augmentaiton cases
                        test_spearman0, test_mse0 = 0,0 
                        test_spearman1, test_mse1, test_spearman2, test_mse2 = 0,0,0,0 
                        test_spearman3, test_mse3, test_spearman4, test_mse4 = 0,0,0,0 
                        test_set_number = -1 #used to track which test set is being recorded in csv output file
                        
                        if 'random' in args.aug_type or args.aug_type.startswith('online'):
                            
                            test_spearman0, test_mse0,  = test_spearman, test_mse
                            test_spearman1, test_mse1,  = model.test_step(test_loader1, epoch, writer)
                            test_spearman2, test_mse2,  = model.test_step(test_loader2, epoch, writer)
                            test_spearman3, test_mse3,  = model.test_step(test_loader3, epoch, writer)
                            test_spearman4, test_mse4,  = model.test_step(test_loader4, epoch, writer)
                            
                            test_spear_list = [test_spearman0, test_spearman1, test_spearman2, test_spearman3, test_spearman4]
                            test_mse_list = [test_mse0, test_mse1, test_mse2, test_mse3, test_mse4]
                            
                            for i in range(5):
                                test_set_number = i
                                test_spearman, test_mse = test_spear_list[i], test_mse_list[i]
                                metric_dict = {'output/best_val_spearman': best_spearman, 'output/best_val_mse': best_mse, 
                                               'output/best_spearman_epoch': best_spearman_epoch, 'output/best_mse_epoch': best_mse_epoch,
                                               'output/final_test_spearman': test_spearman, 'output/final_test_mse': test_mse,
                                               'final_epoch': epoch,'output/test_set_number': test_set_number, 'patience_counter': patience_counter}
                                
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
                                filename = f'results/{args.data_type}_{args.data_set}_{base_model}.csv'
            
                                df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
                        
                        else:
                            metric_dict = {'output/best_val_spearman': best_spearman, 'output/best_val_mse': best_mse, 
                                           'output/best_spearman_epoch': best_spearman_epoch, 'output/best_mse_epoch': best_mse_epoch,
                                           'output/final_test_spearman': test_spearman, 'output/final_test_mse': test_mse,
                                           'final_epoch': epoch, 'output/test_set_number': test_set_number, 'patience_counter': patience_counter}
                            
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
                            filename = f'results/{args.data_type}_{args.data_set}_{base_model}.csv'
        
                            df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
                            
                        print(f'SpearmanR on Test Set = {test_spearman}')
                        print(f'MSE on Test Set = {test_mse}')                
                        print('Now Deleting Model')
                        
                        del model
                        
                        #reinstantiate model option
                        model_dict = {'basic': BaseModel(params,device)}
        
                    elif args.data_type == 'her2':
                                            
                        if args.on_off == 'on' : 
                            args.epochs = 400
                        else: args.epochs = 400
                        
                        patience = 40
                        patience_counter = 0    

                        best_f1, best_f1_epoch, best_loss, best_loss_epoch, best_precision, best_recall, best_precision_epoch, best_recall_epoch = 0,0,100,0,0,0,0,0
                        test_recall, test_precision, test_mcc = 0,0, -10
                        val_mcc, best_mcc,best_mcc_epoch = 0,-10,0
                        
                        #======================     Train & Eval Cycle          ===================================
                        for epoch in range(args.epochs):
                            
                            model.train_step_classifier(train_loader,epoch,writer,args.batch_size)
                            
                            if epoch %args.eval_every_n_epochs == 0:
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

                            else:
                                patience_counter += 1                        
                            print(f'best MCC = {best_mcc}')                        
                            print(f'patience_counter = {patience_counter}')
                            if epoch >= 200 and patience_counter >= patience and args.on_off != 'on':
                                print('Met Early Stopping Patience')
                                break

                        test_f1, test_loss, test_precision, test_recall,  test_mcc = model.test_step_classifier(test_loader, epoch, writer)
                        
                        #the following test metrics are used only for random / probabilistic augmentaiton cases
                        test_mcc0, test_loss0 = -10,100,
                        test_mcc1, test_loss1, test_mcc2, test_loss2 = -10,100,-10,100
                        test_mcc3, test_loss3, test_mcc4, test_loss4 = -10,-10,-10,100,
                        test_set_number = -1 #used to track which test set is being recorded in csv output file
                        
                        if 'random' in args.aug_type or args.aug_type.startswith('online'):
                            test_mcc0, test_loss0,  = test_mcc, test_loss
                            test_f11, test_loss1, test_precision1, test_recall1,  test_mcc1 = model.test_step_classifier(test_loader1, epoch, writer)
                            test_f12, test_loss2, test_precision2, test_recall2,  test_mcc2 = model.test_step_classifier(test_loader2, epoch, writer)
                            test_f13, test_loss3, test_precision3, test_recall3,  test_mcc3 = model.test_step_classifier(test_loader3, epoch, writer)
                            test_f14, test_loss4, test_precision4, test_recall4,  test_mcc4 = model.test_step_classifier(test_loader4, epoch, writer)
                            
                            test_mcc_list = [test_mcc0, test_mcc1, test_mcc2, test_mcc3, test_mcc4]
                            test_loss_list = [test_loss0, test_loss1, test_loss2, test_loss3, test_loss4]
                            
                            for i in range(5):
                                test_set_number = i
                                test_mcc, test_loss = test_mcc_list[i], test_loss_list[i]
                                metric_dict = {'output/best_val_mcc': best_mcc, 'output/best_val_loss': best_loss, 'output/best_val_f1': best_f1,
                                               'output/best_mcc_epoch': best_mcc_epoch, 'output/best_loss_epoch': best_loss_epoch,
                                               'output/best_f1_epoch': best_f1_epoch, 'output/final_test_mcc': test_mcc, 
                                               'output/final_test_loss': test_loss, 'output/final_test_f1': test_f1,
                                               'final_epoch': epoch,'output/test_set_number': test_set_number, 
                                               'patience_counter': patience_counter}
                                
                                non_tensor_keys = ['patience_counter', 'final_epoch', 'output/test_set_number']
                                for key, value in metric_dict.items():
                                    #if key == 'output/best_val_mcc' or key == 'output/best_val_loss' or key == 'output/final_test_mcc' or key == 'output/final_test_loss':
                                    if key not in non_tensor_keys:
                                        m = re.search(r'\((.*)\)', str(value))                                
                                        try:
                                            metric_dict[key] = m.group(1)
                                        except:
                                            pass
                                output_dict = {**hparams, **metric_dict}                        
                                output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
                                output_dict['time'] = output_time
                                filename = f'results/{args.data_type}_{args.data_set}_{base_model}.csv'
            
                                df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))
                                
                                print(f'Test MCC = {test_mcc}')
                                print(f'Test Loss= {test_loss}')              
                            
                            print('Now Deleting Model')
                            del model
                            
                            #reinstantiate model option
                            model_dict = {'basic': BaseModel(params,device)}
                                
                        else:
                        
                            test_f1, test_loss, test_precision, test_recall,  test_mcc = model.test_step_classifier(test_loader, epoch, writer)
                            metric_dict = {'output/best_val_mcc': best_mcc, 'output/best_val_loss': best_loss, 'output/best_val_f1': best_f1,
                                           'output/best_mcc_epoch': best_mcc_epoch, 'output/best_loss_epoch': best_loss_epoch,
                                           'output/best_f1_epoch': best_f1_epoch, 'output/final_test_mcc': test_mcc, 
                                           'output/final_test_loss': test_loss, 'output/final_test_f1': test_f1,
                                           'final_epoch': epoch,'output/test_set_number': test_set_number, 
                                           'patience_counter': patience_counter}
            
                            non_tensor_keys = ['patience_counter', 'final_epoch', 'output/test_set_number']
                            for key, value in metric_dict.items():
                                #if key == 'output/best_val_mcc' or key == 'output/best_val_loss' or key == 'output/final_test_mcc' or key == 'output/final_test_loss':
                                if key not in non_tensor_keys:
                                    m = re.search(r'\((.*)\)', str(value))                                
                                    try:
                                        metric_dict[key] = m.group(1)
                                    except:
                                        pass
                                    
                            output_dict = {**hparams, **metric_dict}
                            output_time = str(datetime.datetime.now().strftime("%d-%m-%Y_%H_%M"))
                            output_dict['time'] = output_time                    
                            filename = f'results/{args.data_type}_{args.data_set}_{base_model}.csv'
                            df = pd.DataFrame.from_dict(output_dict, 'index').T.to_csv(filename, mode='a', index=False, header=(not os.path.exists(filename)))    
                            print(f'Test F1 = {test_f1}')
                            print(f'Test MCC = {test_mcc}')
                            print(f'Test Precision = {test_precision}')
                            print(f'Test Recall = {test_recall}')            
                            print(f'Test Loss= {test_loss}')              
                            print('Now Deleting Model')
                            del model
                            
                            #reinstantiate model option
                            model_dict = {'basic': BaseModel(params,device)}

    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
