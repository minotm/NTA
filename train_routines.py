#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import copy
from models import *
from torchmetrics import Accuracy, F1, MatthewsCorrcoef, SpearmanCorrcoef, MeanSquaredError, AUROC, Precision, Recall, Specificity, ROC, ConfusionMatrix
from scipy.stats import spearmanr



class BaseModel(nn.Module):
    def __init__(self, params, device):
        """
        init for Base Model class whichgenerates the model specified by params and contains all
        necessary training and testing functions for both regression and classification
        
        Parameters
        ----------
        params: dictionary
            dictionary of parameters specified by argparser detailing model architecture, hyperparameter
            and training elements
        device: 
            GPU or CPU for model training/testing
        
        Returns
        -------
        None 
        """    
        super().__init__()
        self.hparams = copy.deepcopy(params)
        self.device = device
        self.num_classes = 1
        
        #assign model input shape for dataset, sequence type (aa or dna) and ngram encoding  
        if self.hparams['data'] == 'her2' and  self.hparams['seq_type'] == 'aa' and self.hparams['ngram'] == 'unigram': self.input_size = 10
        if self.hparams['data'] == 'her2' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'unigram': self.input_size = 30
        if self.hparams['data'] == 'her2' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'tri_unigram': self.input_size = 40
        if self.hparams['data'] == 'her2' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'trigram_only': self.input_size = 10

        if self.hparams['data'] == 'gb1' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'tri_unigram' : self.input_size = 16
        if self.hparams['data'] == 'gb1' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'trigram_only' : self.input_size = 4
        if self.hparams['data'] == 'gb1' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'unigram' : self.input_size = 12
        if self.hparams['data'] == 'gb1' and  self.hparams['seq_type'] == 'aa' and self.hparams['ngram'] == 'unigram' : self.input_size = 4
        if self.hparams['data'] == 'gb1' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'o2o' : self.input_size = 4
                
        if self.hparams['data'] == 'aav' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'trigram_only' : self.input_size = 42
        if self.hparams['data'] == 'aav' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'tri_unigram' : self.input_size = 168
        if self.hparams['data'] == 'aav' and  self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'unigram' : self.input_size = 126
        if self.hparams['data'] == 'aav' and  self.hparams['seq_type'] == 'aa' and self.hparams['ngram'] == 'unigram' : self.input_size = 42
        
        #assign transformer embedding vocabulary size based on sequence type (aa or dna) and ngram encoding
        if self.hparams['seq_type'] == 'aa': self.ntokens = 21
        if self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'unigram' : self.ntokens = 5
        if self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'tri_unigram' : self.ntokens = 66
        if self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'trigram_only' : self.ntokens = 62
        #if self.hparams['seq_type'] == 'aa' and self.hparams['ngram'] == 'tri_unigram' : self.ntokens = 9067        
        
        #assign sequence type to be passed to cnn for model architecture sizing
        if self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'trigram_only': sequence_type = self.hparams['seq_type'] + '_trigram'
        elif self.hparams['seq_type'] == 'dna' and self.hparams['ngram'] == 'tri_unigram': sequence_type = self.hparams['seq_type'] + '_uni_and_trigram'
        else: sequence_type = self.hparams['seq_type']
        
        #base model selection
        if self.hparams['model_name'] == 'cnn':
            filters, dense  = 64, 512
                
            self.model = CNN(input_size = self.input_size, hparams = self.hparams, conv_filters = filters, 
                                 dense_nodes = dense, n_out = self.num_classes, kernel_size = self.hparams['kernel'], dropout = self.hparams['p_dropout']).to(self.device)

        elif self.hparams['model_name'] == 'transformer' :
            embedding, nheads, nhidden, num_layers = 32, 2, 128, 1
                
            self.model = Transformer(ntoken = self.ntokens, emb_dim = embedding, nhead = nheads, nhid = nhidden, nlayers = num_layers, 
                     n_classes = self.num_classes, seq_len = self.input_size, dropout = self.hparams['p_dropout'], 
                     out_dim = 512).to(self.device)

        elif self.hparams['model_name'] == 'cnn2':
            filters, dense  = 1024, 1024
            
            self.model = cnn2layer(input_size = self.input_size, hparams = self.hparams, conv_filters = filters, 
                                 dense_nodes = dense, n_out = self.num_classes, kernel_size = self.hparams['kernel'], dropout = self.hparams['p_dropout']).to(self.device)

        elif self.hparams['model_name'] == 't2' :
            embedding, nheads, nhidden, num_layers = 256, 8, 1024, 4
            
            self.model = Transformer(ntoken = self.ntokens, emb_dim = embedding, nhead = nheads, nhid = nhidden, nlayers = num_layers, 
                     n_classes = self.num_classes, seq_len = self.input_size, dropout = self.hparams['p_dropout'], 
                     out_dim = 1024).to(self.device)

        
        #assign selected learning rate & scheduling parameters
        if 'lr' in self.hparams: self.learning_rate = self.hparams['lr']
        else: self.learning_rate = 1e-1
        
        if 'lr_scheduler' in self.hparams: self.lr_scheduler = self.hparams['lr_scheduler']
        else: self.lr_scheduler = False
        
        if 'opt' in self.hparams:
            if self.hparams['opt'] == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

            elif self.hparams['opt'] == 'adam':                
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True, eps = 1e-8, weight_decay=5e-4)

        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        if self.lr_scheduler == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'max', 
                                                                        factor = 0.2, min_lr = 5e-4, patience=10)
        else:
            self.scheduler = None

        #specify appropriate loss function by data set
        if self.hparams['data'] == 'aav'  or self.hparams['data'] == 'gb1':  self.loss_fn = nn.MSELoss().to(self.device)
        elif self.hparams['data'] == 'her2':  self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
    


    def train_step(self,train_dataloader, epoch, tensorboard_writer, batch_size):
        """
        Function for Executing Training Epoch for Regression (GB1 & AAV datasets).
        
        Parameters
        ----------
        train_dataloader: torch.utils.data.dataloader.DataLoader
            dataloader with training data & labels
        epoch: 
            Training features
        tensorboard_writer: SummaryWriter 
            tensorboard logger
        batch_size: 
            batch size used for model training
            
        Returns
        -------
        None
        """
        mean_squared_error = MeanSquaredError().to(self.device)
        self.model.train()

        for batch, (X, labels, mask) in enumerate(train_dataloader):
            X = X.to(self.device)
            labels = labels.to(self.device)
            if mask is not None: mask = mask.to(self.device)          
            pred = self.model(X, mask)
            pred = torch.flatten(pred)
            loss = self.loss_fn(pred,labels)
            self.optimizer.zero_grad() 
            loss.backward()
            
            #=== Gradient Clipping for Transformer when input is long (unigram & tri_unigram encodings) =====
            if self.hparams['model_name'] == 'transformer' and self.hparams['seq_type'] == 'dna' and self.hparams['data'] == 'aav':
                if self.hparams['ngram'] == 'tri_unigram' or self.hparams['ngram'] == 'unigram':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),20.0)
            
            if self.hparams['model_name'] == 't2':# and self.hparams['seq_type'] == 'dna':
                if self.hparams['data'] == 'aav':# and self.hparams['seq_type'] == 'dna': 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),40.0)
                elif self.hparams['data'] == 'gb1' and self.hparams['seq_type'] == 'dna':
                        if self.hparams['ngram'] == 'tri_unigram' or self.hparams['ngram'] == 'unigram':
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.0)
            
                    
            self.optimizer.step()
            with torch.no_grad():
                batch_mse = mean_squared_error(pred,labels)

        epoch_mse = mean_squared_error.compute()
    
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch, len(train_dataloader),
                        100. * batch / len(train_dataloader), epoch_mse))
        tensorboard_writer.add_scalar('train/ MSE', epoch_mse, epoch)
        tensorboard_writer.flush()

                
    def test_step(self, test_loader, epoch, tensorboard_writer):
        """
        Function for Executing Evaluation on Validation or Test Set on Regression Datasets.
        
        Parameters
        ----------
        test_loader: torch.utils.data.dataloader.DataLoader
            dataloader with training data & labels
        epoch: 
            Training features
        tensorboard_writer: SummaryWriter 
            tensorboard logger
            
        Returns
        -------
        epoch_spearman:
            Spearman Rho for epoch
        epoch_mse:
            MSE for epoch
        """
        self.model.eval()
        test_loss = 0
        mean_squared_error = MeanSquaredError(compute_on_step=False).to(self.device)
        
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, (inputs, labels, mask) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if mask is not None: mask = mask.to(self.device)
                outputs = self.model(inputs, mask)

                outputs = torch.flatten(outputs)
                test_loss +=torch.nn.functional.mse_loss(outputs, labels).item()
                
                pred_list.append(outputs)
                label_list.append(labels)
                
                mean_squared_error(outputs, labels)
        
        epoch_mse = mean_squared_error.compute()

        pred_list = torch.cat(pred_list).cpu().numpy()
        label_list = torch.cat(label_list).cpu().numpy()
        epoch_spearman =spearmanr(label_list, pred_list).correlation
        
        if self.scheduler is not None:
            print(epoch_spearman)
            self.scheduler.step(epoch_spearman)
            print(self.optimizer.param_groups[0]['lr'])
                
        tensorboard_writer.add_scalar('val/ MSE', epoch_mse, epoch)
        tensorboard_writer.add_scalar('val/ Rho', epoch_spearman, epoch)
        tensorboard_writer.flush()
        return epoch_spearman, epoch_mse
    
    def train_step_classifier(self,train_dataloader, epoch, tensorboard_writer, batch_size):
        """
        Function for Executing Training Epoch for Her2 Classification.
        
        Parameters
        ----------
        train_dataloader: torch.utils.data.dataloader.DataLoader
            dataloader with training data & labels
        epoch: 
            Training features
        tensorboard_writer: SummaryWriter 
            tensorboard logger
        batch_size: 
            batch size used for model training
            
        Returns
        -------
        None
        """
        train_loss = 0
        precision = Precision(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        recall = Recall(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        sigmoid = nn.Sigmoid()
        mcc = MatthewsCorrcoef(num_classes = 2, compute_on_step = False).to(self.device)
        
        self.model.train()
        for batch, (X, labels, _) in enumerate(train_dataloader):
            X = X.to(self.device)
            labels = labels.to(self.device)
            pred = self.model(X)
            pred = torch.flatten(pred)
            loss = self.loss_fn(pred,labels)
            train_loss += loss.item()            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None and epoch < 28:
                self.scheduler.step()            

            with torch.no_grad():
                predicted_values = (sigmoid(pred))
                labels = torch.Tensor.int(labels)
                batch_precision_mtr = precision(predicted_values,labels)
                batch_recall_mtr = recall(predicted_values,labels)
                batch_mcc_mtr = mcc(predicted_values,labels)

        epoch_precision = precision.compute()
        epoch_recall  = recall.compute()
        epoch_train_loss = train_loss / batch
        epoch_mcc = mcc.compute()
        
        print('==========================')
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch, len(train_dataloader),
                        100. * batch / len(train_dataloader), epoch_train_loss))
        tensorboard_writer.add_scalar('Train/ Precision', epoch_precision, epoch)
        tensorboard_writer.add_scalar('Train/ recall', epoch_recall, epoch)
        tensorboard_writer.add_scalar('Train/ Loss', epoch_train_loss, epoch)
        tensorboard_writer.add_scalar('Train/ MCC', epoch_mcc, epoch)
        tensorboard_writer.flush()

                
    def test_step_classifier(self, test_loader, epoch, tensorboard_writer):
        """
        Function for Executing Evaluation on Validation or Test Set on Classification Her2 Dataset.
        
        Parameters
        ----------
        test_loader: torch.utils.data.dataloader.DataLoader
            dataloader with training data & labels
        epoch: 
            Training features
        tensorboard_writer: SummaryWriter 
            tensorboard logger
            
        Returns
        -------
        epoch_f1:
            F1 score for epoch
        epoch_auroc:
            AUROC for epoch
        epoch_test_loss:
            BCE Loss for epoch
        """
        self.model.eval()
        test_loss = 0
        f1 = F1(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        precision = Precision(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        recall = Recall(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = 2, compute_on_step = False).to(self.device)
        specificity = Specificity(num_classes = self.num_classes, compute_on_step = False).to(self.device)
        sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for batch_idx, (inputs, labels, _) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                outputs = torch.flatten(outputs)
                
                test_loss += torch.nn.functional.binary_cross_entropy(sigmoid(outputs), labels).item()
                outputs = (sigmoid(outputs))
                labels = torch.Tensor.int(labels)
                batch_f1_mtr = f1(outputs,labels)
                batch_precision_mtr = precision(outputs,labels)
                batch_recall_mtr = recall(outputs,labels)
                batch_mcc_mtr = mcc(outputs,labels)
                batch_spec_mtr = specificity(outputs,labels)
                
        epoch_f1 = f1.compute()
        epoch_precision = precision.compute()
        epoch_recall  = recall.compute()
        epoch_mcc = mcc.compute()
        epoch_specificity = specificity.compute()
        if batch_idx > 0:
            epoch_test_loss = test_loss / batch_idx
        else: epoch_test_loss = test_loss

        tensorboard_writer.add_scalar('val/ Loss', epoch_test_loss, epoch)
        tensorboard_writer.add_scalar('val/ Precision', epoch_precision, epoch)
        tensorboard_writer.add_scalar('val/ recall', epoch_recall, epoch)
        tensorboard_writer.add_scalar('val/ MCC', epoch_mcc, epoch)
        tensorboard_writer.add_scalar('val/ Specificity', epoch_specificity, epoch)
        tensorboard_writer.flush()
        return epoch_f1, epoch_test_loss, epoch_precision, epoch_recall, epoch_mcc