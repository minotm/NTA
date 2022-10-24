#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: mminot
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    

class CNN(nn.Module):
    def __init__(self, input_size, hparams, conv_filters, dense_nodes,n_out, kernel_size, dropout):
        super(CNN, self).__init__()
        
        dilation, stride_var = 1, 1
        if hparams['data'] == 'aav': pad = 0
        else: pad = 1
        mp_pad = 0
        maxpool_kernel = 2
        mp_stride = maxpool_kernel
        self.hparams = hparams.copy()
        if hparams['seq_type'] == 'aa' and hparams['ngram'] == 'unigram': input_vector_len = 21
        if hparams['seq_type'] == 'dna' and hparams['ngram'] == 'o2o': input_vector_len = 21
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'unigram': input_vector_len = 5
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'trigram_only': input_vector_len = 62
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'tri_unigram': input_vector_len = 66
    
        #================= Standard CNN ==========================================
        conv_out_size  =round( ( ( input_vector_len + 2*pad - dilation*(kernel_size - 1) - 1 ) /stride_var) + 1)
        mp_out_size = int( ( ( conv_out_size + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        mp_times_filters = int(mp_out_size * conv_filters)
        transition_nodes = mp_times_filters
        
        self.conv_bn_relu_stack = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size = kernel_size, padding=pad, stride = stride_var, bias=False),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel, stride = mp_stride, padding = mp_pad)
            )

        self.flatten = nn.Flatten()        
        self.dropout = nn.Dropout(p=dropout)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(transition_nodes, dense_nodes), 
            nn.ReLU(),
            )
        self.out_layer = nn.Linear(dense_nodes,n_out)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask = None):
        x = x.float()
        x = self.conv_bn_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.dropout(x)
        return self.out_layer(x)
    
        


class Transformer(nn.Module):

    def __init__(self, ntoken, emb_dim, nhead, nhid, nlayers, n_classes, seq_len, dropout=0.2, out_dim = 512):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, emb_dim)
        self.emb_dim = emb_dim
        self.relu = nn.ReLU()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)        
        self.flatten = nn.Flatten()
        self.decoder = nn.Linear( int(seq_len * emb_dim), out_dim)
        self.out_layer = nn.Linear( out_dim , n_classes )
        self.init_weights()
        self.decoder2 = nn.Linear( int( emb_dim), out_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.out_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    
    
    def forward(self, src, input_mask = None):
        src = self.encoder(src) * math.sqrt(self.emb_dim)
        src = self.pos_encoder(src)
        
        if input_mask is not None:
            output = self.transformer_encoder(src, src_key_padding_mask = input_mask)
        elif input_mask == None:
            output = self.transformer_encoder(src)
        output = self.flatten(output) 
        output = self.decoder(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.out_layer(output)
        return output
    

class cnn2layer(nn.Module):
    def __init__(self, input_size, hparams, conv_filters, dense_nodes,n_out, kernel_size, dropout):
        super(cnn2layer, self).__init__()
        
        pad, dilation, stride_var = 1, 1, 1
        maxpool_kernel = 2
        kernel_2 = 3
        mp_stride = maxpool_kernel
        if hparams['data'] == 'aav' and hparams['ngram'] != 'unigram': pad = 0
        else: pad = 1
        
        if hparams['data'] == 'gb1' and hparams['ngram'] == 'unigram': mp_pad = 1
        else: mp_pad = 1
        
        self.hparams = hparams.copy()
        if hparams['seq_type'] == 'aa' and hparams['ngram'] == 'unigram': input_vector_len = 21
        if hparams['seq_type'] == 'dna' and hparams['ngram'] == 'o2o': input_vector_len = 21
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'unigram': input_vector_len = 5
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'trigram_only': input_vector_len = 62
        elif hparams['seq_type'] == 'dna' and hparams['ngram'] == 'tri_unigram': input_vector_len = 66
        
        conv_out_size  = math.floor( ( ( input_vector_len + 2*pad - dilation*(kernel_size - 1) - 1 ) /stride_var) + 1)
        
        mp_out_size = math.floor( ( ( conv_out_size + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        
        conv_out_size2 = math.floor( ( ( mp_out_size + 2*pad - dilation*(kernel_2 - 1) - 1 ) /stride_var) + 1)
        
        mp_out_size2 = math.floor( ( ( conv_out_size2 + 2*mp_pad - dilation*(maxpool_kernel - 1) - 1 ) /mp_stride) + 1)
        
        transition_nodes =  math.floor( (conv_filters / 2) * mp_out_size2)

        self.conv_bn_relu_stack = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size = kernel_size, padding=pad, stride = stride_var, bias=False),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel, padding = mp_pad),
            nn.Conv1d(conv_filters, int(conv_filters / 2), kernel_size = kernel_2, padding=pad, stride = stride_var, bias=False),
            nn.BatchNorm1d(int(conv_filters/2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel, padding = mp_pad)
            )
        self.flatten = nn.Flatten()        
        self.dropout = nn.Dropout(p=dropout)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(transition_nodes, dense_nodes), 
            nn.ReLU(),
            )
        self.out_layer = nn.Linear(dense_nodes,n_out)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask = None, return_h = False):
        x = x.float()
        if mask is not None:
            x = x * mask
        x = self.conv_bn_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.dropout(x)
        
        return self.out_layer(x)
