# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")

import math
import numpy as np
from itf_trainable import Itrainable
import copy
from Encodings.encoding import LearnablePositionalEncoding
from Encodings.encoding import CosineEncoding

from torch import Tensor
from typing import Optional

#TODO: breaks in eval mode with AttributeError: 'TBatchNorm' object has no attribute 'weight'
#Problem: in PyTorch transformer encoder layer base class, the sparsity fast path packs the weight and biase of
#         the norm layers into a tensor_args tuple.

class TBatchNorm(nn.Module):
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(TBatchNorm,self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.weight = self.bn.weight
        self.bias = self.bn.bias
        self.eps = self.bn.eps
    
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.bn(x)
        x = x.permute(0,2,1)
        return x

#Directly subclass from nn.BatchNorm1d so that references to weight and bias is inherited
# class TBatchNorm(nn.BatchNorm1d):
#     def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
#         super(TBatchNorm,self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
#         # self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
#         # self.eps = self.bn.eps
    
#     def forward(self,x):
#         x = x.permute(0,2,1)
#         x = super().forward(x)
#         x = x.permute(0,2,1)
#         return x

class TransformerEncoderLayer_vis(nn.TransformerEncoderLayer):
    """torch.nn.TransformerEncoderLayer modified to retain the attention weights for
    later visualisation."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super(TransformerEncoderLayer_vis,self).__init__(d_model, nhead, dim_feedforward,
                                                         dropout, activation,
                                                         layer_norm_eps, batch_first, norm_first,
                                                         device, dtype)
        self.attn_ = {}
        
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        self.attn_['sa'] = attn_map.detach()
        return self.dropout1(x)


class TransformerDecoderLayer_vis(nn.TransformerDecoderLayer):
    """torch.nn.TransformerDecoderLayer modified to retain the attention weights for
    later visualisation."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super(TransformerDecoderLayer_vis,self).__init__(d_model, nhead, dim_feedforward,
                                                         dropout, activation,
                                                         layer_norm_eps, batch_first, norm_first,
                                                         device, dtype)
        self.attn_ = {}
        
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        self.attn_['sa'] = attn_map.detach()
        return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn_map = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        self.attn_['mha'] = attn_map.detach()
        return self.dropout2(x)


class Transformer_Base(nn.Module):
    def __init__(self, seq_len=2560, out_seq_len=24, inp_dim=1, emb_dim=64,\
                 n_heads=4, n_enc_layers=2, n_dec_layers=2, ffdim=128,
                 pe_drop = 0.1,
                 ffdrop = 0.5,
                 attn_drop = 0.2):
        super(Transformer_Base,self).__init__()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len
        self.emb_dim = emb_dim

        self.input_linear = nn.Linear(inp_dim,emb_dim)

        self.pe = LearnablePositionalEncoding(emb_dim,dropout=pe_drop,max_len=seq_len)
        #self.pe = CosineEncoding(emb_dim,max_len=seq_len,scale_factor=0.5)

        drop_p = ffdrop
        self.trf_el = torch.nn.TransformerEncoderLayer(emb_dim,n_heads,ffdim,
                                                      activation=F.gelu,dropout=drop_p,
                                                      batch_first=True,norm_first=True)
       
        self.trf_dl = torch.nn.TransformerDecoderLayer(emb_dim,n_heads,ffdim,
                                activation=F.gelu,dropout=drop_p,
                                batch_first=True,norm_first=True)
               
        self.trf_el.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        self.trf_el.norm2 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=seq_len)
        
        self.trf_dl.norm1 = TBatchNorm(num_features=emb_dim) #nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm2 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        self.trf_dl.norm3 = TBatchNorm(num_features=emb_dim)#nn.BatchNorm1d(num_features=out_seq_len)
        
        # self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=n_enc_layers)
        # self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=n_dec_layers)
        self.trf_e = None
        self.trf_d = None
        self.constructTransformer(n_enc_layers, n_dec_layers)
        
        self.out = nn.Linear(emb_dim,inp_dim)
        self.drop = nn.Dropout(p=drop_p)
    
    def constructTransformer(self,num_enclayers,num_declayers):
        self.trf_e = nn.TransformerEncoder(self.trf_el,num_layers=num_enclayers)
        self.trf_d = nn.TransformerDecoder(self.trf_dl,num_layers=num_declayers)
        
    def forward(self,x):
    #def forward(self,x, pred_gt = None):
        '''x: [batch dim, sequence length, variable dim]'''
        
        #Normalize input
        # E_x = x.mean(dim=-2,keepdim=True)
        # Var_x = x.var(dim=-2,keepdim=True)
        # sdx = torch.sqrt(Var_x+1e-6)
        # nx = (x-E_x)/sdx
        nx=x
        x1 = self.input_linear(nx) # x1.shape: [batch, seq len, emb dim]
        x1 =  self.pe(x1.permute(1,0,2)).permute(1,0,2)
        mem = self.trf_e(x1)
        
        iv = torch.zeros((x.shape[0],self.out_seq_len,x.shape[2]),device=mem.device)
        #pred_task_iv = self.dec_iv.broadcast_to((x.shape[0],self.out_seq_len,1))
        iv = self.input_linear(iv)
        iv = self.pe(iv.permute(1,0,2)).permute(1,0,2)
        #pred_task_iv = self.pe(pred_task_iv.permute(1,0,2)).permute(1,0,2)
        
        x2 = self.trf_d(iv,mem)
        o1 = self.out(x2)
        
        #Denormalize output
        # uo = sdx*o1 + E_x
        uo = o1
        return uo

class DPTrainable(Itrainable):
    def __init__(self,module):
        self.module = module
        self.dpmod = torch.nn.DataParallel(self.module)
        #self.dpmod = self.module
    
    def train_epoch(self, loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: Optional[torch.device],
                    scaler: Optional[torch.cuda.amp.GradScaler]):
        """
        Runs 1 epoch over the training dataloader and updates weights through the optimizer.
        Validation and learning rate schedulers should be run outside this function.
        """
        
        self.dpmod.train()
        
        losses = np.zeros(len(loader))
        
        optimizer.zero_grad()
        for batch_idx, (src, tar) in enumerate(loader):
            src_ = src.to(device)
            tar_ = tar.to(device)
            
            src_ = src_.nan_to_num()
            #print(src_.shape)
            # with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            with torch.cuda.amp.autocast(dtype = torch.float16):
                out = self.dpmod(src_)

                loss = torch.nn.MSELoss(reduction='mean')(out[~tar_.isnan()],
                                                          tar_[~tar_.isnan()])
            
            losses[batch_idx] = loss.detach().item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        return np.mean(losses), np.std(losses,ddof=1)
    
    def val(self, loader: torch.utils.data.DataLoader, loss_fn, device):
        self.dpmod.eval()
        
        #losses = [None]*len(loader)
        losses = None
        multilossfns = (type(loss_fn) == list)
        
        if multilossfns:
            losses = [[None for i in range(len(loader))] for j in range(len(loss_fn))]
        else:
            losses = [[None for i in range(len(loader))]]
            
        with torch.no_grad():
            for batch_idx, (src, tar) in enumerate(loader):
                src_ = src.to(device)
                tar_ = tar.to(device)
                
                src_ = src_.nan_to_num()
                out = self.dpmod(src_)
                
                out_ = out #(25.695 - 9.581)*out + 9.581
                # loss = torch.nn.MSELoss(reduction='none')(out_, tar_).\
                #         nanmean(dim=-2).\
                #         sqrt_()
                
                if multilossfns:
                    for i in range(len(loss_fn)):
                        loss = loss_fn[i](out_,tar_)
                        losses[i][batch_idx] = loss.detach()
                    
                else:
                    loss = loss_fn(out_,tar_)    
                    losses[0][batch_idx] = loss.detach()
        
        for i in range(len(losses)):
            losses[i] = torch.cat(losses[i],dim=0)
        
        stdevs = copy.deepcopy(losses)
        for i in range(len(losses)):
            losses[i] = torch.nanmean(losses[i]).item()
            stdevs[i] = torch.std(stdevs[i][~stdevs[i].isnan()],unbiased=True).item()
        
        if not multilossfns:
            losses = losses[0]
            stdevs = stdevs[0]
            
        return losses, stdevs
        #return torch.nanmean(losses), torch.std(losses[~losses.isnan()],unbiased=True)