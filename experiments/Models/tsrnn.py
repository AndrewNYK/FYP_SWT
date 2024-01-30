# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy

from torch import Tensor
from typing import Optional
from itf_trainable import Itrainable
# Interface class which contains the training and validation functions.

class Eidetic_LSTM(nn.Module):
    def __init__(self,conv_ndims,
                   input_shape,
                   output_channels,
                   kernel_shape,
                   layer_norm=True,
                   forget_bias=1.0):
        """Construct EideticLSTMCell.

        Args:
          conv_ndims: Convolution dimensionality (1, 2 or 3).
          input_shape: Shape of the input as int tuple, excluding the batch size.
              the input, hidden state and global memory share the same image h and w,
              (im_num_ch,(sub)seq_len,im_h,im_w)
          output_channels: int, number of output channels of the conv LSTM.
          kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
          layer_norm: If `True`, layer normalization will be applied.
          norm_gain: float, The layer normalization gain initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
          norm_shift: float, The layer normalization shift initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
          forget_bias: Forget bias.
          name: Name of the module.

        Raises:
          ValueError: If `input_shape` is incompatible with `conv_ndims`.
        """
        super(Eidetic_LSTM,self).__init__()
        
        assert ((conv_ndims == 1)or
               (conv_ndims == 2)or
               (conv_ndims == 3)) ,"conv_ndims must be 1, 2 or 3"
        
        # input_shape (bsz <not included>, nch, seqlen, h,w)
        input_channels = input_shape[0]
        
        if conv_ndims == 1:
            self.inp_conv = nn.Conv1d(input_channels,7*output_channels,kernel_shape,padding="same")
            self.hid_conv = nn.Conv1d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.gmem_conv = nn.Conv1d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.out_cell_conv = nn.Conv1d(output_channels,output_channels,kernel_shape,padding="same")
            self.out_gmem_conv = nn.Conv1d(output_channels,output_channels,kernel_shape,padding="same")
            #Input: cat((new_cell,new_global_memory)), each with output_channels
            self.out_memory_conv = nn.Conv1d(2*output_channels,output_channels,1,padding="same")
        elif conv_ndims == 2:
            self.inp_conv = nn.Conv2d(input_channels,7*output_channels,kernel_shape,padding="same")
            self.hid_conv = nn.Conv2d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.gmem_conv = nn.Conv2d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.out_cell_conv = nn.Conv2d(output_channels,output_channels,kernel_shape,padding="same")
            self.out_gmem_conv = nn.Conv2d(output_channels,output_channels,kernel_shape,padding="same")
            self.out_memory_conv = nn.Conv2d(2*output_channels,output_channels,1,padding="same")
        elif conv_ndims == 3:
            self.inp_conv = nn.Conv3d(input_channels,7*output_channels,kernel_shape,padding="same")
            self.hid_conv = nn.Conv3d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.gmem_conv = nn.Conv3d(output_channels,4*output_channels,kernel_shape,padding="same")
            self.out_cell_conv = nn.Conv3d(output_channels,output_channels,kernel_shape,padding="same")
            self.out_gmem_conv = nn.Conv3d(output_channels,output_channels,kernel_shape,padding="same")
            self.out_memory_conv = nn.Conv3d(2*output_channels,output_channels,1,padding="same")

        #Normalization layers
        #Input shape is: (bsz,seq_len,img_h,img_w,num_ch) >>> (bsz,num_ch,seq_len,img_h,img_w)
        self.hid_norm = nn.LayerNorm((4*output_channels,*input_shape[1:]))
        self.inp_norm = nn.LayerNorm((7*output_channels,*input_shape[1:]))
        self.cell_norm = nn.LayerNorm((output_channels,*input_shape[1:]))
        self.gmem_norm = nn.LayerNorm((4*output_channels,*input_shape[1:]))
        
        self.layer_norm = layer_norm
        self.forget_bias = forget_bias
    
    def _attn(self,q,k,v):
        if len(q.shape) == 4:
            bsz, n_ch, im_h, im_w = q.shape
        elif len(q.shape) == 5:
            bsz, n_ch, seq_l, im_h, im_w = q.shape
        else:
            raise ValueError("Expect q to have 4 or 5 dimensions")
        q = q.reshape((bsz,-1,n_ch))
        k = k.reshape((bsz,-1,n_ch))
        v = v.reshape((bsz,-1,n_ch))
        
        attn_logits = torch.bmm(q,k.transpose(-2,-1))
        attn_wt = F.softmax(attn_logits,dim=-1)
        out = torch.bmm(attn_wt,v)
        
        if len(q.shape) == 4:
            out = out.reshape((bsz,n_ch,im_h,im_w))
        else:
            out = out.reshape((bsz,n_ch,-1,im_h,im_w))
        
        return out

    def forward(self,inps, hid_state, cell, gmem, eidetic_cell):
        hs = self.hid_conv(hid_state)
        inp = self.inp_conv(inps)
        mem = self.gmem_conv(gmem)
        

        if self.layer_norm:
            hs = self.hid_norm(hs)
            inp = self.inp_norm(inp)
            mem = self.gmem_norm(mem)
        
        i_h,g_h,r_h,o_h = hs.chunk(4,dim=1)
        i_x,g_x,r_x,o_x,t_ix,t_gx,t_fx = inp.chunk(7,dim=1)
        
        i_t = F.sigmoid(i_x + i_h)
        r_t = F.sigmoid(r_x + r_h)
        g_t = F.tanh(g_x + g_h)
        
        nc = cell + self._attn(r_t,eidetic_cell,eidetic_cell)
        if self.layer_norm:
            nc = self.cell_norm(nc)
        nc = nc + i_t*g_t
        
        i_m,f_m,g_m,m_m = mem.chunk(4,dim=1)
        
        t_it = F.sigmoid(t_ix + i_m)
        t_ft = F.sigmoid(t_fx + f_m + self.forget_bias)
        t_gt = F.tanh(t_gx + g_m)
        
        new_gmem = t_ft * F.tanh(m_m) + t_it * t_gt
        
        o_c = self.out_cell_conv(nc)
        o_m = self.out_gmem_conv(new_gmem)
        
        output_gate = F.tanh(o_h + o_x + o_c + o_m)
        memory = torch.cat((nc,new_gmem),dim=1)
        memory = self.out_memory_conv(memory)
        output = F.tanh(memory)*F.sigmoid(output_gate)
        
        return output, nc, new_gmem

class TSRNN(nn.Module):
    """Hardcoded with 2 layers.
    
    Args:
        smpl_rate: Numer of samples per day
        pred_horz: Number of samples to predict
    """
    def __init__(self,smpl_rate,pred_horz,num_weeks,#dp,
                 num_lstm_layers = 2,
                 num_lstm_ch = 64,
                 win_len = 2):
        """smpl_rate: Number of samples per day,
        pred_horz: Number of samples to predict,
        """
        super(TSRNN,self).__init__()
        self.m = smpl_rate #in n/day
        self.H = pred_horz
        self.expected_inseq_len = 7*self.m*num_weeks-self.H
        
        self.sub_seq_h = math.ceil(self.H/self.m)
        self.d = num_weeks
        #self.dp = dp #d' for the auxilliary short sequence
        
        self.win_len = win_len
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_ch = num_lstm_ch

        # self.full_im_shape = (2*7-1,self.m,self.d)
        # tf version uses shape ordering (bsz,seqlen,im_h,im_w,nch)
        # convert to (bsz,nch,seqlen,im_h,im_w) for compatibility with torch.nn.conv3d
        self.full_rnn_layers = \
        [Eidetic_LSTM(3,input_shape=(1,win_len,2*7-1,self.m),output_channels = num_lstm_ch, kernel_shape = (3,3,2)),
         Eidetic_LSTM(3,input_shape=(num_lstm_ch,win_len,2*7-1,self.m),output_channels = num_lstm_ch, kernel_shape = (3,3,2)) ]
        self.full_rnn_layers = nn.ModuleList(self.full_rnn_layers)
        
        #self.aux_rnn = Eidetic_LSTM(3,input_shape=(self.sub_seq_h,self.m,self.dp),output_channels = 64, kernel_shape = (3,3,2))
    
        #self.main_rnn_out_conv = nn.Conv3d(num_lstm_ch,1,[win_len,1,1],padding='same')
        self.main_rnn_out_conv = nn.Conv3d(num_lstm_ch,1,[win_len,1,1],[win_len,1,1])
        # self.image_loss = None
    
    # def get_imloss(self):
    #     return self.image_loss
        
    def forward(self,in_seq: Tensor,
                pred_tar: Optional[Tensor]=None):
        """in_seq: shape [bsz,seq_len,1]"""
        bsz = in_seq.shape[0]
        seq_len = in_seq.shape[1]
        long_seq = in_seq
        sub_seq = long_seq[:]
                
        #Reshape time series into images
        noise_seq = torch.empty((bsz,self.H,1),device=in_seq.device).uniform_()
        long_seq = torch.cat((long_seq, noise_seq), dim = -2)
        long_seq_ims = long_seq.reshape((bsz,self.d,7,self.m)).permute(0,2,3,1)
        
        #Double image h by reflection padding the bottom edge
        long_seq_ims = F.pad(long_seq_ims,pad=(0,0,0,0,0,7-1),mode='reflect')
        
        #For training, prepare ground truth images similarly
        gt_seq = torch.empty(0, device=in_seq.device)
        gt_seq_ims = torch.empty(0, device=in_seq.device)
        if self.training:
            assert (pred_tar is not None), "When training, model requires prediction ground truth for constructing ground truth images." 
            assert (pred_tar.shape[0] == bsz) and (pred_tar.shape[1] == self.H), "Ground truth target is of the wrong shape"

            gt_seq = torch.cat((in_seq,pred_tar),dim = -2)
            gt_seq_ims = gt_seq.reshape((bsz,self.d,7,self.m)).permute(0,2,3,1)
            gt_seq_ims = F.pad(gt_seq_ims,pad=(0,0,0,0,0,7-1),mode='reflect')
        
        #Input images are fed 2 at a time, with stride 2. (In actual implementation) <ie. (1,2),(3,4)(5,.)>
        #Conflict with paper description where images are input with stride 1 | (1,2),(2,3),(3,4)...
        #Following the paper description here
        
        #Setup initial states for both layers
        #im_zero = torch.zeros((bsz,1,self.win_len,2*7-1,self.m),device=long_seq_ims.device,requires_grad = False)
        state_zero = torch.zeros(size = (bsz, self.num_lstm_ch, self.win_len, 2*7-1, self.m),
                                 device=long_seq_ims.device)
        
        hidden_state = [state_zero.clone(),state_zero.clone()]
        cell = [state_zero.clone(),state_zero.clone()]
        eid_cell_hist = [ torch.empty((0),device=state_zero.device) , torch.empty((0),device=state_zero.device) ]
        
        main_out = []
        gmem = state_zero
        
        for i_ch in range(self.d-1):
            # for layer in range(len(self.full_rnn_layers)):
            #     #Input shapes are like (bsz, h, w, d = seq_len (win_len/lstm_out_ch)), convert to (bsz,ch,d,h,w)
            #     lyr_in_im = long_seq_ims[:,:,:,i_ch:i_ch+2].permute(0,3,1,2).unsqueeze(1) \
            #         if layer == 0 else hidden_state[layer-1]

            #     eid_cell_hist[layer] = torch.cat([eid_cell_hist[layer], cell[layer]])

            #     hidden_state[layer], cell[layer], gmem = self.full_rnn_layers[layer](lyr_in_im,
            #                                                                           hidden_state[layer],
            #                                                                           cell[layer],
            #                                                                           gmem,
            #                                                                           eid_cell_hist[layer]
            #                                                                           )
            for idx, layer in enumerate(self.full_rnn_layers):
                #Input shapes are like (bsz, h, w, d = seq_len (win_len/lstm_out_ch)), convert to (bsz,ch,d,h,w)
                lyr_in_im = long_seq_ims[:,:,:,i_ch:i_ch+2].permute(0,3,1,2).unsqueeze(1) \
                    if idx == 0 else hidden_state[idx-1]

                eid_cell_hist[idx] = torch.cat([eid_cell_hist[idx], cell[idx]])

                hidden_state[idx], cell[idx], gmem = layer(lyr_in_im,
                                                           hidden_state[idx],
                                                           cell[idx],
                                                           gmem,
                                                           eid_cell_hist[idx])
            
            gen_im = self.main_rnn_out_conv(hidden_state[-1])
            gen_im = gen_im.squeeze(dim=1)
            main_out.append(gen_im.permute(0,2,3,1))

        main_out = torch.cat(main_out,dim=-1)

        #Reference tf implementation uses 2 LSTM networks, a main net that uses the full
        #sequence and a smaller auxillary net that uses a shorter sub-sequence.
        #However, the two networks are separate and do not share parameters, and the 
        #output is taken from only 1 of the 2 nets. Hence, the other network is redundant 
        #and is omitted.

        image_loss = torch.empty(0, device = gt_seq_ims.device)
        if self.training:
            #compute frame reconstruction loss
            _gt_seq_ims = gt_seq_ims[:,:,:,1:]
            image_loss = F.mse_loss(main_out[~_gt_seq_ims.isnan()],_gt_seq_ims[~_gt_seq_ims.isnan()])
        
        fin_frame = main_out[:,:,:,-1]
        fframe_seq = fin_frame[:,:7,:].reshape((bsz,-1))
        
        pred = fframe_seq[:,-self.H:]
        pred = pred.unsqueeze(-1)
        
        # if self.training:
        #     return pred, image_loss
        # else:
        #     return pred
        return pred, image_loss
    
class DPTrainableTSRNN(Itrainable):
    def __init__(self,module,cuda_devices=None):
        self.module = module
        self.dpmod = torch.nn.DataParallel(self.module,device_ids=cuda_devices)
    
    def train_epoch(self, loader, optimizer, # scheduler,
                    device, scaler):
        self.dpmod.train()
        
        losses = np.zeros(len(loader))
        
        optimizer.zero_grad()
        for batch_idx, (src, tar) in enumerate(loader):
            src_ = src.to(device)
            tar_ = tar.to(device)
            
            src_ = src_.nan_to_num()
            
            # with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out, image_loss = self.dpmod(src_,tar_)
                image_loss = image_loss.sum()
                
                loss = torch.nn.MSELoss(reduction='mean')(out[~tar_.isnan()],
                                                          tar_[~tar_.isnan()]) + image_loss
            
            losses[batch_idx] = loss.detach().item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #scheduler.step()
        
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
                out,_ = self.dpmod(src_)
                
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