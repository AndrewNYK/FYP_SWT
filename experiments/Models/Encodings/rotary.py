# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class RotaryEncoding(nn.Module):
    r"""Rotary position encoding. arxiv: 2104.09864
    
    Args:
        dim: embedding dimension.
        
        max_len: maximum sequence length.
        
        base: :math:`\theta` in :math:`\theta_n = \theta^\frac{-2n}{d}`
    """
        
    def __init__(self,dim,max_len=2048,base=10000):
        super(RotaryEncoding, self).__init__()
        m1 = torch.tensor([(base**(-(2*(i//2))/dim)) for i in range(dim)],requires_grad=False)
        m1 = m1 * torch.arange(0,max_len,requires_grad=False).reshape((-1,1))
        m1 = torch.cos(m1)
        self.register_buffer('m1',m1)
        
        m2 = torch.tensor([(base**(-(2*(i//2))/dim)) for i in range(dim)],requires_grad=False)
        m2 = m2 * torch.arange(0,max_len,requires_grad=False).reshape((-1,1))
        m2 = torch.tensor([(-1)**(i+1) for i in range(dim)]) * torch.sin(m2)
        self.register_buffer('m2',m2)
        
        self.m2p = torch.tensor([2*(i//2) + (-(i + 1) %2) for i in range(dim)],dtype=torch.long)

    def forward(self, x):
        r"""Rotary position encoding. arxiv: 2104.09864
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        slen = x.shape[1]
        #print(x.shape)
        #xp = x.permute(1,0,2)
        xp = x
        x_ = xp[...,self.m2p]
        
        #breakpoint()
        rotx = (xp * self.m1[:slen,...] + x_ * self.m2[:slen,...])
        #rotx = rotx.permute(1,0,2)
        
        return rotx