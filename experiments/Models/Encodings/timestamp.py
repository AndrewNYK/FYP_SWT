# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import math

class TimestampLearnableEmbedding(nn.Module):
    r"""Inject information about the absolute time of the tokens in the sequence.
        The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Sine and cosine
        functions of the following periods are used:
            - 24h (Daily)
            - 168h (Weekly)
            - 730.485h [365.2425*24/12] (Monthly)
            - 8765.82h [365.2425*24] (Yearly)
        These functions cover the first 8 dimensions. Additional dimensions are learnable.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        interval: duration between timestamps (in hours)
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, interval, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(TimestampLearnableEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        pos_av = torch.zeros(max_len, 8)  # Timestamp encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        timescale = 1./interval
        angular_vel = torch.tensor([(2*math.pi/(24*timescale)),
                                    (2*math.pi/(168*timescale)),
                                    (2*math.pi/(730.485*timescale)),
                                    (2*math.pi/(8765.82*timescale))])
        self.timescale = timescale
        
        self.lpe = torch.tensor(())
        if d_model > 8:
            self.lpe = nn.Parameter(torch.empty((max_len,1,d_model-8)))
            nn.init.uniform_(self.lpe, -1, 1)
            
        angular_vel = angular_vel[:(-1*(-d_model//2))]
        
        pos_av = position * angular_vel
        self.register_buffer('pos_av', pos_av)  # this stores the variable in the state_dict (used for non-trainable variables)
        self.scale_factor = scale_factor

    def forward(self, x, start_time):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            start_time: time of the first item in the sequence (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            start_time: (6) tensor containing [year, month, day, hour, minutes, seconds]
                minutes and seconds are not actually used, included for completeness
            output: [sequence length, batch size, embed dim]
        """
        y = start_time[:,0]
        y_sak = y-(start_time[:,1] < 3)*torch.ones_like(y) #Adjusted years for Sakamoto's alg.

        #Count the number of days since 2000 Jan 1 0000
        days = (start_time[:,0]) * 365 +\
                torch.tensor([0,0,31,59,90,120,151,181,212,243,273,304,334],device=start_time.device)[start_time[:,1]] +\
                (start_time[:,2] - 1)
        #[None,0,31,28,31,30,31,30,31,31,30,31,30]
        days = days + y//4 - y//100 + y//400

        phase = torch.stack(
                ( 2*math.pi*(start_time[:,3]/24 + (start_time[:,4]/(60*24))), #Hour of day
                 2*math.pi*(\
                 ( ( (y_sak + y_sak.div(4,rounding_mode='floor') -\
                      y_sak.div(100,rounding_mode='floor') + y_sak.div(400,rounding_mode='floor') +\
                  torch.tensor([0,3,2,5,0,3,5,1,4,6,2,4],device=start_time.device)[start_time[:,1] - 1] +\
                  start_time[:,2]) % 7)*24 +\
                  start_time[:,3] + (start_time[:,4]/60) )\
                     /168), #Sakamoto's method for day of week
                  2*math.pi*((days*24 + start_time[:,3] + (start_time[:,4]/60)) % 730.485)/730.485, #Average num of hours in 1 month
                  2*math.pi*((days*24 + start_time[:,3] + (start_time[:,4]/60)) % 8765.82)/8765.82
                 ) ).T
        
            
        #angles = self.pos_av.clone()
        angles = self.pos_av.broadcast_to((x.size(1),-1,-1)).permute(1,0,2).clone()
        trunc = min(angles.shape[2],4)

        angles[:,:,0:trunc] = angles[:,:,0:trunc] + phase[:,:trunc]
        
        assert(angles.size(2) == trunc)

        pe = torch.zeros_like(x)
        pe[:,:, 0:2*trunc:2] = torch.cos(angles[:x.size(0),:])
        pe[:,:, 1:2*trunc+1:2] = torch.sin(angles[:x.size(0),:])
        pe[:,:,2*trunc:] = self.lpe[:x.size(0),:,:]

        x = x + self.scale_factor*pe
        return self.dropout(x)

class TimestampCosineEmbedding(nn.Module):
    r"""Inject information about the absolute time of the tokens in the sequence.
        The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Sine and cosine
        functions of the following periods are used:
            - 24h (Daily)
            - 168h (Weekly)
            - 730.485h [365.2425*24/12] (Monthly)
            - 8765.82h [365.2425*24] (Yearly)
        These functions cover the first 8 dimensions. Additional dimensions follow
        relative cosine embedding.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        interval: duration between timestamps (in hours)
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, interval, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(TimestampCosineEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        pos_av = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        timescale = 1./interval
        angular_vel = torch.tensor([(2*math.pi/(24*timescale)),
                                    (2*math.pi/(168*timescale)),
                                    (2*math.pi/(730.485*timescale)),
                                    (2*math.pi/(8765.82*timescale))])
        self.timescale = timescale
        if d_model > 8:
            add_term = torch.exp(torch.arange(0, d_model-8, 2).float() * (-math.log(10000.0) / d_model))
            angular_vel = torch.cat([angular_vel,add_term],dim=-1)
            
        angular_vel = angular_vel[:(-1*(-d_model//2))]
        
        pos_av = position * angular_vel
        self.register_buffer('pos_av', pos_av)  # this stores the variable in the state_dict (used for non-trainable variables)
        self.scale_factor = scale_factor

    def forward(self, x, start_time):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            start_time: time of the first item in the sequence (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            start_time: (6) tensor containing [year, month, day, hour, minutes, seconds]
                minutes and seconds are not actually used, included for completeness
            output: [sequence length, batch size, embed dim]
        """
        y = start_time[:,0]
        y_sak = y-(start_time[:,1] < 3)*torch.ones_like(y) #Adjusted years for Sakamoto's alg.

        #Count the number of days since 2000 Jan 1 0000
        days = (start_time[:,0] - 2000) * 365 +\
                torch.tensor([0,0,31,59,90,120,151,181,212,243,273,304,334],device=start_time.device)[start_time[:,1]] +\
                (start_time[:,2] - 1)
        #[None,0,31,28,31,30,31,30,31,31,30,31,30]
        days = days + (y-2000).div(4,rounding_mode='floor') -\
                        (y-2000).div(100,rounding_mode='floor') +\
                        (y-2000).div(400,rounding_mode='floor')

        phase = torch.stack(
                ( 2*math.pi*(start_time[:,3]/24 + (start_time[:,4]/(60*24))), #Hour of day
                 2*math.pi*(\
                 ( ( (y_sak + y_sak.div(4,rounding_mode='floor') -\
                      y_sak.div(100,rounding_mode='floor') + y_sak.div(400,rounding_mode='floor') +\
                  torch.tensor([0,3,2,5,0,3,5,1,4,6,2,4],device=start_time.device)[start_time[:,1] - 1] +\
                  start_time[:,2]) % 7)*24 +\
                  start_time[:,3] + (start_time[:,4]/60) )\
                     /168), #Sakamoto's method for day of week
                  2*math.pi*((days*24 + start_time[:,3] + (start_time[:,4]/60)) % 730.485)/730.485, #Average num of hours in 1 month
                  2*math.pi*((days*24 + start_time[:,3] + (start_time[:,4]/60)) % 8765.82)/8765.82
                 ) ).T
            
        #angles = self.pos_av.clone()
        angles = self.pos_av.broadcast_to((x.size(1),-1,-1)).permute(1,0,2).clone()
        trunc = min(angles.shape[2],4)

        angles[:,:,0:trunc] = angles[:,:,0:trunc] + phase[:,:trunc]
        
        pe = torch.zeros_like(x)
        pe[:,:, 0::2] = torch.cos(angles[:x.size(0),:])
        pe[:,:, 1::2] = torch.sin(angles[:x.size(0),:])

        x = x + self.scale_factor*pe
        return self.dropout(x)