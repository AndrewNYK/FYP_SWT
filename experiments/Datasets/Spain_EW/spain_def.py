# -*- coding: utf-8 -*-

import os
import subprocess
import torch
from torch.utils.data import Dataset

import pickle
import pgzip
import copy

import datetime

class REE(Dataset):
    """
    Spain, Red Electric Espana dataset. Instantaneous power reported in MW.
    Ranges from around 18.041 GW (18,041 MW) to 41.015 GW (41,015 MW).
    
    https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
    
    Weather data adds 6 additional dimensions.
    """
    def __init__(self,path = '.',start_idx = 0, end_idx = 9999999,
                 seq_len = 816, pred_horz = 24, stride=336, timestamp = True, weather = False):
        assert(end_idx - start_idx > seq_len+pred_horz)
        #if 'aep_tensor.pkl' not in os.listdir(path) or 'aep_timestamps.pkl' not in os.listdir(path):
        if 'spain_dict.pkl.pgz' not in os.listdir(path):
            raise FileNotFoundError(os.listdir(path))
            #subprocess.check_call('python ./aep_.py . aep_dict.pkl.pgz')
        if weather and ('spainWeather.pkl.pgz' not in os.listdir(path)):
            raise FileNotFoundError(os.listdir(path))
            
        # with open(os.path.join(path,'aep_tensor.pkl'),'rb') as f:
        #     series = pickle.load(f)
        # with open(os.path.join(path,'aep_timestamps.pkl'),'rb') as f:
        #     timestamps = pickle.load(f)
        
        with pgzip.open(os.path.join(path,'spain_dict.pkl.pgz')) as f:
            sd = pickle.load(f)
        
        if weather:
            with pgzip.open(os.path.join(path,'spainWeather.pkl.pgz')) as f2:
                self.weather_dict = pickle.load(f2)
        
        self.return_timestamps = timestamp
        self.return_weather = weather
        
        series = sd["tensor"] #35064 total elements
        starttime = sd['start_time']
        
        end_idx = min(end_idx,len(series))
        
        wset = []
        ser_start_ = []
        pred_start_ = []
        i = start_idx
        while i + seq_len + pred_horz < end_idx:
            wset.append(series[i:i+seq_len+pred_horz])
            # ser_start.append(timestamps[i])
            # pred_start.append(timestamps[i+seq_len+1])
            ser_start_.append(starttime + i*datetime.timedelta(hours=1))
            pred_start_.append(starttime + (i + seq_len)*datetime.timedelta(hours=1))
            i += stride
        
        wset = torch.stack(wset).unsqueeze(-1)
        self.series = wset
        
        ser_start = [None]*len(ser_start_)
        pred_start = [None]*len(pred_start_)
        for i in range(len(ser_start)):
            tmptime = ser_start_[i]
            ser_start[i] = [tmptime.year,
                            tmptime.month,
                            tmptime.day,
                            tmptime.hour,
                            tmptime.minute,
                            tmptime.second]
            
        for i in range(len(pred_start)):
            tmptime = pred_start_[i]
            pred_start[i] = [tmptime.year,
                            tmptime.month,
                            tmptime.day,
                            tmptime.hour,
                            tmptime.minute,
                            tmptime.second]

        #Series normalization (Handled outside)
        #Convert nans so that they do not count toward the min/max
        # smin = self.series.nan_to_num(nan=torch.finfo(self.series.dtype).max).amin(dim=-2,keepdim=True)
        # smax = self.series.nan_to_num(nan=torch.finfo(self.series.dtype).min).amax(dim=-2,keepdim=True)
        
        #Reference tsrnn does this normalization over the entire set at once, not per sample
        self._min = self.series[~self.series.isnan()].min()
        self._max = self.series[~self.series.isnan()].max()
        
        #self.series = (self.series - smin.broadcast_to(self.series.shape))/(smax-smin).broadcast_to(self.series.shape)
        self.series_starttimes_ = ser_start_
        self.pred_start_ = pred_start_
        self.series_starttimes = torch.tensor(ser_start,dtype=torch.long)
        self.pred_start = torch.tensor(pred_start,dtype=torch.long)
        
        self.length = len(self.series)
        self.return_timestamps = timestamp
        
        self.seq_len = seq_len
        self.pred_horz = pred_horz
        
        #Dataset in use will contain nans, to be handled by the training code
        
        if weather:
            #Weather series normalization
            #Only dimensions 0, 3, 4, 5, 6, 7 needs normalization
            wdmin = self.weather_dict['tensor'].nan_to_num(nan=torch.finfo(self.weather_dict['tensor'].dtype).max).amin(dim=-2,keepdim=True)
            wdmax = self.weather_dict['tensor'].nan_to_num(nan=torch.finfo(self.weather_dict['tensor'].dtype).min).amax(dim=-2,keepdim=True)
            self.weather_dict['tensor'] = (self.weather_dict['tensor'] - wdmin.broadcast_to(self.weather_dict['tensor'].shape))\
                /(wdmax-wdmin + 1e-10).broadcast_to(self.weather_dict['tensor'].shape)
            
            self.weather_dict['tensor'] = self.weather_dict['tensor'].type(torch.float32)

    def min(self):
        """Returns the minimum load power in GW"""
        return self._min
    
    def max(self):
        """Returns the maximum load power in GW"""
        return self._max
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if self.return_weather:
            weather_start = (self.series_starttimes_[idx] - self.weather_dict['start_time'])//datetime.timedelta(hours=1)
            wd = self.weather_dict['tensor']\
                [weather_start: weather_start + self.seq_len + self.pred_horz]
            
        if self.return_timestamps and self.return_weather:
            return self.series[idx][:self.seq_len],\
                    wd[:self.seq_len],\
                    self.series[idx][self.seq_len:],\
                    wd[self.seq_len:self.seq_len + self.pred_horz],\
                    self.series_starttimes[idx],\
                    self.pred_start[idx]
        
        elif self.return_timestamps and (not self.return_weather):
            return self.series[idx][:self.seq_len],\
                    self.series[idx][self.seq_len:],\
                        self.series_starttimes[idx],\
                        self.pred_start[idx]
        
        elif (not self.return_timestamps) and self.return_weather:
            return self.series[idx][:self.seq_len], wd[:self.seq_len],\
                    self.series[idx][self.seq_len:], wd[self.seq_len:self.seq_len + self.pred_horz]
        
        elif (not self.return_timestamps) and (not self.return_weather):
            return self.series[idx][:self.seq_len], self.series[idx][self.seq_len:]