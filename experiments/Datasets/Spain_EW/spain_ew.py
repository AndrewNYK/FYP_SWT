# -*- coding: utf-8 -*-

import datetime

import torch
import pickle
import zipfile
import pgzip
import os

import sys
import re


if __name__ == '__main__':
    args = sys.argv
    
    dset_root = ""
    output_filename = ""
    try:
        dset_root = args[1]
        output_filename = args[2]
    except IndexError:
        dset_root = "."
        output_filename = "spain_dict.pkl"
    
    zf = zipfile.ZipFile(os.path.join(dset_root,"spain_energy_weather.zip"))
    f = zf.open('energy_dataset.csv','r')
    lines = f.readlines()
    
    for i in range(len(lines)):
        if type(lines[i]) != str:
            lines[i] = lines[i].decode()

    #Exclude header
    lines = lines[1:]

    entries = [None]*(len(lines))

    for i in range(len(lines)):
        splitline = lines[i].split(',')
        time = datetime.datetime.fromisoformat(splitline[0])
        time = time.astimezone(tz = datetime.timezone(\
                                    datetime.timedelta(hours=1)))
        #index 26: total load actual
        value = -1
        try:
            value = float(splitline[26])
        except ValueError:
            value = float("NaN")
        # splitline[0] = re.split('-| |:',splitline[0])
        # for j in range(len(splitline[0])):
        #     splitline[0][j] = int(splitline[0][j])
        
        splitline[1] = value
        entries[i] = [time,value]

    entries.sort(key = lambda itm: itm[0])
    
    fl = (entries[-1][0] - entries[0][0])//datetime.timedelta(hours=1) + 1
    
    _series = [float("NaN")]*fl
    start_time = entries[0][0]
    for i in range(len(entries)):
        j = (entries[i][0] - start_time)//datetime.timedelta(hours=1)
        _series[j] = entries[i][1]

    _series = torch.tensor(_series)
    #series_dict contains the 1d tensor and the start time of the whole dataset
    series_dict = {"start_time":start_time,"tensor":_series}
    with pgzip.open(os.path.join(dset_root,output_filename + '.pgz'), "wb",thread=8,blocksize=512*10**3) as ff:
    #ff = open(os.path.join(dset_root,output_filename),'wb')
        pickle.dump(series_dict,ff)
    
    #pickle.dump(series_dict,ff)
    #ff.close()