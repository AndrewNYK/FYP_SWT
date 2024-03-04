# -*- coding: utf-8 -*-

import datetime
import numpy as np
import scipy
import math

import copy

import zipfile

import torch
import pickle
import pgzip
import os

import sys

path = './'
zpath = './spain_energy_weather.zip'
fname = 'weather_features.csv'

zf = zipfile.ZipFile(zpath)


_rows = []
#with open(os.path.join(path,fname)) as ff:
with zf.open(fname) as ff:
    _rows = ff.readlines()

header = _rows[0]
_rows = _rows[1:]

city_weather = {'Valencia': [], 'Madrid': [], 'Bilbao': [], 'Barcelona': [], 'Seville': []}

for i in range(len(_rows)):
    _rows[i] = _rows[i].decode().split(',')
    tmp = [None]*7
    
    #time
    tmp[0] = datetime.datetime.fromisoformat(_rows[i][0]).astimezone(tz = datetime.timezone(\
                                datetime.timedelta(hours=1)))
    
    #City name, already a string
    city = _rows[i][1].strip()
    #temperature: float
    tmp[1] = float(_rows[i][2])
    
    #Pressure
    tmp[2] = float(_rows[i][5])
    #Humidity
    tmp[3] = float(_rows[i][6])
    #Wind speed
    tmp[4] = float(_rows[i][7])
    #wind direction
    tmp[5] = float(_rows[i][8]) #math.cos(2*math.pi*(float(rows[i][1])/360))
    tmp[6] = (lambda x: x - (x>180)*360)(tmp[5]) #math.sin(2*math.pi*(float(rows[i][1])/360))
    
    # #precipType: {'rain': 1, 'snow': 0}
    # tmp[9] = {'rain': 1, 'snow': 0}[rows[i][8]]
    
    # #icon
    # tmp[10] = {'wind': 0,
    #            'clear-night':0.1667,
    #            'partly-cloudy-night':0.3334,
    #            'fog':0.5001,
    #            'cloudy':0.6668,
    #            'partly-cloudy-day':0.8335,
    #            'clear-day':1}[rows[i][9]]
    # #(relative) humidity: float
    # tmp[11] = float(rows[i][10])
    
    # #summary
    # summ_str = rows[i][11]
    # toks = summ_str.split(' and ')
    # windlevel = 0
    # cloudcover = 1

    # for token in toks:
    #     try:
    #         windlevel = {'Breezy':0.5,'Windy':1}[token.strip()]
    #     except KeyError:
    #         pass
        
    #     try:
    #         cloudcover = {'Clear':1,
    #                       'Partly Cloudy':0.75,
    #                       'Mostly Cloudy':0.5,
    #                       'Overcast':0.25,
    #                       'Foggy':0}[token.strip()]
    #     except KeyError:
    #         pass
        
    # tmp[12] = cloudcover; tmp[13] = windlevel
    city_weather[city].append(tmp)

for city in city_weather.keys():
    city_weather[city].sort(key = lambda row: row[0])
#rows.sort(key = lambda row: row[0])

# skips = []
# for i in range(len(rows)-1):
#     td = rows[i+1][4] - rows[i][4]
#     if td != datetime.timedelta(hours=1):
#         skips.append((i,rows[i],td))

rows = city_weather['Madrid']

start_time = rows[0][0]
end_time = rows[-1][0]

#Expand the dataset to include NaNs for missing timeslots
full_len = (end_time - start_time)//datetime.timedelta(hours=1) + 1
nr = [[float('nan')]*7 for i in range(full_len)]

#Strip the datetime field
for i in range(len(rows)):
    time = rows[i][0]
    j = (time - start_time)//datetime.timedelta(hours=1)
    
    nr[j] = rows[i][1:]

nr = np.array(nr)

#Spain weather and power dataset have the same frequency, no need to upsample
# itrf = scipy.interpolate.interp1d(np.arange(len(nr)),nr,axis=0)
# xr = itrf(np.arange(0,len(nr),0.5)[:-1])

xr = nr

for i in range(0,len(xr),1):
    #Select the correct interpolant, not used
    
    # ma = (xr[i-1][4] + xr[i][4] + xr[i+1][4])/3
    # va = ((xr[i-1][4]-ma)**2 + (xr[i][4] - ma)**2 + (xr[i+1][4] - ma)**2)/3
    
    # mb = (xr[i-1][5] + xr[i][5] + xr[i+1][5])/3
    # vb = ((xr[i-1][5]-mb)**2 + (xr[i][5] - mb)**2 + (xr[i+1][5] - mb)**2)/3
    
    th = xr[i][4]
    # if 1.33 * vb < va:
    #     th = xr[i][6]
    xr[i][4] = math.cos(2*math.pi*(th/360))
    xr[i][5] = math.sin(2*math.pi*(th/360))

# for i in range(0,len(xr),2):
#     th = xr[i][5]
#     xr[i][5] = math.cos(2*math.pi*(th/360))
#     xr[i][6] = math.sin(2*math.pi*(th/360))

weather = torch.tensor(xr)
dc = {'start_time':start_time,'tensor':weather}

with pgzip.open(os.path.join(path,"spainWeather.pkl.pgz"),'wb') as fw:
    pickle.dump(dc,fw)