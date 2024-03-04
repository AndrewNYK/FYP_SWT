# -*- coding: utf-8 -*-

import datetime
from zoneinfo import ZoneInfo

import torch
import pickle
import zipfile
import pgzip
import os

import sys
import re

import math

class Bloom:
    def __init__(self):
        self.arr = [0]*(2**14)

    def _hash(self,s,n):
        h = 0x811c9dc5
        sbytes = s.encode()
        
        for i in range(len(sbytes)):
            h = h ^ (sbytes[i] + (n**(i+1))%256)%256
            h = (h * 0x01000193)%(2**32)
        
        return h%(2**14)
    
    def insert(self,s):
        for i in range(4):
            h = self._hash(s,((i+1)**3)%256)
            self.arr[h] = 1
    
    def contains(self,s):
        for i in range(4):
            h = self._hash(s,((i+1)**3)%256)
            if self.arr[h] == 0:
                return False
        return True
    
    def reset(self):
        self.arr = [0]*(2**14)

class EST(datetime.tzinfo):
    def __init__(self):
        super(EST,self).__init__()
    
    def utcoffset(self, dt):
        return datetime.timedelta(hours=-5) + self.dst(dt)
    
    def dst(self, dt):
        # Code to set dston and dstoff to the time zone's DST
        # transition times based on the input dt.year, and expressed
        # in standard local time.
        dston = {2003:datetime.datetime(2003,4,6,2), 2004:datetime.datetime(2004,4,4,2),
                 2005:datetime.datetime(2005,4,3,2), 2006:datetime.datetime(2006,4,2,2),
                 2007:datetime.datetime(2007,3,11,2), 2008:datetime.datetime(2008,3,9,2),
                 2009:datetime.datetime(2009,3,8,2), 2010:datetime.datetime(2010,3,14,2),
                 2011:datetime.datetime(2011,3,13,2), 2012:datetime.datetime(2012,3,11,2),
                 2013:datetime.datetime(2013,3,10,2), 2014:datetime.datetime(2014,3,9,2),
                 2015:datetime.datetime(2015,3,8,2), 2016:datetime.datetime(2016,3,13,2),
                 2017:datetime.datetime(2017,3,12,2), 2018:datetime.datetime(2018,3,11,2),
                 2019:datetime.datetime(2019,3,10,2)}[dt.year]
        dstoff = {2003:datetime.datetime(2003,10,26,2), 2004:datetime.datetime(2004,10,31,2),
                 2005:datetime.datetime(2005,10,30,2), 2006:datetime.datetime(2006,10,29,2),
                 2007:datetime.datetime(2007,11,4,2), 2008:datetime.datetime(2008,11,2,2),
                 2009:datetime.datetime(2009,11,1,2), 2010:datetime.datetime(2010,11,7,2),
                 2011:datetime.datetime(2011,11,6,2), 2012:datetime.datetime(2012,11,4,2),
                 2013:datetime.datetime(2013,11,3,2), 2014:datetime.datetime(2014,11,2,2),
                 2015:datetime.datetime(2015,11,1,2), 2016:datetime.datetime(2016,11,6,2),
                 2017:datetime.datetime(2017,11,5,2), 2018:datetime.datetime(2018,11,4,2),
                 2019:datetime.datetime(2019,11,3,2)}[dt.year]
    
        if dston < dt.replace(tzinfo=None) <= dstoff:
            if dstoff + datetime.timedelta(hours=-1) <= dt.replace(tzinfo=None) <= dstoff:
                #Within 1 hour before dstoff, duplicate timestamps are possible,
                #disambiguated by fold
                if dt.fold == 1:
                    #fold = 1 occurs after dstoff
                    return datetime.timedelta(0)
            return datetime.timedelta(hours=1)
        else:
            return datetime.timedelta(0)

#American Electric Power Co. Major city served: Columbus
#Timezone: US Eastern
#AEP DST duplicate timestamps 2014,2015,2016,2017
#Missing timestamps all occur on DST boundaries

if __name__ == '__main__':
    args = sys.argv
    
    dset_root = ""
    output_filename = ""
    try:
        dset_root = args[1]
        output_filename = args[2]
    except IndexError:
        dset_root = "."
        output_filename = "aep_dict.pkl"
    
    f = open(os.path.join(dset_root,'AEP_hourly.csv'),'r')
    lines = f.readlines()

    #Exclude header
    lines = lines[1:]

    entries = [None]*(len(lines))
    
    filts = [Bloom(),Bloom()]
    fills = [0,0]
    inserts = [[],[]]
    _tofill = 0

    for i in range(len(lines)):
        splitline = lines[i].split(',')
        tstr = splitline[0]
        
        is_second_fold = filts[0].contains(tstr) or filts[1].contains(tstr) 
        fld = 0
        if is_second_fold:
            #print(tstr)
            fld = 1
            
        splittime = re.split('-| |:',splitline[0])
        splitline[0] = datetime.datetime(int(splittime[0]),
                                         int(splittime[1]),
                                         int(splittime[2]),
                                         int(splittime[3]),
                                         int(splittime[4]),
                                         int(splittime[5]),
                                         #tzinfo = ZoneInfo("US/Eastern"),
                                         tzinfo = EST(),
                                         fold = fld)
        #splitline[0] = datetime.datetime.fromisoformat(splitline[0])
        
        if fills[_tofill] >= 50: #Switch filter when at capacity
            _tofill = (_tofill+1)%2
            fills[_tofill] = 0
            filts[_tofill].reset() #reset alternate filter
            inserts[_tofill] = []
            
        filts[_tofill].insert(tstr)
        inserts[_tofill].append(tstr)
        fills[_tofill] += 1
        
        splitline[0] = splitline[0].astimezone(
            datetime.timezone(datetime.timedelta(hours=-5)))
        #Adjust back to US Eastern Standard Time without DST shenanigans
        
        value = -1
        try:
            value = float(splitline[1])
        except ValueError:
            value = float("NaN")
        # splitline[0] = re.split('-| |:',splitline[0])
        # for j in range(len(splitline[0])):
        #     splitline[0][j] = int(splitline[0][j])
        
        splitline[1] = value
        entries[i] = splitline

    entries.sort(key = lambda itm: itm[0])
    
    fl = (entries[-1][0] - entries[0][0])//datetime.timedelta(hours=1) + 1
    
    aep_series = [float("NaN")]*fl
    start_time = entries[0][0]
    for i in range(len(entries)):
        j = (entries[i][0] - start_time)//datetime.timedelta(hours=1)
        # if not math.isnan(aep_series[j]):
        #     print(entries[i][0])
        aep_series[j] = entries[i][1]

    aep_series = torch.tensor(aep_series)
    #series_dict contains the 1d tensor and the start time of the whole dataset
    series_dict = {"start_time":start_time,"tensor":aep_series}
    with pgzip.open(os.path.join(dset_root,output_filename + '.pgz'), "wb",thread=8,blocksize=512*10**3) as ff:
    #ff = open(os.path.join(dset_root,output_filename),'wb')
        pickle.dump(series_dict,ff)
    
    #pickle.dump(series_dict,ff)
    #ff.close()


