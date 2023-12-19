
"""AEP & DAYTON Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

class InputHandle(object):
    """Class for handling dataset inputs."""

    def __init__(self, input_param):
        self.path = input_param['path']
        self.name = input_param['name']
        self.batch_size = input_param['batch_size']
        self.is_normalized = input_param['is_normalized']
        self.is_test = input_param['is_test']
        self.seq_length = input_param['seq_length']
        self.img_width = input_param['img_width']
        self.img_height = input_param['img_height']
        self.horizon = input_param['horizon']
        self.height_in_hole = int(np.floor(input_param['horizon'] / 24))
        self.days = self.img_height #int((self.img_height + 1 )/2)
        self.slots_per_week = self.img_width * self.days
        self.slots_per_week_hole = self.img_width * self.height_in_hole
        self.slots_per_week_seq = self.days * self.img_width + \
                                  (self.seq_length - 1) * self.slots_per_week
        self.slots_per_week_seq_hole = self.height_in_hole * self.img_width + \
                                       (self.seq_length - 1) * self.horizon

        self.iweek = input_param['iweek']
        self.init_train = 4042 * 24  #       + self.iweek * self.slots_per_week

        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load(input_param)

    def flip(self, data_3d):
        tmp_data = np.zeros([data_3d.shape[0], self.img_height, self.img_width])
        for i in range(data_3d.shape[0]):
          tmp_data[i, 0:self.days, :] = data_3d[i]
          for j in range(self.days-1):
            tmp_data[i, self.days+j] = tmp_data[i, self.days-j-2]
        return tmp_data

    def flip_hole(self, data_3d):
        #breakpoint()
        tmp_data = np.zeros([data_3d.shape[0], self.height_in_hole*2-1, self.img_width])
        for i in range(data_3d.shape[0]):
          tmp_data[i, 0:self.height_in_hole, :] = data_3d[i]
          for j in range(self.height_in_hole-1):
            tmp_data[i, self.height_in_hole+j] = tmp_data[i, self.height_in_hole-j-2]
        return tmp_data

    def transform_hole(self):
        self.data['img_seq_hole'] = []
        for i in range(len(self.input_data_slices_hole)):
          data_2d = []
          for j in range(self.seq_length):
            data_2d.append(np.reshape(self.input_data_slices_hole[i][j*self.horizon:j*self.horizon+self.slots_per_week_hole],
                                          [-1, self.img_width]))
          self.data['img_seq_hole'].append(np.asarray(data_2d))#self.flip_hole(np.asarray(data_2d))) # by lyz
          #self.data['img_seq_hole'].append(self.flip_hole(np.asarray(data_2d))) #self.flip_hole(np.asarray(data_2d))) # by lyz
        self.data['img_seq_hole'] = np.asarray(self.data['img_seq_hole'])

    def transform(self):
        self.data['img_seq'] = []
        for i in range(len(self.input_data_slices)):
          data_2d = []
          for k in range(self.seq_length):  # by lyz for j in
            j = self.seq_length - 1  # repeat last frame
            data_2d.append(np.reshape(self.input_data_slices[i][j*self.slots_per_week:j*self.slots_per_week+self.slots_per_week],
                                          [-1, self.img_width]))
          self.data['img_seq'].append(np.asarray(data_2d)) #self.flip(np.asarray(data_2d)))  # by lyz
        self.data['img_seq'] = np.asarray(self.data['img_seq'])

    def load(self, input_param):
        """Load the data."""
        #self.data['input_raw_data'] = np.loadtxt(self.path, delimiter=",", usecols=(1))  # path is a list with size 1
        self.data['input_raw_data'] = np.loadtxt(self.path, delimiter=",", skiprows=1,usecols=(1))  # path is a list with size 1
        train = self.data['input_raw_data'][: self.init_train]
        print("full set  length ",len(self.data['input_raw_data']))
        print("train set length ",len(train))
        print(np.min(train))
        print(np.max(train))
        norm_data = (self.data['input_raw_data'] - np.min(train)) / (np.max(train) - np.min(train) + 1e-10)

        self.input_data_slices = []
        self.input_data_slices_hole = []
        if self.is_test:
          test_ending = self.init_train + self.horizon
          i = test_ending
          while i - self.horizon + 168 <= len(norm_data):
            self.input_data_slices.append(norm_data[i - self.slots_per_week_seq: i])
            self.input_data_slices_hole.append(norm_data[i - self.slots_per_week_seq_hole: i])
            i = i + self.slots_per_week
        else:
          norm_train = norm_data[:self.init_train]
          i = len(norm_train)
          while i-self.slots_per_week_seq >= 0:
            self.input_data_slices.append(norm_train[i - self.slots_per_week_seq: i])
            self.input_data_slices_hole.append(norm_train[i - self.slots_per_week_seq_hole: i])
            i = i - self.horizon

        self.transform()
        self.transform_hole()

        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        """Returns the total number of clips."""
        return self.data['img_seq'].shape[0]

    def begin(self, do_shuffle=True):
        """Move to the begin of the batch."""
        self.indices = np.arange(self.total(), dtype='int32')
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.batch_size <= self.total():
            self.current_batch_size = self.batch_size
        else:
            self.current_batch_size = self.total()
        self.current_batch_indices = self.indices[:self.current_batch_size]

    def next(self):
        """Move to the next batch."""
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None

        if self.current_position + self.batch_size <= self.total():
            self.current_batch_size = self.batch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position >= self.total():
          return True
        else:
          return False

    def get_batch(self):
        if self.no_batch_left():
            return None
        batch_hole = np.zeros(
            (self.current_batch_size, self.seq_length, self.height_in_hole*2-1, self.img_width)).astype(np.float32)  # by lyz
        batch = np.zeros(
            (self.current_batch_size, self.seq_length, self.days, self.img_width)).astype(np.float32)

        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            #batch_hole[i] = self.data['img_seq_hole'][batch_ind]
            batch_hole[i] = self.flip_hole(self.data['img_seq_hole'][batch_ind])
            batch[i] = self.data['img_seq'][batch_ind]
        #return batch, batch_hole
        return batch, batch_hole
