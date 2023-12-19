"""Builds an E3D RNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.layers.rnn_cell import Eidetic3DLSTMCell as eidetic_lstm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np

def add_noise_horizon(input_frm, configs):
  days = int((configs.img_height + 1) / 2)

  # add noise on last frame
  flatten = tf.reshape(input_frm[:, -1, :days], [tf.shape(input_frm)[0], -1])  # [bz, seq, 7, 24]=>[bz, 7*24]
  noise = tf.random.normal([tf.shape(input_frm)[0], configs.horizon])  # noise of horizon length
  input_w_noise = tf.concat([flatten[:, :-configs.horizon], noise], 1)  # [bz, 7*24]
  input_w_noise = tf.reshape(input_w_noise, [tf.shape(input_frm)[0], days, -1])  # [bz,7,24]

  # flip
  input_w_noise_reverse = tf.reverse(input_w_noise, [1])
  input_w_noise_whole = tf.concat([input_w_noise, input_w_noise_reverse[:,1:]], axis=1)
  return tf.concat([input_frm[:,:-1], tf.expand_dims(input_w_noise_whole,1)], 1)

def add_gen_horizon(input_frm, configs, gen_images):
  days = int((configs.img_height + 1) / 2)
  height = int(configs.horizon/24)
  flatten_frm = tf.reshape(gen_images[:, -1, :height], [tf.shape(gen_images)[0], -1])
  flatten = tf.reshape(input_frm[:, -1, :days], [tf.shape(input_frm)[0], -1])  # [bz, seq, 7, 24]=>[bz, 7*24]

  input_w_gen = tf.concat([flatten[:, :-configs.horizon], flatten_frm[:,-configs.horizon:]], 1)  # [bz, 7*24]
  input_w_gen = tf.reshape(input_w_gen, [tf.shape(input_frm)[0], days, -1])  # [bz,7,24]

  # flip
  input_w_gen_reverse = tf.reverse(input_w_gen, [1])
  input_w_gen_whole = tf.concat([input_w_gen, input_w_gen_reverse[:,1:]], axis=1)
  return tf.concat([input_frm[:,:-1], tf.expand_dims(input_w_gen_whole,1)], 1)

def get_mask(input_frm, configs):
  # mask with noisy position as 0; noise with noisy position as random [0-1], otherwise 0
  mask_dims = tf.stack([tf.shape(input_frm)[0], configs.img_height, configs.img_width])
  mask = tf.fill(mask_dims, 1.0)
  for j in range(tf.shape(input_frm)[0]):
    sz = np.int(configs.img_height*configs.img_width*configs.noise_percent)
    idx = np.random.randint(low=0, high=configs.img_height*configs.img_width-1, size=sz)
    for i in range(sz):
      mask[j,int(idx[i]/configs.img_width),idx[i] % configs.img_width] = 0
  noise = tf.convert_to_tensor((1-mask)*np.random.rand(tf.shape(input_frm)[0], configs.img_height, configs.img_width),dtype=tf.float32)
  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  return mask, noise

def loss_on_horizon(gen_images, images, frm, configs):
    days = int((configs.img_height + 1) / 2)
    x = tf.reshape(gen_images[:, :, :days], [tf.shape(gen_images)[0], tf.shape(gen_images)[1], -1])
    gx = tf.reshape(images[:, :, :days], [tf.shape(images)[0], tf.shape(images)[1], -1])
    if frm == 'last':
      loss = tf.sqrt(tf.reduce_mean(tf.square(x[:,-1:,-configs.horizon:]-gx[:,-1:,-configs.horizon:])))
    elif frm == 'all':
      loss = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, -configs.horizon:] - gx[:, :, -configs.horizon:])))
    return loss


def rnn(images, images_hole, configs):
  """Builds a RNN according to the config."""
  gen_frame, ground_truth_hole = e3d_hole(images_hole, configs)
  loss = tf.sqrt(tf.reduce_mean(tf.square(gen_frame[:,-1] - ground_truth_hole[:,-1])))

  gen_images, ground_truth = e3d_whole(images, configs)
  loss += loss_on_horizon(gen_images, ground_truth,'last', configs)
  out_ims = gen_frame[:, -1]
  return [out_ims, loss]


def e3d_hole(images, configs):
  gen_images, ground_truth, mask_list, lstm_layer, cell, hidden, c_history = [], [], [], [], [], [], []
  batch_size = tf.shape(images)[0]
  shape = images.get_shape().as_list()
  ims_height = shape[2]
  ims_width = shape[3]
  total_length = shape[1]
  num_hidden = [int(x) for x in configs.num_hidden.split(',')]
  num_layers = len(num_hidden)
  filter_shape = [int(x) for x in configs.filter_shape.split(',')]

  window_length = total_length - 1
  for i in range(num_layers):
    if i == 0:
      num_hidden_in = 1 # 1 channel for input images
    else:
      num_hidden_in = num_hidden[i - 1]
    new_lstm = eidetic_lstm(
        name='e3d' + str(i),
        input_shape=[window_length, ims_height, ims_width, num_hidden_in],
        output_channels=num_hidden[i],
        kernel_shape=[filter_shape[0], filter_shape[1], filter_shape[2]])    # by lyz [2, 5, 5] ->2d
    lstm_layer.append(new_lstm)

    zeros_dims = tf.stack([batch_size, window_length, ims_height, ims_width, num_hidden[i]])
    zero_state = tf.fill(zeros_dims, 0.0)
    cell.append(zero_state)
    hidden.append(zero_state)
    c_history.append(None)

  memory = zero_state
  with tf.variable_scope('generator'):
    input_list = []
    reuse = False

    #breakpoint()
    for time_step in range(total_length-1):
      with tf.variable_scope('e3d-lstm', reuse=tf.AUTO_REUSE):
        input_frm = images[:, time_step]
        input_list.append(input_frm)
        print(time_step)

        if (time_step+1) % window_length == 0:
          input_frm = tf.stack(input_list[-window_length:])
          input_frm = tf.transpose(input_frm, [1, 0, 2, 3])
          ground_truth.append(images[:, time_step+1])
          for i in range(num_layers):
            if time_step == window_length - 1:
              c_history[i] = cell[i]
            else:
              c_history[i] = tf.concat([c_history[i], cell[i]], 1)
            if i == 0:
              inputs = tf.expand_dims(input_frm, axis=4)
            else:
              inputs = hidden[i - 1]
            hidden[i], cell[i], memory = lstm_layer[i](inputs, hidden[i], cell[i], memory, c_history[i])
          #breakpoint()
          x_gen = tf.layers.conv3d(hidden[num_layers - 1], 1, [window_length, 1, 1], [window_length, 1, 1], 'same')
          x_gen = tf.squeeze(x_gen, [1,4])
          gen_images.append(x_gen)

  gen_images = tf.stack(gen_images)
  gen_images = tf.transpose(gen_images, [1, 0, 2, 3])

  ground_truth = tf.stack(ground_truth)
  ground_truth = tf.transpose(ground_truth, [1, 0, 2, 3])

  return [gen_images, ground_truth]

def e3d_whole(images, configs):
  """Builds a RNN to predict last whole frame."""
  gen_images, ground_truth, mask_list, lstm_layer, cell, hidden, c_history = [], [], [], [], [], [], []
  batch_size = tf.shape(images)[0]
  shape = images.get_shape().as_list()
  ims_height = shape[2]
  ims_width = shape[3]
  total_length = shape[1]
  num_hidden = [int(x) for x in configs.num_hidden.split(',')]
  num_layers = len(num_hidden)
  filter_shape = [int(x) for x in configs.filter_shape.split(',')]

  window_length = 2 #total_length
  window_stride = 0
  for i in range(num_layers):
    if i == 0:
      num_hidden_in = 1 # 1 cjhannel for input images
    else:
      num_hidden_in = num_hidden[i - 1]
    new_lstm = eidetic_lstm(
        name='e3d' + str(i),
        input_shape=[window_length, ims_height, ims_width, num_hidden_in],
        output_channels=num_hidden[i],
        kernel_shape=[filter_shape[0], filter_shape[1], filter_shape[2]])    # by lyz [2, 5, 5] ->2d
    lstm_layer.append(new_lstm)

    zeros_dims = tf.stack([batch_size, window_length, ims_height, ims_width, num_hidden[i]])
    zero_state = tf.fill(zeros_dims, 0.0)
    cell.append(zero_state)
    hidden.append(zero_state)
    c_history.append(None)

  memory = zero_state
  #with tf.variable_scope('generator'):
  with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
    input_list = []
    reuse = False

    for time_step in range(window_length - 1):
      input_list.append(tf.zeros_like(images[:, 0]))

    for time_step in range(total_length-1):
      with tf.variable_scope('e3d-whole', reuse=reuse):
        input_frm = images[:, time_step] # [bz,window_length,13,24]
        input_list.append(input_frm)

        if (time_step+1) % (window_length - window_stride) == 0:
          input_frm = tf.stack(input_list[time_step:])
          input_frm = tf.transpose(input_frm, [1, 0, 2, 3])
          ground_truth.append(images[:, time_step + 1])

          for i in range(num_layers):
            if time_step == window_length - 1:
              c_history[i] = cell[i]
            else:
              c_history[i] = tf.concat([c_history[i], cell[i]], 1)
            if i == 0:
              inputs = tf.expand_dims(input_frm, axis=4)
            else:
              inputs = hidden[i - 1]
            hidden[i], cell[i], memory = lstm_layer[i](inputs, hidden[i], cell[i], memory, c_history[i])

          x_gen = tf.layers.conv3d(hidden[num_layers - 1], 1, [window_length, 1, 1], [window_length, 1, 1], 'same')
          x_gen = tf.squeeze(x_gen, [1,4])
          gen_images.append(x_gen)
        reuse = tf.AUTO_REUSE #True

  gen_images = tf.stack(gen_images)
  gen_images = tf.transpose(gen_images, [1, 0, 2, 3])

  ground_truth = tf.stack(ground_truth)
  ground_truth = tf.transpose(ground_truth, [1, 0, 2, 3])

  return [gen_images, ground_truth]