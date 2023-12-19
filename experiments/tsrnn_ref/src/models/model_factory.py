"""Factory to get E3D-LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from src.models import eidetic_3d_lstm_net
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import time

class Model(object):
  """Model class for E3D-LSTM model."""

  def __init__(self, configs):
    #breakpoint()
    self.configs = configs
    self.x = tf.placeholder(tf.float32, [
            None, self.configs.seq_length,
            int(self.configs.img_height), self.configs.img_width
        ])
    self.x_h = tf.placeholder(tf.float32, [
            None, self.configs.seq_length,
            int(self.configs.horizon/24*2-1), self.configs.img_width
        ])  # by lyz

    # Architecture hyper-parameters from configurations
    with tf.variable_scope(tf.get_variable_scope()):
      # define a model
      self.pred_seq, self.loss_train = self.construct_model(self.x, self.x_h)
      all_params = tf.trainable_variables()

    # keep track of moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.9995)
    maintain_averages_op = tf.group(ema.apply(all_params))

    self.train_op = tf.group(tf.train.AdamOptimizer(self.configs.lr).minimize(self.loss_train),
                               maintain_averages_op)

    # session
    variables = tf.global_variables()
    self.saver = tf.train.Saver(variables)
    init = tf.global_variables_initializer()
    config_prot = tf.ConfigProto()
    self.sess = tf.Session(config=config_prot)
    self.sess.run(init)

  def train(self, inputs, inputs_hole):
    """Trains the model."""
    feed_dict = {self.x: inputs}
    feed_dict.update({self.x_h: inputs_hole})
    loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
    return loss

  def test(self, inputs, inputs_hole):
    """Predicts for inputs using the model"""
    feed_dict = {self.x: inputs}
    feed_dict.update({self.x_h: inputs_hole})
    #Instrumentation point
    start = time.perf_counter()#time.time_ns()
    gen_ims = self.sess.run(self.pred_seq, feed_dict)
    end = time.perf_counter()#time.time_ns()
    print("inference time: {0}".format(end-start))
    return gen_ims

  def save(self, itr):
    """Saves the model."""
    checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
    self.saver.save(self.sess, checkpoint_path, global_step=itr)
    print('saved to ' + self.configs.save_dir)

  def load(self, checkpoint_path):
    """Loads a pre-trained model."""
    print('load model:', checkpoint_path)
    #self.saver.restore(self.sess, checkpoint_path)
    self.saver = tf.train.import_meta_graph(checkpoint_path+'.meta')
    self.saver.restore(self.sess, checkpoint_path)

  def construct_model(self, images, images_hole):
    """Contructs a model."""
    networks_map = {
        'e3d_lstm': eidetic_3d_lstm_net.rnn,
    }

    if self.configs.model_name in networks_map:
      func = networks_map[self.configs.model_name]
      return func(images, images_hole, self.configs)
    else:
      raise ValueError('Name of network unknown %s' % self.configs.model_name)
