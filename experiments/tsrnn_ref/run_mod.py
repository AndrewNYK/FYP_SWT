
"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import argparse
import sys
sys.path.append(".")

parser = argparse.ArgumentParser(description="A complete implementation of electricity prediction for chosen dataset",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--train_data_path', default='..\Datasets\PJM_energy_datasets\AEP_hourly.csv', help='train data path.')
parser.add_argument('--valid_data_path', default='..\Datasets\PJM_energy_datasets\AEP_hourly.csv', help='validation data path.')
parser.add_argument('--save_dir', default='checkpoints/aep_48_mask', help='dir to store trained net.')

parser.add_argument('--is_training', default=True, action='store_true', help='training or testing') #T->F
parser.add_argument('--dataset_name', default='aep', help='the name of dataset.')
parser.add_argument('--iweek', type=int, default=0, help='min value of the series for normalization.') # default=25695.0 3746.0
parser.add_argument('--max_value', type=float, default=25695.0, help='maximu value of the series for normalization.') # default=25695.0 3746.0
parser.add_argument('--noise_percent', type=float, default=0.3, help='the percentage of noise embeded in training to avoid overfitting')
parser.add_argument('--seq_length', type=int, default=5, help='image sequence length.')  #5
parser.add_argument('--img_width', type=int, default=24, help='input image width.')
parser.add_argument('--img_height', type=int, default=7, help='input image width.')  # 13 by lyz
parser.add_argument('--horizon', type=int, default=48, help='predicting horizon length')
parser.add_argument('--model_name', default='e3d_lstm', help='The name of the architecture.')
parser.add_argument('--pretrained_model', default='')
#''/home/yanzhu/PROJ/rcGAN/checkpoints/aep_48_9000_0.001/model.ckpt-9000',
                  #  help='.ckpt file to initialize from.')
parser.add_argument('--num_hidden', default='4,4',
                     help='COMMA separated number of units of e3d lstms.')  #default='64,64,64,64'
parser.add_argument('--filter_shape', default='2,2,1', help='filter of a e3d lstm layer.') # 2,3,3
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--batch_size', type=int, default=80, help='batch size for training.') #default=8
parser.add_argument('--max_iterations', type=int, default=1, help='max num of steps.') #default=80000 #10000->1
parser.add_argument('--display_interval', type=int, default=4,
                      help='number of iters showing training loss.')
parser.add_argument('--test_interval', type=int, default=1, help='number of iters for test.') #default=1000
parser.add_argument('--snapshot_interval', type=int, default=1000,
                      help='number of iters saving models.')  #default=1000

FLAGS = parser.parse_args()


def main(_):
  """Main function."""
  if tf.gfile.Exists(FLAGS.save_dir):
    tf.gfile.DeleteRecursively(FLAGS.save_dir)
  tf.gfile.MakeDirs(FLAGS.save_dir)

  print('Initializing models')
  model = Model(FLAGS)

  if FLAGS.is_training:
    train_wrapper(model)
  else:
    test_wrapper(model)

def train_wrapper(model):
  """Wrapping function to train the model."""

  if FLAGS.pretrained_model:
    model.load(FLAGS.pretrained_model)

  # load data
  train_input_handle, test_input_handle = datasets_factory.data_provider(
          FLAGS.dataset_name,
          FLAGS.train_data_path,
          FLAGS.valid_data_path,
          FLAGS.batch_size,
          seq_length=FLAGS.seq_length,
          img_width=FLAGS.img_width,
          img_height=FLAGS.img_height,
          horizon=FLAGS.horizon,
          is_normalized=False,
          iweek=FLAGS.iweek,
          max_value=FLAGS.max_value,
          is_training=True)
  

  for itr in range(1, FLAGS.max_iterations + 1):
    if train_input_handle.no_batch_left():
      train_input_handle.begin(do_shuffle=False)

    # instances dimension: (batch_size, seq_length, img_height, img_width)
    #breakpoint()
    ims, ims_hole = train_input_handle.get_batch()
    #breakpoint()
    trainer.train(model, ims, ims_hole, FLAGS, itr)

    if itr % FLAGS.snapshot_interval == 0:
      model.save(itr)

    if itr % FLAGS.test_interval == 0:
      trainer.test(model, test_input_handle, FLAGS, itr)

    train_input_handle.next()


def test_wrapper(model):
  """Wrapping function to test the model."""
  model.load(FLAGS.pretrained_model)

  test_input_handle = datasets_factory.data_provider(
      FLAGS.dataset_name,
      FLAGS.train_data_path,
      FLAGS.valid_data_path,
      FLAGS.batch_size,
      horizon=FLAGS.horizon,
      seq_length=FLAGS.seq_length,
      img_width=FLAGS.img_width,
      img_height=FLAGS.img_height,
      is_normalized=False,
      iweek=FLAGS.iweek,
      max_value=FLAGS.max_value,
      is_training=False)
  trainer.test(model, test_input_handle, FLAGS, 'test_result')

if __name__ == '__main__':
  tf.app.run()
