"""Functions to train and evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os.path
#import cv2
import numpy as np

def train(model, ims, ims_hole, configs, itr):
  """Trains a model."""
 # breakpoint()
  cost = model.train(ims, ims_hole)

  if itr % configs.display_interval == 0:
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: '+str(itr))
    print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, save_name):
  """Evaluates a model."""
  print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
  test_input_handle.begin(do_shuffle=False)

  days = int((configs.img_height + 1) / 2)
  rmse_o, mae_o, mape_o, smape_o, nd_o = [], [], [], [], []

  while not test_input_handle.no_batch_left():
    test_ims, test_ims_hole = test_input_handle.get_batch()
    #breakpoint()
    gen_ims = model.test(test_ims, test_ims_hole)
    current_batchsize = test_ims.shape[0]

    # flatten each instance in minibatch to 1d to get last steps of horizon length
    x = np.reshape(gen_ims[:, :days, :], [current_batchsize, -1])
    gx = np.reshape(test_ims_hole[:, -1, :days, :], [current_batchsize, -1])
    x = x[:, -configs.horizon:]
    gx = gx[:, -configs.horizon:]

    rmse = [np.sqrt(np.square(x[j,:] - gx[j,:]).mean()) for j in range(current_batchsize)]
    rmse_o = np.append(rmse_o, rmse)
    mae = [np.abs(x[j,:]- gx[j,:]).mean() for j in range(current_batchsize)]
    mae_o = np.append(mae_o, mae)
    mape = [np.abs((x[j,:] - gx[j,:])/ gx[j,:]).mean() for j in range(current_batchsize)]
    mape_o = np.append(mape_o, mape)
    smape = [2 * (np.abs(x[j,:] - gx[j,:])/(np.abs(gx[j,:]) + np.abs(x[j,:]))).mean()
               for j in range(current_batchsize)]
    smape_o = np.append(smape_o, smape)
    nd = [np.sum(np.abs(x[j,:] - gx[j,:])) / np.sum(gx[j,:]) for j in range(current_batchsize)]
    nd_o = np.append(nd_o, nd)

    test_input_handle.next()

  print('rmse: ' + str(np.asarray(rmse_o).mean()) + 'std: ' + str(np.asarray(rmse_o).std(ddof=1)))
  print('mae: ' + str(np.asarray(mae_o).mean()) + 'std: ' + str(np.asarray(mae_o).std(ddof=1)))
  print('mape: ' + str(np.asarray(mape_o).mean()) + 'std: ' + str(np.asarray(mape_o).std(ddof=1)))
  print('smape: ' + str(np.asarray(smape_o).mean()) + 'std: ' + str(np.asarray(smape_o).std(ddof=1)))
  print('nd: ' + str(np.asarray(nd_o).mean()) + 'std: ' + str(np.asarray(nd_o).std(ddof=1)))
