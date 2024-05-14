from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import torch
#torch.backends.cudnn.benchmark = True

import sys
import time
sys.path.append("./Models")
import os
os.system('')

from Models.tsrnn import TSRNN, DPTrainableTSRNN
from Models.BigBirdSparse.SparseTransformerBB import TransformerBBSparse, TransformerBBFixed
from Models.transformer_base import Transformer_Base, DPTrainable
from Models.NT.NoAttention import NoAttention
# from Models.tsrnn_vit import TSRNN_vit

from Datasets.LondonSmartMeter.lsm_def import LondonSmartMeter
from Datasets.PJM_energy_datasets.aep_def import AEP #PJM AEP
from Datasets.PJM_energy_datasets.dayton_def import DAYTON
from Datasets.Spain_EW.spain_def import REE #Spain

import copy
import random
#import matplotlib.pyplot as plt
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from tqdm.auto import tqdm

# Takes in calling python file arguments
args = sys.argv

# Set to empty list
args = []

#Default settings for experiment
arg_model = "tsrnn" #Options: 'trfbb', 'tsrnn', 'trfbf'
arg_dset = "lsm" #Datasets -- Spain: 'ree', AEP, DAYTON: 'dyt' London: 'lsm'

attr_dset_smpl_rt = 24 if arg_dset == "AEP" else (48 if arg_dset == "lsm" else 24) #Samples per day. Spain, AEP: 24, London: 48
param_dset_lookback_weeks = 5
param_dset_forecast = 48
param_dset_train_stride = 48 #Choose a coprime value to the forecast so all reading frames are eventually considered
param_dset_test_stride = 'same' #tsrnn paper uses 1 week
param_dset_lookback = param_dset_lookback_weeks*7*attr_dset_smpl_rt - param_dset_forecast

opt_cudadevcs = None

#Transformer only parameters
param_trf_edim = 24
param_trf_heads = 4
param_trf_elyr = 4
param_trf_dlyr = 4
param_trf_ffdim = 256
param_trf_weather = False

#Bigbird only parameters
param_trf_bksz = 48

#Training Params
arg_ini_lr = 1e-3
arg_batchsz = 'auto'
arg_epochs = 1000

#Output settings
arg_outdir = "Output"

# Read arguments
if (len(args) != 1):
    args = args[1:]  # exclude first argument which is the file name
    # Assert that the following values come in argument-value pairs
    assert (len(args) % 2 == 0), "Argument-option mismatch"

    while (len(args) >= 2):
        agn = args[0];
        agv = args[1]

        if agn == '--model':
            assert agv in ['trfbb', 'tsrnn', 'trfbf']
            arg_model = agv
        # Dataset related options
        elif agn == '--dset':
            assert agv in ['aep', 'lsm', 'ree', 'dyt']
            arg_dset = agv
        elif agn == '--lkbckwk':
            param_dset_lookback_weeks = int(agv)
        elif agn == '--frcst':
            param_dset_forecast = int(agv)
        elif agn == '--dset-train-stride':
            if agv == 'same':
                param_dset_train_stride = 'same'
            else:
                param_dset_train_stride = int(agv)
        elif agn == '--dset-test-stride':
            if agv == 'same':
                param_dset_test_stride = 'same'
            else:
                param_dset_test_stride = int(agv)

        # CUDA options
        elif agn == '--devices':
            dev_num = agv.split(',')
            for i in range(len(dev_num)):
                dev_num[i] = int(dev_num[i])
            opt_cudadevcs = dev_num

        # No parameters available for adjusting tsrnn
        # Transformer parameters
        elif agn == '--trf-edim':
            param_trf_edim = int(agv)
        elif agn == '--trf-heads':
            param_trf_heads = int(agv)
        elif agn == '--trf-elyr':
            param_trf_elyr = int(agv)
        elif agn == '--trf-dlyr':
            param_trf_dlyr = int(agv)
        elif agn == '--trf-ffdm':
            param_trf_ffdim = int(agv)

        # Transformer specific option: include weather data
        elif agn == '--weather':
            assert agv in ['True', 'False']
            if agv == 'True':
                param_trf_weather = True

        # Training parameters
        elif agn == '--init-lr':
            arg_ini_lr = float(agv)
        elif agn == '--bchsz':
            if agv == 'auto':
                arg_batchsz = "auto"
            else:
                arg_batchsz = int(agv)
        elif agn == "--epochs":
            arg_epochs = int(agv)
        # Output parameters
        elif agn == '--odir':
            arg_outdir = agv
        else:
            raise ValueError("Unknown argument")

        args = args[2:]

# Compute remaining settings
param_dset_lookback = param_dset_lookback_weeks * 7 * attr_dset_smpl_rt - param_dset_forecast
if param_dset_train_stride == 'same': param_dset_train_stride = param_dset_forecast
if param_dset_test_stride == 'same': param_dset_test_stride = param_dset_forecast
attr_dset_smpl_rt = {'ree': 24, 'aep': 24, 'lsm': 48, 'dyt': 24}[arg_dset]

param_trf_inp_dim = {'ree': 7, 'lsm': 14}[arg_dset] if param_trf_weather else 1

#Compute BBSparse blocksize iteratively
param_trf_bksz = 48
nl_bsz = param_trf_bksz
nu_bsz = param_trf_bksz

while ((param_dset_lookback%nl_bsz) != 0) and ((param_dset_lookback%nu_bsz) != 0):
    nl_bsz = nl_bsz - 1
    nu_bsz = nu_bsz + 1

if (param_dset_lookback % nl_bsz) == 0:
    param_trf_bksz = nl_bsz
else:
    param_trf_bksz = nu_bsz

assert (param_trf_bksz >= 24), "Computed block size too small"
assert (param_trf_bksz <= 64), "Computed block size too large"
if arg_model == 'trfbb':
    print("BigBird block size autoset to: " + str(param_trf_bksz))
# Setup experiment
# Seed RNG
seed = (time.time_ns() // 1000) % 1000000
torch.random.manual_seed(seed)  # 126982

# Setup Model
model = None
dpm = None
if arg_model == "trfbb":
    model = TransformerBBSparse(seq_len=param_dset_lookback,
                                out_seq_len=param_dset_forecast,
                                inp_dim=param_trf_inp_dim,
                                emb_dim=param_trf_edim,
                                n_heads=param_trf_heads,
                                n_enc_layers=param_trf_elyr,
                                n_dec_layers=param_trf_dlyr,
                                block_size=param_trf_bksz,
                                ffdim=param_trf_ffdim)
    dpm = DPTrainable(model)

elif arg_model == 'tsrnn':
    model = TSRNN(smpl_rate=attr_dset_smpl_rt,
                  pred_horz=param_dset_forecast,
                  num_weeks=param_dset_lookback_weeks)
    dpm = DPTrainableTSRNN(model, cuda_devices=opt_cudadevcs)

elif arg_model == 'trfbf':
    model = TransformerBBFixed(seq_len=param_dset_lookback,
                               out_seq_len=param_dset_forecast,
                               inp_dim=param_trf_inp_dim,
                               emb_dim=param_trf_edim,
                               n_heads=param_trf_heads,
                               n_enc_layers=param_trf_elyr,
                               n_dec_layers=param_trf_dlyr,
                               block_size=param_trf_bksz,
                               ffdim=param_trf_ffdim)
    dpm = DPTrainable(model)

# cuda_lead = torch.device("cuda",opt_cudadevcs[0]) if type(opt_cudadevcs) is list else torch.device("cuda:0")
cuda_lead = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(cuda_lead)
model.to(cuda_lead)
# Try to JIT script model
jit_test = torch.jit.script(model)
# print("Model structure:\n",jit_test)
del (jit_test)

#Automatic batchsize
num_active_devices = len(opt_cudadevcs) if type(opt_cudadevcs) is list else torch.cuda.device_count()
if arg_batchsz == 'auto':
    torch.cuda.reset_max_memory_allocated()
    #Dummy passes to measure memory use
    max_mem = 0
    num_batches = 0
    test_limit = 1000
    for step in tqdm([10,3,1]):
        for num_batches in tqdm(range(max(num_batches,step),test_limit,step)):
            try:
                x = None
                im_l = None
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if arg_model in ['tsrnn','tsvit']:
                        x, im_l = model(torch.zeros((num_batches*16,param_dset_lookback,param_trf_inp_dim)).to(cuda_lead),
                                  torch.zeros((num_batches*16,param_dset_forecast,param_trf_inp_dim)).to(cuda_lead))
                        torch.cuda.amp.GradScaler().scale((x*x).sum() + im_l).backward(retain_graph = True)
                    else:
                        x = model(torch.zeros((num_batches*16,param_dset_lookback,param_trf_inp_dim)).to(cuda_lead))
                        torch.cuda.amp.GradScaler().scale((x*x).sum()).backward(retain_graph = True)
                max_mem = torch.cuda.max_memory_allocated(cuda_lead)
            except RuntimeError as err: #trap the runtime OOM error
                if 'CUDA out of memory' not in str(err):
                    raise err
                test_limit = num_batches
                num_batches -= step
                break
            finally:
                if arg_model in ['tsrnn','tsvit']:
                    del(im_l)
                del(x)
                torch.cuda.empty_cache()

    avail_mem = 0
    if type(opt_cudadevcs) is list:
        for dev_num in opt_cudadevcs:
            avail_mem += torch.cuda.mem_get_info(torch.cuda.device(dev_num))[0]
    else:
        for dev in range(torch.cuda.device_count()):
            avail_mem += torch.cuda.mem_get_info(torch.cuda.device(dev))[0]
    auto_bsz = (int(0.87*num_batches)*16*num_active_devices) #int((0.85*avail_mem)//max_mem)*16*num_active_devices
    arg_batchsz = auto_bsz
    print("Max memory allocated: ", max_mem)
    print("Current available memory: ",avail_mem)
    print("Automatic batch size selection: {0} ({1})".format(arg_batchsz,auto_bsz//16))

# Setup Dataset
train_set = val_set = test_set = None
if arg_dset == 'aep':
    train_offset = 0
    val_offset = 9
    # with the definitions for split boundaries, this offset for the start of the validation set ensures the start time matches the train and test sets.
    train_set = AEP(path="Datasets/PJM_energy_datasets",
                    start_idx=train_offset, end_idx=(4042 * 24) + param_dset_forecast - 12129,
                    seq_len=param_dset_lookback,
                    pred_horz=param_dset_forecast,
                    stride=param_dset_train_stride,
                    timestamp=False)
    val_set = AEP(path="Datasets/PJM_energy_datasets",
                  start_idx=val_offset + (4042 * 24) + param_dset_forecast - 12129,
                  end_idx=(4042 * 24) + param_dset_forecast,
                  seq_len=param_dset_lookback,
                  pred_horz=param_dset_forecast,
                  stride=param_dset_train_stride,
                  timestamp=False)
    test_set = AEP(path="Datasets/PJM_energy_datasets",
                   start_idx=(4042 * 24) + param_dset_forecast,  # ~Last 20% of dataset
                   seq_len=param_dset_lookback,
                   pred_horz=param_dset_forecast,
                   stride=param_dset_test_stride,
                   timestamp=False)

    # Monkey patch the dataset to normalize the series
    train_set.series = (train_set.series - 9581) / (25695 - 9581)
    val_set.series = (val_set.series - 9581) / (25695 - 9581)
    test_set.series = (test_set.series - 9581) / (25695 - 9581)

elif arg_dset == 'dyt':
    full_set = DAYTON(path="Datasets/PJM_energy_datasets",
                      seq_len=param_dset_lookback,
                      pred_horz=param_dset_forecast,
                      timestamp=False)
    dytmax = full_set.max()
    dytmin = full_set.min()
    del (full_set)

    train_set = DAYTON(path="Datasets/PJM_energy_datasets",
                       start_idx=0, end_idx=97036,
                       seq_len=param_dset_lookback,
                       pred_horz=param_dset_forecast,
                       stride=29,
                       timestamp=False)
    val_set = DAYTON(path="Datasets/PJM_energy_datasets",
                     start_idx=97036, end_idx=97036 + 12129,
                     seq_len=param_dset_lookback,
                     pred_horz=param_dset_forecast,
                     stride=param_dset_forecast,
                     timestamp=False)
    test_set = DAYTON(path="Datasets/PJM_energy_datasets",
                      start_idx=97036 + 12129,
                      seq_len=param_dset_lookback,
                      pred_horz=param_dset_forecast,
                      stride=param_dset_forecast,
                      timestamp=False)

    train_set.series = (train_set.series - dytmin) / (dytmax - dytmin)
    val_set.series = (val_set.series - dytmin) / (dytmax - dytmin)
    test_set.series = (test_set.series - dytmin) / (dytmax - dytmin)

elif arg_dset == 'ree':
    train_set = REE(path="Datasets/Spain_EW",
                    start_idx=0, end_idx=28051,
                    seq_len=param_dset_lookback,
                    pred_horz=param_dset_forecast,
                    stride=7,
                    timestamp=False, weather=param_trf_weather)
    val_set = REE(path="Datasets/Spain_EW",
                  start_idx=28051, end_idx=28051 + 3506,
                  seq_len=param_dset_lookback,
                  pred_horz=param_dset_forecast,
                  stride=19,
                  timestamp=False, weather=param_trf_weather)
    test_set = REE(path="Datasets/Spain_EW",
                   start_idx=28051 + 3506,
                   seq_len=param_dset_lookback,
                   pred_horz=param_dset_forecast,
                   stride=19,
                   timestamp=False, weather=param_trf_weather)

    reemin = train_set.min()
    reemax = train_set.max()

    # Monkey patch dataset to normalize series
    train_set.series = (train_set.series - reemin) / (reemax - reemin)
    val_set.series = (val_set.series - reemin) / (reemax - reemin)
    test_set.series = (test_set.series - reemin) / (reemax - reemin)

elif arg_dset == 'lsm':

    # Create class
    dset = LondonSmartMeter(path='Datasets/LondonSmartMeter',
                            seq_len=param_dset_lookback,
                            pred_horz=param_dset_forecast, weather=param_trf_weather)

    # Use __getitem__ method in class
    print(dset[0])

    h_idcs = dset.get_household_indices()
    # h_idcs = [(hno,idcs) for hno, idcs in enumerate(h_idcs)]
    random.seed(seed)
    random.shuffle(h_idcs)

    train_idcs, val_idcs, test_idcs = \
        h_idcs[:3 * len(h_idcs) // 5], \
        h_idcs[3 * len(h_idcs) // 5:4 * len(h_idcs) // 5], \
        h_idcs[4 * len(h_idcs) // 5:]

    train_idcs = [idx for h in train_idcs for idx in h]
    val_idcs = [idx for h in val_idcs for idx in h]
    test_idcs = [idx for h in test_idcs for idx in h]

    train_set, val_set = torch.utils.data.Subset(dset, train_idcs), \
                         torch.utils.data.Subset(dset, val_idcs)

    test_set = torch.utils.data.Subset(dset, test_idcs)

# Setup dataloaders
custom_collate = torch.utils.data.default_collate
if param_trf_weather:
    if (arg_dset == 'ree'):
        custom_collate = lambda dat: torch.utils.data.default_collate(
            [(torch.cat((elem[0], elem[1]), dim=-1), elem[2]) for elem in dat])
    # elif (arg_dset == 'lsm'):
    #     custom_collate = lambda dat: torch.utils.data.default_collate(
    #         [(elem[0],elem[1][:,0]) for elem in dat])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=arg_batchsz,
                                           shuffle=True, collate_fn=custom_collate,
                                           pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=arg_batchsz, collate_fn=custom_collate, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg_batchsz, collate_fn=custom_collate)

opt = torch.optim.AdamW(model.parameters(), lr=arg_ini_lr)
scheduler = None
if arg_model == "tsrnn":
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1)

# Transformers
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,
                                                  lr_lambda=lambda epoch: 1 if epoch < 2 * arg_epochs // 6 else
                                                  0.1 if epoch < 3 * arg_epochs // 6 else
                                                  0.03 if epoch < 4 * arg_epochs // 6 else
                                                  0.01)

print(len(next(iter(train_loader))[0]))

scaler = torch.cuda.amp.GradScaler()

best_model_state = None
best_model_dpt = copy.deepcopy(dpm)
best_val = 999999
current_test = 999999

epoch_start = 0
epoch_end = 0
avg_epoch_dur = 0

for i in tqdm(range(arg_epochs), total=len(range(arg_epochs)), desc="epochs"):
    epoch_start = time.time_ns()

    l, ls = dpm.train_epoch(train_loader, opt, device=cuda_lead, scaler=scaler)
    vl, vls = dpm.val(val_loader,
                      loss_fn=lambda x, y: torch.nn.MSELoss(reduction='none')(x, y). \
                      nanmean(dim=-2). \
                      sqrt_(),
                      device=cuda_lead)

    epoch_end = time.time_ns()
    vl_improved = (vl < best_val)
    if vl_improved:
        best_val = vl
        best_model_state = copy.deepcopy(model.state_dict())

    if (i % 30 == 0):
        best_model_dpt.module.load_state_dict(best_model_state)
        current_test, _ = best_model_dpt.val(test_loader, lambda x, y: torch.nn.MSELoss(reduction='none')(x, y). \
                                             nanmean(dim=-2).sqrt_(), device=cuda_lead)
        del (_)
    # print("\r                                                     ",end="\r")
    print("\033[2K\033[1A\033[2K\033[1A\033[2K", end="\r")
    if vl_improved:
        print("Epoch {0}: loss = {1}, val_loss =\033[1;32m {2} \033[1;37m".format(i, l, vl))
    else:
        print("Epoch {0}: loss = {1}, val_loss = {2}".format(i, l, vl))

    # Estimate completion time
    epoch_dur_sec = (epoch_end - epoch_start) / (1000000000)
    if i == 0:
        avg_epoch_dur = epoch_dur_sec
    avg_epoch_dur = 0.3 * epoch_dur_sec + 0.7 * avg_epoch_dur
    etc_sec = (arg_epochs - i - 1) * avg_epoch_dur
    est_hrs = int(etc_sec // 3600)
    est_mins = round((etc_sec % 3600) / 60)
    if est_mins == 60:
        est_mins = 0
        est_hrs += 1
    print("\033[1;36mBest validation loss: {0}\033[1;37m".format(best_val))
    print("\033[1;94mCurrent test loss (Updated every 30 epochs): {0}\033[1;37m".format(current_test))
    print("Estimated time to complete: {0}h {1}m".format(est_hrs, est_mins), end="", flush=True)

print("\n")
final_model_state = model.state_dict()
for key in final_model_state:
    final_model_state[key] = final_model_state[key].cpu().detach().numpy()

#best_model_dpt = copy.deepcopy(dpm)
best_model_dpt.module.load_state_dict(best_model_state)


torch.cuda.empty_cache()

losses = [ lambda x,y: torch.nn.MSELoss(reduction='none')(x, y).\
                          nanmean(dim=-2),
           lambda x,y: torch.nn.MSELoss(reduction='none')(x, y).\
                                     nanmean(dim=-2).sqrt_(),
           lambda x,t: (x-t).abs_().nanmean(dim=-2),
           lambda x,t: (2*(t-x).abs_() / (t.abs() + x.abs())).nanmean(dim=-2)]

test_loss , tls = dpm.val(test_loader,
                          loss_fn = losses,
                          device = cuda_lead)
print("Test loss (Final Epoch): {0}".format(test_loss))
test_loss , tls = best_model_dpt.val(test_loader,
                          loss_fn = losses,
                          device = cuda_lead)
print("Test loss (Best Validation): {0}".format(test_loss))

#Write output
ofile = None
try:
    ofile = open(arg_outdir + "/results.csv", mode = 'a')
except FileNotFoundError:
    os.makedirs(arg_outdir)
    ofile = open(arg_outdir + "/results.csv", mode = 'a')
    ofile.write("Seed, MSE, RMSE, MAE, sMAPE, Trained Model Filename\n")

scripted_model = torch.jit.script(best_model_dpt.module)
ofname = arg_model + str(param_dset_forecast) + "_" + arg_dset + "_" + str(seed) + ".smd"
scripted_model.save(arg_outdir + "/" + ofname)

ofile.write(str(seed) + ","
            + str(test_loss[0]) + ","
            + str(test_loss[1]) + ","
            + str(test_loss[2]) + ","
            + str(test_loss[3]) + ","
            + ofname + "\n")

ofile.close()