import tonic
import torch
from snntorch import functional as SF

sensor_size = tonic.datasets.NMNIST.sensor_size
filter_time = 10000
time_window = 1000
rRotation = 10
batch_size = 128
beta = 0.5

class Net:
    channels = [2, 12, 32]
    kernels = [5, 5]
    classes = 10

epochs = 1
num_iters = 5
lr = 2e-2
correct_rate = 0.8
betas_Adam = [0.9, 0.999]