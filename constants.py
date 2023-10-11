import tonic
import torch
from snntorch import functional as SF

SEED = 20
SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size
FILTER_TIME = 10000
TIME_WINDOW = 1000
ROTATION = 10
BATCH_SIZE = 128
NUM_STEP = 25
BETA = 0.5

EPOCHS = 3
DTYPE = torch.float
NUM_ITERS = 5
LR = 2e-2
CORRECT_RATE = 0.8
BETAS_ADAM = [0.9, 0.999]


class NMNIST_Net():
    CHANNELS = [2, 12, 32]
    KERNELS = [5, 5]
    RES_DIM = 5
    CLASSES = 10 # DVS-Gesture has 11 classes

class FashionMNIST_Net():
    CHANNELS = [1, 4, 8]
    KERNELS = [5, 3]
    RES_DIM = 5
    CLASSES = 10 

class DVS_Net():
    CHANNELS = [0, 0, 0]
    KERNELS = [0, 0]
    ResDim = 0
    CLASSES = 11 # DVS-Gesture has 11 classes

