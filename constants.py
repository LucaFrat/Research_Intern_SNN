import tonic
import torch
from snntorch import surrogate

SEED = 20
SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size
SUBSET = 20
FILTER_TIME = 10000
TIME_WINDOW = 1000
ROTATION = 10
BATCH_SIZE = 128
NUM_STEPS = 25
BETA = 0.2

EPOCHS = 20
DTYPE = torch.float
NUM_ITERS = 5
LR = 4e-2
CORRECT_RATE = 0.8
BETAS_ADAM = [0.9, 0.99]
SURR_NAMES = ['Atan', 'Sigmoid']
SURR_FUNCTIONS = [surrogate.atan(alpha=3), surrogate.sigmoid(slope=20)] # surrogate.triangular()


class NMNIST_Net():
    CHANNELS = [2, 12, 32]
    KERNELS = [5, 5]
    RES_DIM = 5
    CLASSES = 10 

class FashionMNIST_Net():
    CHANNELS = [1, 3, 6]
    KERNELS = [5, 3]
    RES_DIM = 5
    CLASSES = 10 
