import tonic
import torch
from snntorch import surrogate

BETA = 0.93
EPOCHS = 2
LR = 2e-2

SURR_NAMES = ['Atan', 'Sigmoid', 'FastSigm']
SURR_SLOPES = {'Atan': [1, 4, 7], 'Sigmoid': [4, 6, 8], 'FastSigm': [0.5, 0.7, 0.9]}

def get_surrogate_function(index: int, coeff):
    return [surrogate.atan(alpha=coeff),\
            surrogate.sigmoid(slope=coeff),\
            surrogate.fast_sigmoid(slope=coeff)][index]

# SURR_FUNCTIONS = [surrogate.atan(alpha=), surrogate.sigmoid(slope=25), surrogate.fast_sigmoid(slope=25)]

SEED = 20
# divide the size of the original dataset by SUBSET
SUBSET = 20 
SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size
FILTER_TIME = 10000
TIME_WINDOW = 1000
ROTATION = 10
BATCH_SIZE = 128
NUM_STEPS = 25
DTYPE = torch.float
NUM_ITERS = 5
CORRECT_RATE = 0.8
BETAS_ADAM = [0.9, 0.99]

class FashionMNIST_Net():
    CHANNELS = [1, 3, 6]
    KERNELS = [5, 3]
    RES_DIM = 5
    CLASSES = 10 
