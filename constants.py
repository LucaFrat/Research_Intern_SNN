import tonic
import torch
from snntorch import functional as SF

SEED = 20
SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size
FILTER_TIME = 10000
TIME_WINDOW = 1000
ROTATION = 10
BATCH_SIZE = 128
BETA = 0.5

class Net:
    CHANNELS = [2, 12, 32]
    KERNELS = [5, 5]
    CLASSES = 10 # DVS-Gesture has 11 classes

EPOCHS = 1
NUM_ITERS = 5
LR = 2e-2
CORRECT_RATE = 0.8
BETAS_ADAM = [0.9, 0.999]