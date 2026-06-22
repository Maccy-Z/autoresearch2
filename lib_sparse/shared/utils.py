import math
import torch

# Constant for RELU^2 scaling
RELU2_SCALE = 3 ** -0.5

def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda")/1024**2
    print(f'{msg}: {memory:.2f} MB')


