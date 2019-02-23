import math
import sys

import torch
import numpy as np

def initialize(matrix):
    in_, out_ = matrix.size()
    stdv = math.sqrt(3. / (in_ + out_))
    matrix.data.uniform_(-stdv, stdv)

def log_uniform(class_id, range_max):
    return (math.log(class_id+2) - math.log(class_id+1)) / math.log(range_max+1)

def log_uniform_distribution(range_max, N):
    distribution = np.asarray([log_uniform(idx, range_max) for idx in range(range_max)])
    freq = N * distribution
    return torch.from_numpy(distribution), torch.from_numpy(freq)

def log_uniform_sample(N, size):
    log_N = math.log(N)
    x = torch.Tensor(size).uniform_(0, 1)
    value = torch.exp(x * log_N).long() - 1
    return torch.remainder(value, N)

def fixed_unigram_distribution(word_freq, N, unique=False):
    total = torch.sum(word_freq)
    distribution = word_freq.float() / float(total)
    if unique:
        freq = 1.0 - np.exp(N * torch.log(1.0 - distribution))
    else:
        freq = N * distribution
    return distribution, freq

def reverse(item):
    new_item = np.zeros(len(item))
    for idx, val in enumerate(item):
        new_item[val] = idx
    return new_item

def load_np(filepath):
    npzfile = np.load(filepath)
    return torch.from_numpy(npzfile['item'])
