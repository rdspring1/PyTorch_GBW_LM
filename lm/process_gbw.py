import os
import sys
import pickle

import numpy as np
import torch
from torch.utils.serialization import load_lua

def build(tensor_path):
    """ Convert data (sentence id, word id) into a list of sentences """
    tensor = load_lua(tensor_path).long()
    size = tensor.size()[0]

    data = np.empty((size, 3), dtype=np.int32)
    current_sentence_id = tensor[0, 0]
    start_idx = 0
    for idx, value in enumerate(tensor):
        if (idx % 100000) == 0:
            print(idx, size)

        sentence_id, word_id = value
        if current_sentence_id != sentence_id:
            length = idx - start_idx
            data[current_sentence_id, 0] = start_idx
            data[current_sentence_id, 1] = idx
            data[current_sentence_id, 2] = length

            start_idx = idx
            current_sentence_id = sentence_id
    return data

assert(len(sys.argv) == 3)
path = sys.argv[1]
filename = sys.argv[2]
print("Filepath:", path)
print("Output:", filename)

word_freq = load_lua(os.path.join(path, 'word_freq.th7')).numpy()
print("Loaded Tensor")

data = build(os.path.join(path, 'train_data.th7'))
print("Build Sentence ID Tensor")

with open(filename, 'wb') as f:
    np.savez(f, item=data)
print("Saved Sentence ID Tensor")
