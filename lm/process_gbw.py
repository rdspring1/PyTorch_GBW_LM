import os
import sys
import pickle

import numpy as np
import torch

def build(tensor_path):
    """ Convert data (sentence id, word id) into a list of sentences """
    tensor = torch.load(tensor_path).long()
    num_words = tensor.size()[0]

    print("Processing words to find sentences")
    sentences = dict()
    current_sentence_id = tensor[0, 0]
    start_idx = 0
    for idx, value in enumerate(tensor):
        if (idx % 100000) == 0:
            print(idx, num_words)

        sentence_id, word_id = value
        if current_sentence_id != sentence_id:
            length = idx - start_idx
            sentences[current_sentence_id] = (start_idx, length)

            start_idx = idx
            current_sentence_id = sentence_id

    print("Processing sentences - Building SID Tensor")
    num_sentences = len(sentences)
    data = np.empty((num_sentences, 2), dtype=np.int32)
    for idx, item in enumerate(sentences.items()):
        if (idx % 100000) == 0:
            print(idx, num_sentences)

        key, value = item
        start_idx, length = value
        data[idx, 0] = start_idx
        data[idx, 1] = length
    return data

assert(len(sys.argv) == 3)
path = sys.argv[1]
filename = sys.argv[2]
print("Filepath:", path)
print("Output:", filename)

word_freq = torch.load(os.path.join(path, 'word_freq.pt')).numpy()
print("Loaded Tensor")

data = build(os.path.join(path, 'train_data.pt'))
print("Build Sentence ID Tensor")

with open(filename, 'wb') as f:
    np.savez(f, item=data)
print("Saved Sentence ID Tensor")
