import os
import random
import util
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class FastGBWDataset(Dataset):
    """Preprocessed Google 1-Billion Word dataset."""

    def __init__(self, path, name, sid, mapto, seq_length, batch_size):
        """
        Args:
            path (string): path to data file
            name (string): file name
            sid (string): file name - sentence id
            seq_length: length of sequence
            batch_size: number of examples
        """
        dataset = torch.load(os.path.join(path, name))
        self.corpus = mapto[dataset[:, 1]-1]
        print("loaded tensor", self.corpus.size())

        self.sentence_id = util.load_np(os.path.join(path, sid))
        print("loaded tensor", self.sentence_id.size())

        self.num_words = self.corpus.shape[0]
        self.length, dim = self.sentence_id.shape
        print("#sentences", self.length)

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.wrd_cnt = self.batch_size * self.seq_length
        batch_len = self.num_words // self.batch_size
        self.batch_num = (batch_len - 1) // self.seq_length

    def build(filepath):
        tensor = torch.load(filepath)
        vocab = dict()
        for idx, value in enumerate(tensor):
            vocab[value] = idx
        start_id = vocab["<S>"]
        end_id = vocab["</S>"]
        unk_id = vocab["<UNK>"]
        return vocab, start_id, end_id, unk_id

    def batch_generator(self, shuffle=False):
        """Generates a random batch for training or validation

        Structures each element of the batch as an 'episode'.
        Each episode contains episode_length examples and
        episode_width distinct labels.

        Args:
          N: number of rounds
          evaluation: True if testing otherwise False

        Returns:
          A tuple (x, y) where x is the input tensor (batch_size, sequence_length) 
          and y is the output tensor (batch_size, sequence_length)
        """
        if self.batch_num == 0:
            raise ValueError("batch_num == 0, decrease batch_size or seq_length")

        source = torch.zeros(self.seq_length, self.batch_size).long()
        target = torch.zeros(self.seq_length, self.batch_size).long()

        if shuffle:
            # sample random set of initial positions for each example in batch
            self.ordering = np.random.permutation(self.length)
        else:
            # deterministic ordering
            self.ordering = np.arange(self.length)

        tracker_list = [(self.ordering[seq_idx], 0) for seq_idx in range(self.batch_size)]
        self.pos = self.batch_size

        # track current sequence and position
        #initial = random.sample(range(self.length), batch_size)
        #tracker_list = [(seq_idx, 0) for seq_idx in initial]

        #for seq_idx in range(self.batch_num):
        self.stop_condition = False
        while not self.stop_condition:
            source.fill_(0)
            target.fill_(0)
            for idx, tracker in enumerate(tracker_list):
                # add sequence to new minibatch
                tracker_list[idx] = self.add(self.seq_length, source, target, idx, tracker)
            yield (source, target, self.wrd_cnt, self.batch_num)

    def add(self, seq_length, source, target, batch_idx, tracker):
        seq_id, seq_pos = tracker
        start_idx, length = self.sentence_id[seq_id]

        # Bounds-Check
        if seq_id >= 0:
            start_idx, length = self.sentence_id[seq_id]

        curr = 0 
        while curr != seq_length:
            # Initialize Sequence Position
            if seq_id == -1 or seq_pos+1 == length:
               # Stop Condition
               if self.pos >= self.length:
                   self.stop_condition = True
                   return (self.length, 0)
               else:
                   seq_id = self.ordering[self.pos]
                   self.pos += 1
               #seq_id = 0 if (seq_id+1) == self.length else (seq_id+1)
               seq_pos = 0
               start_idx, length = self.sentence_id[seq_id]

            seq_remaining = length - 1 - seq_pos
            batch_remaining = seq_length - curr
            size = min(seq_remaining, batch_remaining)
            batch_end = curr + size

            seq_start = start_idx + seq_pos
            seq_end = seq_start + size

            # target is offset from source by 1
            source[curr:batch_end, batch_idx] = self.corpus[seq_start:seq_end]
            target[curr:batch_end, batch_idx] = self.corpus[seq_start+1:seq_end+1]

            curr = batch_end
            seq_pos += size
            assert(seq_pos < length)
        return (seq_id, seq_pos)
