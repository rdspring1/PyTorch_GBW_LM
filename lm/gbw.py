import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GBWDataset(Dataset):
    """Google 1-Billion Word dataset."""

    def __init__(self, path, name, mapto):
        """
        Args:
            path (string): path to data file
            name (string): file name
        """
        self.corpus, self.num_words = self.build(os.path.join(path, name), mapto)
        self.length = len(self.corpus)
        print("#sentences", self.length)

    def build(self, tensor_path, mapto):
        """ Convert data (sentence id, word id) into a list of sentences """
        tensor = torch.load(tensor_path).long()
        num_words = tensor.shape[0]

        data = list()
        current_sentence_id = -1 
        start_idx = -1
        for idx, value in enumerate(tensor):
            sentence_id, word_id = value
            if current_sentence_id != sentence_id:
                if start_idx > 0:
                    data.append(mapto[tensor[start_idx:idx, 1]-1])
                current_sentence_id = sentence_id
                start_idx = idx
        return data, num_words

    def batch_generator(self, seq_length, batch_size, shuffle=False):
        """Generates a random batch for training or validation

        Structures each element of the batch as an 'episode'.
        Each episode contains episode_length examples and
        episode_width distinct labels.

        Args:
          seq_length: length of sequence
          batch_size: number of examples
          N: number of rounds
          evaluation: True if testing otherwise False

        Returns:
          A tuple (x, y) where x is the input tensor (batch_size, sequence_length) 
          and y is the output tensor (batch_size, sequence_length)
        """
        wrd_cnt = batch_size * seq_length
        batch_len = self.num_words // batch_size
        self.batch_num = (batch_len - 1) // seq_length

        if self.batch_num == 0:
            raise ValueError("batch_num == 0, decrease batch_size or seq_length")

        source = torch.zeros(seq_length, batch_size).long()
        target = torch.zeros(seq_length, batch_size).long()

        if shuffle:
            # sample random set of initial positions for each example in batch
            self.ordering = np.random.permutation(self.length)
        else:
            # deterministic ordering
            self.ordering = np.arange(self.length)

        # track current sequence and position
        tracker_list = [(-1, 0) for seq_idx in range(batch_size)]
        self.pos = 0

        self.stop_condition = False
        while not self.stop_condition:
            source.fill_(0)
            target.fill_(0)
            for idx, tracker in enumerate(tracker_list):
                # add sequence to new minibatch
                tracker_list[idx] = self.add(seq_length, source, target, idx, tracker)
            yield (source, target, wrd_cnt, self.batch_num)

    def add(self, seq_length, source, target, batch_idx, tracker):
        seq_id, seq_pos = tracker

        if seq_id != -1:
            sequence = self.corpus[seq_id]

        curr = 0 
        while curr != seq_length:
            # Initialize Sequence Position
            if seq_id == -1 or seq_pos+1 == len(sequence):
               # Stop Condition
               if self.pos >= self.length:
                   self.stop_condition = True
                   return (self.length, 0)
               else:
                   seq_id = self.ordering[self.pos]
                   self.pos += 1
               seq_pos = 0
               sequence = self.corpus[seq_id]

            seq_remaining = len(sequence) - 1 - seq_pos
            batch_remaining = seq_length - curr
            size = min(seq_remaining, batch_remaining)
            batch_end = curr + size
            seq_end = seq_pos + size

            # target is offset from source by 1
            source[curr:batch_end, batch_idx] = sequence[seq_pos:seq_end]
            target[curr:batch_end, batch_idx] = sequence[seq_pos+1:seq_end+1]

            curr = batch_end
            seq_pos = seq_end
            assert(seq_pos < len(sequence))
        return (seq_id, seq_pos)
