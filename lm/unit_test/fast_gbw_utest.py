import unittest
import torch
import numpy as np
from fast_gbw import FastGBWDataset

class FastGBWDataset():
    """Preprocessed Google 1-Billion Word dataset."""

    def __init__(self, corpus, sid):
        """
        Args:
            path (string): path to data file
            name (string): file name
            sid (string): file name - sentence id
        """
        self.corpus = corpus
        self.sentence_id = sid
        self.num_words = self.corpus.shape[0]
        self.length, dim = self.sentence_id.shape

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

        tracker_list = [(-1, 0) for seq_idx in range(batch_size)]
        self.pos = 0

        self.stop_condition = False
        while not self.stop_condition:
            source.fill_(0)
            target.fill_(0)
            for idx, tracker in enumerate(tracker_list):
                # add sequence to new minibatch
                tracker_list[idx] = self.add(seq_length, source, target, idx, tracker)
            yield (source, target)

    def add(self, seq_length, source, target, batch_idx, tracker):
        seq_id, seq_pos = tracker
        start_idx, end_idx, length = self.sentence_id[seq_id]

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
               seq_pos = 0
               start_idx, end_idx, length = self.sentence_id[seq_id]

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

class DataUtilsTestCase(unittest.TestCase):

    def test_dataset(self):
        # sentences
        def generator():
            for i in range(1, 10):
                yield [0] + list(range(1, i + 1)) + [0]

        SIZE = 1000
        counts = [0 for idx in range(SIZE)]

        corpus = list()
        sid = list()
        start_id = 0
        for seq in generator():
            for v in seq:
                corpus.append(v)
                counts[v] += 1
            sid.append([start_id, start_id+len(seq), len(seq)])
            start_id += len(seq)

        dataset = FastGBWDataset(torch.from_numpy(np.asarray(corpus)).long(), np.asarray(sid))
        counts2 = [0 for idx in range(SIZE)]
        for x, y in dataset.batch_generator(4, 2):
            for v in x.numpy().ravel():
                counts2[v] += 1

        for i in range(1, SIZE):
            self.assertEqual(counts[i], counts2[i], "Mismatch at i=%d. counts[i]=%s, counts2[i]=%s" % (i,counts[i], counts2[i]))

if __name__ == '__main__':
    unittest.main()
