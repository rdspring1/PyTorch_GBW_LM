import unittest
import torch
import numpy as np

class GBWDataset():
    """Google 1-Billion Word dataset."""

    def __init__(self, corpus, num_words):
        """
        Args:
            path (string): path to data file
            name (string): file name
            sid (string): file name - sentence id
        """
        self.corpus = corpus
        self.num_words = num_words 
        self.length = len(self.corpus)

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
            yield (source, target)

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

class DataUtilsTestCase(unittest.TestCase):

    def test_dataset(self):
        # sentences
        def generator():
            for i in range(1, 10):
                yield [0] + list(range(1, i + 1)) + [0]

        SIZE = 10
        counts = [0 for idx in range(SIZE)]

        corpus = list()
        for seq in generator():
            corpus.append(torch.from_numpy(np.asarray(seq, dtype=np.int64)))
            for v in seq:
                counts[v] += 1
        num_words = sum(counts)

        dataset = GBWDataset(corpus, num_words)
        counts2 = [0 for idx in range(SIZE)]
        for x, y in dataset.batch_generator(4, 2):
            print(x)
            for v in x.numpy().ravel():
                counts2[v] += 1

        for i in range(1, SIZE):
            self.assertEqual(counts[i], counts2[i], "Mismatch at i=%d. counts[i]=%s, counts2[i]=%s" % (i,counts[i], counts2[i]))

if __name__ == '__main__':
    unittest.main()
