import unittest
import functools
import numpy as np

from log_uniform import LogUniformSampler

def EXPECT_NEAR(x, y, epsilon):
  return abs(x - y) <= epsilon

class LogUniformSamplerTestCase(unittest.TestCase):
    def test_probability(self):
      N = 1000000
      sampler = LogUniformSampler(N)
      idx = 100
      while idx < N:
        ratio = sampler.probability(idx) / sampler.probability(idx / 2)
        EXPECT_NEAR(ratio, 0.5, 0.1);
        idx *= 2

    def test_unique(self):
      N = 100
      nsampled = 50
      RND = 100

      sampler = LogUniformSampler(N)
      histogram = [0 for idx in range(N)]

      all_values = [idx for idx in range(N)]
      sample_ids, expected, sample_freq = sampler.sample(nsampled, np.asarray(all_values, dtype=np.int32))

      sample_set = set(sample_ids)
      self.assertEqual(len(sample_set), nsampled)

      for rnd in range(RND):
        sample_ids, true_freq, sample_freq = sampler.sample(nsampled, np.asarray(all_values, dtype=np.int32))

        for idx in range(N): 
          self.assertTrue(EXPECT_NEAR(expected[idx], true_freq[idx], expected[idx] * 0.5))

        for idx in range(nsampled):
          histogram[sample_ids[idx]] += 1

      for idx in range(N):
        average_count = histogram[idx] / RND
        self.assertTrue(EXPECT_NEAR(expected[idx], average_count, 0.2))

    def test_avoid(self):
      N = 100
      nsampled = 98
      sampler = LogUniformSampler(N)
      labels = [17, 23]
      sample_ids = sampler.sample_unique(nsampled, np.asarray(labels, dtype=np.int32))

      total = functools.reduce(lambda x,y: x+y, sample_ids, 0.0)
      expected_sum = 100 * 99 / 2 - labels[0] - labels[1]
      self.assertEqual(expected_sum, total)

if __name__ == '__main__':
    unittest.main()
