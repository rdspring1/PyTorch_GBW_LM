import unittest
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

import model
from log_uniform import LogUniformSampler

def EXPECT_NEAR(x, y, epsilon):
  return np.all(abs(x - y) <= epsilon)

class ComputeSampledLogitsTest(unittest.TestCase):
  def _GenerateTestData(self, num_classes, dim, batch_size, num_true, labels, sampled, subtract_log_q):
    """Randomly generates input/output data for a single test case.
    This function returns numpy constants for use in a test case.
    Args:
      num_classes: An int. The number of embedding classes in the test case.
      dim: An int. The dimension of the embedding.
      batch_size: An int. The batch size.
      num_true: An int. The number of target classes per training example.
      labels: A list of batch_size * num_true ints. The target classes.
      sampled: A list of indices in [0, num_classes).
      subtract_log_q: A bool corresponding to the parameter in
          _compute_sampled_logits().
    Returns:
      weights: Embedding weights to use as test input. It is a numpy array
          of shape [num_classes, dim]
      biases: Embedding biases to use as test input. It is a numpy array
          of shape [num_classes].
      hidden_acts: Forward activations of the network to use as test input.
          It is a numpy array of shape [batch_size, dim].
      sampled_vals: A tuple based on `sampled` to use as test input in the
          format returned by a *_candidate_sampler function.
      exp_logits: The output logits expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
      exp_labels: The output labels expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
    """
    weights = np.random.randn(num_classes, dim).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    hidden_acts = np.random.randn(batch_size, dim).astype(np.float32)

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_vals = (torch.LongTensor(sampled), torch.from_numpy(np.squeeze(true_exp)), torch.from_numpy(sampled_exp))

    sampled_w, sampled_b = weights[sampled], biases[sampled]
    true_w, true_b = weights[labels], biases[labels]

    true_logits = np.sum(hidden_acts.reshape((batch_size, 1, dim)) * true_w.reshape((batch_size, num_true, dim)), axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if subtract_log_q:
      true_logits -= np.log(true_exp)
      sampled_logits -= np.log(sampled_exp[np.newaxis, :])

    exp_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    exp_labels = np.hstack((np.ones_like(true_logits) / num_true, np.zeros_like(sampled_logits)))

    return weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels

  def test_SampledSoftmaxLoss(self):
    # A simple test to verify the numerics.

    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits - np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    np.random.seed(1000)
    num_classes = 5
    batch_size = 3
    nsampled = 4
    nhid = 10
    labels = [0, 1, 2]

    (weights, biases, hidden_acts, sampled_values, exp_logits, exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=nhid,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
		 subtract_log_q=True)

    ss = model.SampledSoftmax(num_classes, nsampled, nhid, tied_weight=None)
    ss.params.weight.data = torch.from_numpy(weights)
    ss.params.bias.data = torch.from_numpy(biases)
    ss.params.cuda()

    hidden_acts = Variable(torch.from_numpy(hidden_acts)).cuda()
    labels = Variable(torch.LongTensor(labels)).cuda()

    logits, new_targets = ss.sampled(hidden_acts, labels, sampled_values)
    self.assertTrue(EXPECT_NEAR(exp_logits, logits.data.cpu().numpy(), 1e-4))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, nsampled+1), new_targets)
    expected_sampled_softmax_loss = np.mean(_SoftmaxCrossEntropyWithLogits(exp_logits, exp_labels))
    self.assertTrue(EXPECT_NEAR(expected_sampled_softmax_loss, loss.item(), 1e-4))

  def test_AccidentalMatch(self):
    np.random.seed(1000)
    num_classes = 5
    batch_size = 3
    nsampled = 4
    nhid = 10
    labels = np.random.randint(low=0, high=num_classes, size=batch_size)

    (weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels) = self._GenerateTestData(
         num_classes=num_classes,
         dim=nhid,
         batch_size=batch_size,
         num_true=1,
         labels=labels,
         sampled=[1, 0, 2, 3],
		 subtract_log_q=True)

    ss = model.SampledSoftmax(num_classes, nsampled, nhid, tied_weight=None)
    ss.params.weight.data = torch.from_numpy(weights)
    ss.params.bias.data = torch.from_numpy(biases)
    ss.params.cuda()

    hidden_acts = Variable(torch.from_numpy(hidden_acts)).cuda()
    labels = Variable(torch.LongTensor(labels)).cuda()

    sampler = LogUniformSampler(nsampled)
    sampled_values = sampler.sample(nsampled, labels.data.cpu().numpy())
    sample_ids, true_freq, sample_freq = sampled_values
    logits, new_targets = ss.sampled(hidden_acts, labels, sampled_values, remove_accidental_match=True)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, nsampled+1), new_targets)

    np_logits = logits.data.cpu().numpy()
    for row in range(batch_size):
      label = labels[row]
      for col in range(nsampled):
        if sample_ids[col] == label:
          self.assertTrue(EXPECT_NEAR(np.exp(np_logits[row, col+1]), 0, 1e-4))

if __name__ == '__main__':
    unittest.main()
