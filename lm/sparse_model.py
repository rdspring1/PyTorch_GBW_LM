import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from log_uniform import LogUniformSampler
import util

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        '''
        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            util.initialize(self.params.weight)
        '''

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        assert(inputs.data.get_device() == labels.data.get_device())
        device_id = labels.data.get_device()

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids)).cuda(device_id)
        true_freq = Variable(torch.FloatTensor(true_freq)).cuda(device_id)
        sample_freq = Variable(torch.FloatTensor(sample_freq)).cuda(device_id)

        # gather true labels - weights and frequencies
        true_weights = F.embedding(labels, self.params.weight, sparse=True)
        true_bias = torch.index_select(self.params.bias, 0, labels)

        # gather sample ids - weights and frequencies
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = torch.index_select(self.params.bias, 0, sample_ids)

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = sample_weights.matmul(inputs.t()).t() + sample_bias

        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long()).cuda(device_id)
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels

class RNNModel(nn.Module):
    """A recurrent module"""

    def __init__(self, ntokens, ninp, nhid, nout, nlayers, proj, dropout):
        super(RNNModel, self).__init__()
        # Parameters
        self.nhid = nhid
        self.nlayers = nlayers

        # Create Layers
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

        if proj:
            self.proj = nn.Linear(nhid, nout)
            util.initialize(self.proj.weight)
        else:
            self.proj = None

    def forward(self, inputs, hidden):
        inputs = self.drop(inputs)
        output, hidden = self.rnn(inputs, hidden)

        if self.proj is not None:
           output = self.proj(output)

        output = self.drop(output)
        return output.view(output.size(0)*output.size(1), output.size(2)), hidden

    def init_hidden(self, bsz):
        return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)),
               Variable(torch.zeros(self.nlayers, bsz, self.nhid)))
