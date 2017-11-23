import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import util

from log_uniform import LogUniformSampler

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            util.initialize(self.params, self.ntokens)
        self.params.bias.data.fill_(0)

    def forward(self, inputs, labels, train=False):
        if train:
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values):
        assert(inputs.data.get_device() == labels.data.get_device())
        device_id = labels.data.get_device()

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        # sample ids according to word distribution - Unique
        sample_ids = Variable(torch.LongTensor(sample_ids), requires_grad=False).cuda(device_id)
        true_freq = Variable(torch.FloatTensor(true_freq), requires_grad=False).cuda(device_id)
        sample_freq = Variable(torch.FloatTensor(sample_freq), requires_grad=False).cuda(device_id)

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels.data, :]
        true_bias = self.params.bias[labels.data]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids.data, :]
        sample_bias = self.params.bias[sample_ids.data]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long(), requires_grad=False).cuda(device_id)
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels

class RNNModel(nn.Module):
    """A recurrent module"""

    def __init__(self, ntokens, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        # Parameters
        self.nhid = nhid
        self.nlayers = nlayers

        # Create Layers
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

    def forward(self, inputs, hidden):
        inputs = self.drop(inputs)
        output, hidden = self.rnn(inputs, hidden)
        output = self.drop(output)
        return output.view(output.size(0)*output.size(1), output.size(2)), hidden

    def init_hidden(self, bsz):
        return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)),
               Variable(torch.zeros(self.nlayers, bsz, self.nhid)))
