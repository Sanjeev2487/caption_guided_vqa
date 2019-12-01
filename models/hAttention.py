## Basecode for this hAttention Network is taken from the following Github Repo:
## https://github.com/SSinyu/Hierarchical-Attention-Networks

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class AttentionLayer(nn.Module):
    def __init__(self, num_hid=100, bi=True):
        super(AttentionLayer, self).__init__()
        self.num_hid = num_hid
        self.linear_ = nn.Linear(self.num_hid, self.num_hid)
        self.tanh_ = nn.Tanh()
        self.softmax_ = nn.Softmax(dim=1)

    def forward(self, x):
        print("x.shape: ", x.shape)
        u_context = torch.nn.Parameter(torch.FloatTensor(self.num_hid).normal_(0, 0.01)).cuda()
        h = self.tanh_(self.linear_(x)).cuda()
        sm = torch.mul(h, u_context)
        return sm, 0
        print("sm.shape: ", sm.shape)
        alpha = self.softmax_(sm.sum(dim=0, keepdim=True))  # (x_dim0, x_dim1, 1)
        print("alpha.shape: ", alpha.shape)
        attention_output = torch.mul(alpha, x).sum(dim=1)  # (x_dim0, x_dim2)
        print("attention_output.shape: ", attention_output.shape)
        return attention_output, alpha
        return attention_output, alpha


class HierarchicalAttentionNet(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        super(HierarchicalAttentionNet, self).__init__()
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.ndirections = 1 + int(bidirect)
        self.word_rnn = self.rnn(in_dim, num_hid, nlayers, dropout=dropout, bidirectional=bidirect, batch_first=True)
        self.word_att = AttentionLayer(num_hid, bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        print("x.shap: ", x.shape)
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.word_rnn.flatten_parameters()
        output, hidden = self.word_rnn(x, hidden)
        print("output.shape: ", output.shape)

        if self.ndirections == 1:
            print("output.shape: ", output.shape)
            print("output[:, -1].shape: ", output[:, -1].shape)
            attn, _ = self.word_att(output[:, -1])
            print("attn.shape: ", attn.shape)
            return attn

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        attn, _ = self.word_att((forward_, backward), dim=1)
        return attn

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.word_rnn.flatten_parameters()
        output, hidden = self.word_rnn(x, hidden)
        attn, _ = self.word_att(output)
        return attn
