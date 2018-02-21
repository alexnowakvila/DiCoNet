#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import matplotlib
matplotlib.use('Agg')

# Pytorch requirements
import unicodedata
import string
import re
import random
import pdb

import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


###############################################################################
#                                PTR-NET                                      #
###############################################################################


class PtrNet_tanh(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(PtrNet_tanh, self).__init__()
        print('Initializing Parameters Merge')
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.n = 12
        self.init_var = 0.08
        self.init_token = nn.Parameter(-1 * torch.ones((self.input_size)))
        self.pad_token = nn.Parameter(-1 * torch.ones((self.input_size)))
        self.end_state = nn.Parameter(-1 * torch.ones((self.hidden_size)))
        self.W1 = nn.Parameter(torch.randn((self.hidden_size,
                               self.hidden_size)) * self.init_var)
        self.W2 = nn.Parameter(torch.randn((self.hidden_size,
                               self.hidden_size)) * self.init_var)
        self.v = nn.Parameter(torch.randn((self.hidden_size, 1)) *
                              self.init_var)
        # cells
        self.encoder_cell = nn.GRUCell(input_size, hidden_size)
        self.decoder_cell = nn.GRUCell(input_size, hidden_size)
        self.NLLoss = nn.NLLLoss(size_average=True)
        # initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell) or isinstance(m, nn.GRUCell):
                m.weight_ih.data.uniform_(-self.init_var, self.init_var)
                m.weight_hh.data.uniform_(-self.init_var, self.init_var)
                m.bias_ih.data.uniform_(-self.init_var, self.init_var)
                m.bias_hh.data.uniform_(-self.init_var, self.init_var)
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, self.init_var)
                m.weight.data.uniform_(-self.init_var, self.init_var)

    def softmax_m(self, phis_m, u):
        mask = phis_m
        # masked softmax
        u_m = u
        u_m = u_m * mask
        maxims = torch.max(u_m, 1)[0]
        maxims = (maxims.squeeze().unsqueeze(1).expand(self.batch_size,
                  self.n + 1))
        exps = torch.exp(u_m - maxims)
        exps_m = exps
        exps_m = exps_m * mask
        exps_sum = (torch.sum(exps_m, 1).squeeze().unsqueeze(1)
                    .expand(self.batch_size, self.n + 1))
        return exps_m / exps_sum

    def Encoder(self, input, phis):
        hidden_encoder = (Variable(torch.zeros(self.n + 1, self.batch_size,
                          self.hidden_size),
                          requires_grad=True).type(dtype))
        hidden_init = (self.end_state.unsqueeze(0).expand(
                       self.batch_size, self.hidden_size))
        hidden_encoder[0] = hidden_init
        hidden = hidden_init
        for n in range(self.n):
            input_step = input[:, n + 1]
            # decouple interaction between different scopes using subdiagonal
            if n > 0:
                t = (phis[:, n, n - 1].squeeze().unsqueeze(1).expand(
                     self.batch_size, self.hidden_size))
                hidden = t * hidden + (1 - t) * hidden_init
            # apply cell
            hidden = self.encoder_cell(input_step, hidden)
            hidden_encoder[n + 1] = hidden
        hidden_encoder = hidden_encoder.permute(1, 0, 2)
        return hidden_encoder

    def attention(self, hidden, W1xe, hidden_encoder):
        # train
        W2xdn = torch.mm(hidden, self.W2)
        W2xdn = W2xdn.unsqueeze(1).expand(self.batch_size, self.n + 1,
                                          self.hidden_size)
        u = (torch.bmm(torch.tanh(W1xe + W2xdn), self.v.unsqueeze(0)
             .expand(self.batch_size, self.hidden_size, 1)))
        u = u.squeeze()
        # test
        # W2xdn = torch.mm(hidden, self.W2)
        # u = Variable(torch.zeros(self.batch_size, self.n + 1)).type(dtype)
        # for n in range(self.n + 1):
        #     aux = torch.tanh(W1xe[:, n].squeeze() + W2xdn)  # size bs x hidd
        #     aux2 = (torch.bmm(aux.unsqueeze(1), self.v.unsqueeze(0)
        #             .expand(self.batch_size, self.hidden_size, 1)))
        #     u[:, n] = aux2.squeeze()
        return u

    def policy_loss(self, logsoftmax, target_col, logprobs):
        pg_logsoftmax = (sum([logp.unsqueeze(1).expand_as(logsoftmax) 
                             * logsoftmax for logp in logprobs]))
        pg_logsoftmax /= float(len(logprobs))
        mask = (Variable(target_col.data > 0).type(dtype).unsqueeze(1)
                .expand_as(pg_logsoftmax))
        pg_logsoftmax = pg_logsoftmax * mask
        pg_loss_step = self.NLLoss(pg_logsoftmax, target_col.type(dtype_l))
        return pg_loss_step

    def compute_loss(self, output, target, lp=None):
        loss = 0.0
        pg_loss = 0.0
        for n in range(target.size()[1]):
            attn = output[:, n + 1] + 1e-6
            logsoftmax = torch.log(attn)
            if lp is not None and len(lp) > 0:
                pg_loss_step = self.policy_loss(logsoftmax, target[:, n], lp)
                pg_loss += pg_loss_step
            loss_step = self.NLLoss(logsoftmax, target[:, n].type(dtype_l))
            loss += loss_step
        return loss, pg_loss

    def Decoder(self, input, hidden_encoder, phis,
                input_target=None, target=None):
        feed_target = False
        if target is not None:
            feed_target = True
        # N[:, n] is the number of elements of the scope of the n-th element
        N = phis.sum(2).squeeze().unsqueeze(2).expand(self.batch_size, self.n,
                                                      self.hidden_size)
        output = (Variable(torch.ones(self.batch_size, self.n, self.n + 1))
                  .type(dtype))
        index = ((N[:, 0] - 1) % (self.n)).type(dtype_l).unsqueeze(1).detach()
        hidden = (torch.gather(hidden_encoder, 1, index + 1)).squeeze()
        # W1xe size: (batch_size, n + 1, hidden_size)
        W1xe = (torch.bmm(hidden_encoder, self.W1.unsqueeze(0).expand(
                self.batch_size, self.hidden_size, self.hidden_size)))
        # init token
        start = (self.init_token.unsqueeze(0).expand(self.batch_size,
                 self.input_size))
        input_step = start
        for n in range(self.n):
            # decouple interaction between different scopes by looking at
            # subdiagonal elements of Phi
            if n > 0:
                t = (phis[:, n, n - 1].squeeze().unsqueeze(1).expand(
                     self.batch_size, self.hidden_size))
                index = (((N[:, n] + n - 1) % (self.n)).type(dtype_l)
                         .unsqueeze(1)).detach()
                init_hidden = (torch.gather(hidden_encoder, 1, index + 1)
                               .squeeze())
                hidden = t * hidden + (1 - t) * init_hidden
                t = (phis[:, n, n - 1].squeeze().unsqueeze(1).expand(
                     self.batch_size, self.input_size))
                input_step = t * input_step + (1 - t) * start
            # Compute next state
            hidden = self.decoder_cell(input_step, hidden)
            # Compute pairwise interactions
            u = self.attention(hidden, W1xe, hidden_encoder)
            # Normalize interactions by taking the masked softmax by phi
            pad = Variable(torch.ones(self.batch_size, 1)).type(dtype)
            mask = torch.cat((pad, phis[:, n].squeeze()), 1)
            attn = self.softmax_m(mask, u)
            if feed_target:
                # feed next step with target
                next = (target[:, n].unsqueeze(1).unsqueeze(2)
                        .expand(self.batch_size, 1, self.input_size)
                        .type(dtype_l))
                input_step = torch.gather(input_target, 1, next).squeeze()
            else:
                # not blend
                index = attn.max(1)[1].squeeze()
                next = (index.unsqueeze(1).unsqueeze(2)
                        .expand(self.batch_size, 1, self.input_size)
                        .type(dtype_l))
                input_step = torch.gather(input, 1, next).squeeze()
                # blend inputs
                # input_step = (torch.sum(attn.unsqueeze(2).expand(
                #               self.batch_size, self. n + 1,
                #               self.input_size) * input, 1)).squeeze()
            # Update output
            output[:, n] = attn
        return output

    def forward(self, input, phis, input_target=None, target=None):
        # Encoder
        self.batch_size = input.size()[0]
        hidden_encoder = self.Encoder(input, phis)
        # Pointer Decoder
        output = self.Decoder(input, hidden_encoder, phis,
                              input_target=input_target, target=target)
        return output
