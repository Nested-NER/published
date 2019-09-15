#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: Transformer.py 
@time: 2019/04/13

@note:
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math
from torch.autograd import Variable


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def sequent_mask(mask, reverse=False):
    return torch.cat([_.triu().unsqueeze(0) for _ in mask]) if reverse \
        else torch.cat([_.tril().unsqueeze(0) for _ in mask])


def attention(query, key, value, mask=None, dropout=None):
    """
    An attention function can be described as mapping a query and a set of key-value pairs
    to an output. The output is computed as a weighted sum of the values, where the weight
    assigned to each value is computed by a compatibility function of the query with the
    corresponding key.
    Note for the dimension of the query and key vectors are equal.
    The two most commonly used attention functions(are similar in theoretical complexity):
        additive attention : using a feed-forward network with a single hidden layer
        dot-product(multiplicative) attention: much faster and more space-efficient in practice
                                               since it can be implemented using highly
                                               optimized matrix multiplication code
    Return Attention(Q, K, V)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Transformer(nn.Module):
    def __init__(self, N, d_model, h, dropout, bidirectional):
        super(Transformer, self).__init__()
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, dropout=dropout)
        self.bidirectional = bidirectional
        self.model = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
        if self.bidirectional:
            self.model = clones(self.model, 2)

    def forward(self, word_embed, mask=None):
        if self.bidirectional:
            return self.model[0](word_embed, sequent_mask(mask)) + \
                   self.model[1](word_embed, sequent_mask(mask, True))
        else:
            return self.model(word_embed, mask)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        :return LayerNorm（x + Sublayer(x))

        """
        return x + self.dropout(sublayer(self.norm(x)))
        # return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    Each layer has two sub-layers:
        multi-head self-attention mechanism : all of the keys, values, queries come from
                                              the output of the previous layer in the encoder
        feed-forward network (position-wise fully connection)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, head_2, ... head_n) * Wo
        where head_i = Attention(Q * Wq_i, K * Wk_i, V * Wv_i)
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: num of heads(parallel attention layers)
        :param d_model: dimension of input(embedding)

        :parameter linears:
                        Wq_i [d_model, d_k] * h
                        Wk_i [d_model, d_k] * h
                        Wv_i [d_model, d_v] * h
                        Wo [d_v * h, d_model]

        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous(): after transpose() before view()
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Applied to each position separately and identically(use different parameters)

    FFN(x) = max(0, w1x + b1)w2 + b2
    """

    def __init__(self, d_model, argument=2, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * argument)
        self.w_2 = nn.Linear(d_model * argument, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

