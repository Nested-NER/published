#!/usr/bin/env python
#-*- coding:utf-8 _*-
"""
@file: model.py
@time: 2019/05/19
"""


import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from Transformer import Transformer


class TOI_Pooling(nn.Module):
    def __init__(self, input_size, if_gpu):
        super(TOI_Pooling, self).__init__()
        self.input_size = input_size
        padding = torch.zeros((input_size, 1))
        self.device = torch.device("cuda:0" if if_gpu else "cpu")
        self.register_buffer('padding', padding)

    def forward(self, features, tois):
        result = []
        for i in range(len(tois)):
            result.append(self.HAT_TOI_Pooling(features[i], tois[i]))

        return torch.cat(result, 0), np.cumsum([len(s) for s in tois]).astype(np.int32)

    def HAT_TOI_Pooling(self, feature, tois):
        start = tois[:, 0]
        end = tois[:, 1]
        cumsum = torch.cat([Variable(self.padding, requires_grad=False), torch.cumsum(feature, 1)], dim=1)
        return torch.cat([feature[:, start], self.WeightedAvg(cumsum, start, end), feature[:, end - 1]]).t()

    def WeightedAvg(self, cumsum, start, end):
        boundary_len = Variable(torch.FloatTensor(end - start), requires_grad=False).to(self.device)
        return (cumsum[:, end] - cumsum[:, start]) / boundary_len


class CharEmbed(nn.Module):
    def __init__(self, char_kinds, embedding_size):
        super(CharEmbed, self).__init__()
        self.char_embed = nn.Embedding(char_kinds, embedding_size)
        self.char_bilstm = nn.LSTM(embedding_size, embedding_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, char_id, char_len):
        char_vec = self.char_embed(char_id)
        char_vec = torch.cat(tuple(char_vec))
        chars_len = np.concatenate(char_len)
        perm_idx = chars_len.argsort(0)[::-1].astype(np.int32)
        back_idx = perm_idx.argsort().astype(np.int32)
        pack = pack_padded_sequence(char_vec[perm_idx], chars_len[perm_idx], batch_first=True)
        lstm_out, (hid_states, cell_states) = self.char_bilstm(pack)
        return torch.cat(tuple(hid_states), 1)[back_idx]


class TOICNN(nn.Module):
    def __init__(self, config):
        super(TOICNN, self).__init__()
        self.config = config
        self.outfile = None

        self.input_size = config.word_embedding_size
        if self.config.if_pos:
            self.pos_embed = nn.Embedding(self.config.pos_tag_kinds, self.config.pos_embedding_size)
            self.input_size += self.config.pos_embedding_size
        if self.config.if_char:
            self.char_embed = CharEmbed(self.config.char_kinds, self.config.char_embedding_size)
            self.input_size += (2 * self.config.char_embedding_size)
        self.word_embed = nn.Embedding(self.config.word_kinds, self.config.word_embedding_size)
        self.input_dropout = nn.Dropout(self.config.dropout)

        if self.config.if_transformer:
            self.transformer = Transformer(d_model=self.input_size, N=self.config.N, h=self.config.h,
                                           dropout=0.1, bidirectional=self.config.if_bidirectional)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.config.feature_maps_size,
                      kernel_size=(self.config.kernel_size, self.input_size),
                      padding=(self.config.kernel_size // 2, 0)),
            nn.ReLU())

        self.hat = TOI_Pooling(self.config.feature_maps_size, self.config.if_gpu)
        self.flatten_feature = self.config.feature_maps_size * 3

        self.cls_fcs = nn.Sequential(
            nn.Linear(self.flatten_feature, self.flatten_feature),
            nn.Dropout(self.config.dropout), nn.ReLU(),
            nn.Linear(self.flatten_feature, self.config.label_kinds))

        self.cls_ce_loss = nn.CrossEntropyLoss()

    def forward(self, mask_batch, word_batch, char_batch, char_len_batch, pos_batch, toi_batch):
        word_vec = self.word_embed(word_batch)
        if self.config.if_char:
            char_vec = self.char_embed(char_batch, char_len_batch)
            word_vec = torch.cat([word_vec, char_vec.view(word_vec.shape[0], word_vec.shape[1], -1)], 2)
        if self.config.if_pos:
            pos_vec = self.pos_embed(pos_batch)
            word_vec = torch.cat([word_vec, pos_vec], 2)
        word_vec = self.input_dropout(word_vec)

        if self.config.if_transformer:
            word_vec = self.transformer(word_vec, mask_batch)

        features = self.cnn(word_vec.unsqueeze(1))
        features, toi_section = self.hat(features.squeeze(-1), toi_batch)

        return self.cls_fcs(features), toi_section

    def load_vector(self):
        with open(self.config.get_pkl_path("word2vec"), "rb") as f:
            vectors = pickle.load(f)
            w_v = torch.Tensor(vectors)
            print(f"Loading from {self.config.get_pkl_path('word2vec')}")
            self.word_embed.weight = nn.Parameter(w_v)
            if self.config.if_freeze:
                self.word_embed.weight.requires_grad = False

    def calc_loss(self, cls_s, gold_label):
        return self.cls_ce_loss(cls_s, gold_label)
