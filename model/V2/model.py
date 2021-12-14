#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/13 20:39
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, feat_mat):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size=hidden_size, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )
        self.feat_mat = torch.FloatTensor(feat_mat)
        self.mat_linear = nn.Linear(feat_mat.shape[1], hidden_size)
        self.lstm_linear = nn.Linear(hidden_size * 128 * 2, hidden_size)
        self.mix_linear = nn.Linear(hidden_size * 2, 300)

    def forward(self, input_ids, ins_id):
        feat_mat = self.mat_linear(self.feat_mat[ins_id])
        embed = self.embed(input_ids)
        lstm_out, _ = self.lstm(embed)
        lstm_out = self.lstm_linear(torch.flatten(lstm_out, start_dim=1))
        mix_feat = self.mix_linear(torch.cat([feat_mat, lstm_out], dim=-1))
        logit = self.linear(mix_feat)
        return {"pred": logit, "mix_feat": mix_feat}
