import torch as th
from torch import nn
from fastNLP import MetricBase

class CnnReg(nn.Module):
    def __init__(self, num_embed, embd_dim):
        super(CnnReg, self).__init__()
        self.embed = nn.Embedding(num_embed, embd_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 5, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3968, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    def forward(self, input_ids):
        embed = self.embed(input_ids)
        cnn_feats = self.cnn(embed)
        cnn_feats = th.flatten(cnn_feats, start_dim=1)
        return {'pred': self.fc(cnn_feats).squeeze(1)}

class RegLSTM(nn.Module):
    def __init__(self, num_embed, embd_dim):
        super(RegLSTM, self).__init__()
        self.embed = nn.Embedding(num_embed, embd_dim)
        self.rnn = nn.LSTM(embd_dim, 128, bidirectional=True, num_layers=1)
        self.rnn = nn.GRU(embd_dim, 128, bidirectional=True, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(128*256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    def forward(self, input_ids):
        embed = self.embed(input_ids)
        rnn_feats, _ = self.rnn(embed)
        rnn_feats = th.flatten(rnn_feats, start_dim=1)
        return {'pred': self.classifier(rnn_feats).squeeze()}

class RegMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super(RegMetric, self).__init__()

        self._init_param_map(pred=pred, target=target)

        self.total = 0
        self.mse_loss = 0
        self.rmse_loss = 0
        self.mae_loss = 0
        self.r_square = 0

    def evaluate(self, pred, target):
        self.total += target.size(0)
        self.mse_loss += th.mean((pred-target)**2).item()
        self.rmse_loss += th.sqrt(th.mean((pred-target)**2)).item()
        self.mae_loss += th.mean(th.abs((pred-target))).item()
        self.r_square += 1 - th.mean((pred-target)**2)/ th.mean((pred - th.mean(target))**2)

    def get_metric(self, reset=True):
        mse = self.mse_loss / self.total
        rmse = self.rmse_loss / self.total
        mae = self.mae_loss / self.total
        r_square = self.r_square / self.total
        if reset:
            self.total = 0
            self.mse_loss = 0
            self.rmse_loss = 0
            self.mae_loss = 0
            self.r_square = 0
        return {'r_square': r_square, 'mse': mse, 'rmse': rmse, 'mae': mae}



class RegressionView:
    def __init__(self):
        pass
    def describe(self):
        pass