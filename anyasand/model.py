import torch
import torch.nn as nn
import torch.nn.functional as F


class AnyaAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, z_dim=32, hid_num=1):
        super().__init__()
        self._hid_num = hid_num

        # encoder
        self._in_layer = nn.Linear(in_dim, hidden_dim)
        self._en_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self._en_hid_layer = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hid_num)])
        self._en_out_layer = nn.Linear(hidden_dim, z_dim)

        # decoder
        self._de_in_layer = nn.Linear(z_dim, hidden_dim)
        self._de_hid_layer = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_dim)])
        self._de_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self._out_layer = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = F.relu(self._in_layer(x))
        x, lh = self._en_lstm(x)
        for i in range(self._hid_num):
            x = F.relu(self._en_hid_layer[i](x))
        z = F.relu(self._en_out_layer(x))

        x = F.relu(self._de_in_layer(z))
        for i in range(self._hid_num):
            x = F.relu(self._de_hid_layer[i](x))
        x, _ = self._de_lstm(x, lh)
        return self._out_layer(x)
