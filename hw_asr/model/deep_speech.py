from torch import nn
import torch
from hw_asr.base import BaseModel
import math

rnn_types = {
    'rnn': nn.RNNBase,
    'lstm': nn.LSTM,
    'gru': nn.GRU
}

class RNNBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, rnn_type, use_norm=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size) if use_norm else nn.Identity()
    
    def forward(self, args):
        x = args[0]
        h = args[1]
        x = self.layer_norm(x)
        x, h = self.rnn(x, h)
        return x, h

class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, rnn_type='lstm', rnn_hidden=1024, bidirectional=True, num_conv_layers=2, num_rnn_layers=5, **batch):
        super().__init__(n_feats, n_class, **batch)
        assert 3 >= num_conv_layers > 0
        assert num_rnn_layers > 0
        assert rnn_type in rnn_types
        self.bidirectional = bidirectional
        self.conv_layers = nn.Sequential()
        conv1 = nn.Sequential(
            nn.Conv2d(padding=(20, 5), kernel_size=(41, 11), in_channels=1, out_channels=32, stride=(2, 2)),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv_layers.add_module("conv1", conv1)
        if num_conv_layers == 2:
            conv2 = nn.Sequential(
                nn.Conv2d(padding=(10, 5), kernel_size=(21, 11), in_channels=32, out_channels=32, stride=(2, 1)),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(num_features=32),
            )
            self.conv_layers.add_module("conv2", conv2)
        if num_conv_layers == 3:
            conv3 = nn.Sequential(
                nn.Conv2d(padding=(10, 5), kernel_size=(21, 11), in_channels=32, out_channels=96, stride=(2, 1)),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(num_features=96),
            )
            self.conv_layers.add_module("conv3", conv3)
        
        rnn_type = rnn_types[rnn_type]
        self.rnn_input_size = n_feats
        self.rnn_input_size = int(math.floor(self.rnn_input_size + 2 * 20 - 41) / 2 + 1)
        self.rnn_input_size = int(math.floor(self.rnn_input_size + 2 * 10 - 21) / 2 + 1) if num_conv_layers == 2 else self.rnn_input_size
        self.rnn_input_size = int(math.floor(self.rnn_input_size + 2 * 10 - 21) / 2 + 1) if num_conv_layers == 3 else self.rnn_input_size
        self.rnn_input_size *= (96 if num_conv_layers == 3 else 32)
        
        self.all_rnn = nn.Sequential(
        )

        for i in range(num_rnn_layers):
            if i == 0:
                rnn_0 = nn.Sequential(
                    RNNBatchNorm(input_size=self.rnn_input_size, hidden_size=rnn_hidden, rnn_type=rnn_type, bidirectional=bidirectional),
                )
                self.all_rnn.add_module(f"rnn_0", rnn_0)
            else:
                rnn_i = nn.Sequential(
                    RNNBatchNorm(input_size=rnn_hidden * 2, hidden_size=rnn_hidden, rnn_type=rnn_type, bidirectional=bidirectional, use_norm=True),
                )
                self.all_rnn.add_module(f"rnn_{i}", rnn_i)
        
        # if not self.bidirectional:
        #     self.look_ahead = nn.Sequential(

        #     )
        self.look_ahead = nn.Identity()
        self.fc = nn.Linear(in_features=rnn_hidden * 2, out_features=n_class)


    def forward(self, spectrogram, **batch):
        x = self.conv_layers(spectrogram.unsqueeze(1).transpose(2, 3))
        x = x.view(x.shape[0], -1, self.rnn_input_size)
        x, _ = self.all_rnn((x, None))
        x = self.look_ahead(x)
        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths / 2).int()