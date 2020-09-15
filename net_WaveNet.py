# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
from torchsummary import summary


class ResBlock_WaveNet(nn.Module): 

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(ResBlock_WaveNet, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            # paddingのサイズは端っこも使うためにdilation_rateと一緒でいいのでは
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=dilation_rate, dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=dilation_rate, dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class cbr(nn.Module):    
    def __init__(self,channel_in,channel_out, kernel_size):
        super(cbr,self).__init__()
        cbr_blocks = []
        cbr_blocks.append(nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size))
        cbr_blocks.append(nn.BatchNorm1d(channel_out))
        cbr_blocks.append(nn.ReLU(inplace=True))
        self.cbr_blocks = nn.Sequential(*cbr_blocks)
        
    def forward(self,x):
        out = self.cbr_blocks(x)
        return out


class Classifier(nn.Module):
    def __init__(self, inch=8, kernel_size=3):
        super().__init__()
        wave_blocks = []
        wave_blocks.append(cbr(channel_in=inch, channel_out=8, kernel_size=1))
        wave_blocks.append(ResBlock_WaveNet(8, 16, 12, kernel_size))
        wave_blocks.append(ResBlock_WaveNet(16, 32, 8, kernel_size))
        wave_blocks.append(ResBlock_WaveNet(32, 64, 4, kernel_size))
        wave_blocks.append(ResBlock_WaveNet(64, 128, 1, kernel_size))
        wave_blocks.append(cbr(channel_in=128, channel_out=128, kernel_size=1))
        self.wave_blocks = nn.Sequential(*wave_blocks)
        #self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        # permute無いとだめ．
        x = x.permute(0, 2, 1)
        x = self.wave_blocks(x)
        x = x.permute(0, 2, 1)
        #x, _ = self.LSTM(x)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    inputs = torch.randn(1, 8, 8)
    model = Classifier()
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias, a=-1.0, b=0.0)
    model.apply(init_weights)
    print('*'*30)
    print(f'Network structure\n')
#     summary(model, (1, 128, 64), device='cpu')
    print(model)
    print('*'*30)
    outputs = model(inputs) 
    print(f'\noutput size : {outputs.size()}\n')

    num_parameters=count_parameters(model)
    print(num_parameters)
    print(summary(model, (128, 8), device='cpu'))
#     print(torch.squeeze(outputs))   

    


    


# class Classifier(nn.Module):
#     def __init__(self, inch=8, kernel_size=3):
#         super().__init__()
#         wave_blocks = []
#         #self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
#         wave_blocks.append(cbr(channel_in=inch, channel_out=16, kernel_size=1))
#         wave_blocks.append(ResBlock_WaveNet(16, 16, 12, kernel_size))
#         wave_blocks.append(ResBlock_WaveNet(16, 32, 8, kernel_size))
#         wave_blocks.append(ResBlock_WaveNet(32, 64, 4, kernel_size))
#         wave_blocks.append(ResBlock_WaveNet(64, 128, 1, kernel_size))
#         wave_blocks.append(nn.Linear(128, 11))
#         self.wave_blocks = nn.Sequential(*wave_blocks)

#     def forward(self, x):
#         # x = x.permute(0, 2, 1)
#         out = self.wave_blocks(x)
#         return out
