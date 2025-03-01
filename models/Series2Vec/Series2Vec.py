import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.Series2Vec.Attention import Attention_Rel_Scl


class Seires2Vec(nn.Module):
    def __init__(self, config, num_classes):
        super(Seires2Vec, self).__init__()
        input_channels, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        rep_size = config['rep_size']
        # Embedding Layer -----------------------------------------------------------
        # self.embed_layer = ConvEncoder(input_channels, [emb_size]*config['layers']+[rep_size], kernel_size=3)
        # self.embed_layer_f = ConvEncoder(input_channels, [emb_size] * config['layers'] + [rep_size], kernel_size=3)
        self.embed_layer = DisjoinEncoder(input_channels, emb_size, rep_size, kernel_size=8)
        self.embed_layer_f = DisjoinEncoder(input_channels, emb_size, rep_size, kernel_size=8)
        # self.embed_layer = ConvTarnEncoder(input_channels, seq_len, emb_size, rep_size, kernel_size=8)

        self.LayerNorm = nn.LayerNorm(rep_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(rep_size, eps=1e-5)
        # self.attention_layer = Attention(rep_size, num_heads, config['dropout'])
        self.attention_layer = nn.MultiheadAttention(rep_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(rep_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, rep_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap_f = nn.AdaptiveAvgPool1d(1)
        self.C_out = nn.Linear(2*rep_size, num_classes)
        # self.C_out = nn.Linear(rep_size, num_classes)

    def linear_prob(self, x):
        out = self.embed_layer(x)
        out = self.gap(out)

        x_f = torch.fft.fft(x).float()
        out_f = self.embed_layer_f(x_f)
        out_f = self.gap_f(out_f)
        return torch.cat((out.squeeze(), out_f.squeeze()), dim=1)

    def Pretrain_forward(self, x):
        x_src = self.embed_layer(x)
        x_src = self.gap(x_src)
        x_src = x_src.permute(2, 0, 1)

        att, _ = self.attention_layer(x_src, x_src, x_src)
        att += x_src
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        # ----------------------------------------
        x_f = torch.fft.fft(x).float()
        x_f = self.embed_layer_f(x_f)
        x_f = self.gap_f(x_f)
        x_f = x_f.permute(2, 0, 1)
        # Distance out ---------------------------
        Rep_out = out.squeeze()
        Rep_out_f = x_f.squeeze()
        distance = torch.cdist(Rep_out, Rep_out)
        distance_f = torch.cdist(Rep_out_f, Rep_out_f)
        return distance, distance_f, Rep_out, Rep_out_f

    def forward(self, x):
        x_src = self.embed_layer(x)
        out = self.gap(x_src)
        x_f = torch.fft.fft(x).float()
        out_f = self.embed_layer_f(x_f)
        out_f = self.gap_f(out_f)
        C_out = self.C_out(torch.cat((out.squeeze(), out_f.squeeze()), dim=1))
        # C_out = self.C_out(out_f.squeeze())
        return C_out


class DisjoinEncoder(nn.Module):
    def __init__(self, channel_size, emb_size, rep_size, kernel_size):
        super().__init__()
        self.temporal_CNN = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, kernel_size], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.spatial_CNN = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.rep_CNN = nn.Sequential(nn.Conv1d(emb_size, rep_size, kernel_size=3),
                                     nn.BatchNorm1d(rep_size),
                                     nn.GELU())
        # Initialize the weights
        self.initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x)
        x = self.rep_CNN(x.squeeze())
        return x

    def initialize_weights(self):
        # Custom weight initialization, you can choose different methods
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layer weights using Xavier/Glorot initialization
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    # Initialize biases with zeros
                    init.constant_(m.bias, 0)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.ConvEncoder = nn.Sequential(*[ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.ConvEncoder(x)


class ConvTarnEncoder(nn.Module):
    def __init__(self, channel_size, seq_len, emb_size, rep_size, kernel_size):
        super().__init__()
        self.temporal_CNN = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, kernel_size], padding='same'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.spatial_CNN = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.attention_layer = Attention_Rel_Scl(emb_size, 8, seq_len, 0.1)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(rep_size, eps=1e-5)
        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_size, rep_size),
            nn.Dropout(0.1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x).squeeze(2).transpose(1, 2)

        att = x + self.attention_layer(x)
        att = self.LayerNorm1(att)
        out = self.FeedForward(att)
        out = self.LayerNorm2(out)

        return out.transpose(1, 2)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual