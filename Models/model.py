import numpy as np
import torch
from torch import nn
from Models.AbsolutePositionalEncoding import SinPositionalEncoding, LearnablePositionalEncoding
from Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def Encoder_factory(config):
    if config['Encoder_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Encoder_Type'][0] == 'C-T':
        model = ConvTran(config, num_classes=config['num_labels'])
    elif config['Encoder_Type'][0] == 'C':
        model = Conv(config, num_classes=config['num_labels'])
    elif config['Encoder_Type'][0] == 'TS-TCC':
        model = TS_TCC(config, num_classes=config['num_labels'])
    else:
        model = Rep(config)
    return model


def Rep_factory(config):
    if config['Rep_Type'][0] == 'S2V':
        model = Rep(config)
    else:
        model = Rep(config)
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = SinPositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.C_out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        # Self Supervised out ---------------------------
        SS_out = out.permute(1, 0, 2)
        SS_out = SS_out[-1]
        # Supervised Out ---------------------------------
        # S_out = out.permute(0, 2, 1)
        # S_out = self.gap(S_out)
        # out = self.flatten(out)
        S_out = self.C_out(SS_out)

        return SS_out, S_out


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        rep_size = config['rep_size']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())
        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = SinPositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(rep_size, eps=1e-5)

        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, rep_size),
            nn.Dropout(config['dropout']))
        self.Residual = nn.Linear(emb_size, rep_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        # self.out_sim = nn.Linear(emb_size, config['batch_size']-1)
        self.C_out = nn.Linear(rep_size, num_classes)

    def forward(self, x):
        # x = self.causal_embed_layer(x)
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        # Augmenting

        # Adding class token
        # cls_tokens = self.cls_token.expand(x_src.shape[0], -1, -1)
        # x_src = torch.cat((cls_tokens, x_src), dim=1)

        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = self.Residual(att) + self.FeedForward(att)
        out = self.LayerNorm2(out)

        # Distance out ---------------------------
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        Representation_out = self.flatten(out)
        # Sim_out = self.out_sim(out)
        C_out = self.C_out(Representation_out)

        return Representation_out, C_out


class NewConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.causal_embed_layer = nn.Sequential(CausalConv1d(channel_size, emb_size, kernel_size=8, dilation=2),
                                                nn.BatchNorm1d(emb_size),
                                                nn.GELU())

        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, 8], stride=2, padding='valid'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = SinPositionalEncoding(emb_size, dropout=config['dropout'], max_len=86)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, 86, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        # self.out_sim = nn.Linear(emb_size, config['batch_size']-1)
        self.C_out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # x = self.causal_embed_layer(x)
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        # Augmenting

        # Adding class token
        # cls_tokens = self.cls_token.expand(x_src.shape[0], -1, -1)
        # x_src = torch.cat((cls_tokens, x_src), dim=1)

        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        # Distance out ---------------------------
        out = out.permute(0, 2, 1)
        # out = self.gap(out)
        Representation_out = self.flatten(out)
        # Sim_out = self.out_sim(out)
        C_out = self.C_out(Representation_out)

        return Representation_out, C_out


class Rep(nn.Module):

    def __init__(self, config):
        super().__init__()
        rep_size = config['rep_size']
        self.Rep = nn.Linear(2 * rep_size, 1)

    def forward(self, x):
        representation = self.Rep(x)
        return representation


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(nn.functional.pad(x, (self.__padding, 0)))


class Conv(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        rep_size = config['rep_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        # Embedding Layer -----------------------------------------------------------
        self.causal_Conv1 = nn.Sequential(CausalConv1d(channel_size, emb_size, kernel_size=8, stride=2, dilation=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv2 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=5, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(CausalConv1d(emb_size, rep_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(rep_size), nn.GELU())
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.attention_layer = Attention_Rel_Scl(rep_size, num_heads, 1, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, rep_size),
            nn.Dropout(config['dropout']))

        self.LayerNorm = nn.LayerNorm(rep_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(rep_size, eps=1e-5)

        self.flatten = nn.Flatten()
        self.C_out = nn.Linear(rep_size, num_classes)

    def forward(self, x):
        x = self.causal_Conv1(x)
        x = self.causal_Conv2(x)
        x = self.causal_Conv3(x)
        x = self.gap(x)
        x_src = x.permute(0, 2, 1)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        Rep_out = self.flatten(out)
        C_out = self.C_out(Rep_out)
        return Rep_out, C_out


class TS_TCC(nn.Module):
    def __init__(self, config, num_classes):
        super(TS_TCC, self).__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(channel_size, 32, kernel_size=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config['dropout'])
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.flatten = nn.Flatten()
        self.C_out = nn.Linear(emb_size, num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = self.flatten(x)
        C_out = self.C_out(x_flat)
        return x_flat, C_out
