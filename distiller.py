import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math
from simam import *


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def dist_loss(source, target):
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args
        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5
        self.simam = simam_module(256)

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        
        feat_num = len(t_feats)
        # for idx in range(feat_num):
        #     print()
        #     print(idx)
        #     print(t_feats[idx].shape, s_feats[idx].shape) 


        both_simmam_loss = 0 
        if self.args.both_simmam_loss is not None:
            t_simam = self.simam(t_feats[-1])
            s_simam = self.simam(s_feats[-1])
            t_simam_3 = self.simam(t_feats[-2])
            s_simam_3 = self.simam(self.Connectors[-3](s_feats[-2]))
            b,c,h,w = t_simam.shape
            both_simmam_loss += (s_simam / torch.norm(s_simam, p = 2) - t_simam / torch.norm(t_simam, p = 2)).pow(2).sum() / (b)
            b,c,h,w = t_simam_3.shape
            both_simmam_loss += (s_simam_3 / torch.norm(s_simam_3, p = 2) - t_simam_3 / torch.norm(t_simam_3, p = 2)).pow(2).sum() / (b)
            both_simmam_loss = self.args.both_simmam_loss * both_simmam_loss

        t_simmam_loss = 0 
        if self.args.t_simmam_loss is not None:
            t_simam = self.simam(t_feats[-1])
            b,c,h,w = t_simam.shape
            t_simmam_loss = self.args.t_simmam_loss * (s_feats[-1] / torch.norm(s_feats[-1], p = 2) - t_simam / torch.norm(t_simam, p = 2)).pow(2).sum() / (b)

        both_simmam_kld_loss = 0 
        if self.args.both_simmam_kld_loss is not None:
            t_simam = self.simam(t_feats[-1])
            s_simam = self.simam(s_feats[-1])
            both_simmam_kld_loss =  self.args.both_simmam_kld_loss * torch.nn.KLDivLoss()(F.log_softmax(t_simam / self.temperature, dim=1), F.softmax(s_simam / self.temperature, dim=1))

        # print(both_simmam_loss, t_simmam_loss, both_simmam_kld_loss)
        return s_out, both_simmam_loss, t_simmam_loss, both_simmam_kld_loss

