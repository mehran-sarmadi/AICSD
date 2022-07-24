import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

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
    def __init__(self, t_net, s_net):
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

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]

    def forward(self, x):

        #print('Teacherrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        t_feats, t_out, dist_t = self.t_net.extract_feature(x)
        #print('Studentttttttttttttttttttttttttttttttttttttttttt')
        s_feats, s_out, dist_s = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        TF = torch.mean((dist_t[0]), axis=1).cpu().detach()
        SF = torch.mean((dist_s[0]), axis=1).cpu()

        #print()
        #print(TF.shape)
        #print(SF.shape)

        loss_distill = 0
        loss_distill = dist_loss(SF, TF.detach())
        #print(loss_distill)
        #for i in range(feat_num):
         #   s_feats[i] = self.Connectors[i](s_feats[i])
            #print(s_feats[i].shape)
         #   loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
          #                  / self.loss_divider[i]

        #print(loss_distill)


        #T = 1
        #loss = nn.KLDivLoss()(F.log_softmax(s_out/T, dim=1), F.softmax(s_out/T, dim=1))

        #print('###########################################################################')
        #print(loss*1e9)
        #print('###########################################################################')
        #print(loss_distill)
        #print('###########################################################################')

        return s_out, 0, loss_distill
