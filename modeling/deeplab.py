import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from vit_pytorch import CBAMViT
import numpy as  np
import cv2
import torch

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.cbam_vit = CBAMViT(
                        channels = 256,
                        image_size = 33,
                        patch_size = 11,
                        num_classes = 256,
                        dim = 256,
                        depth = 4,
                        heads = 8,
                        mlp_dim = 1025,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )
        print('doing with CBANViT ....')

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        channel_attn, spatial_attn = self.cbam_vit(x)
        x = x + (x * channel_attn.view(x.size(0), -1, 1, 1)) * spatial_attn.view(x.size(0), 1, 33, 33) 
        # print(f"\n\\n\ndeeplab decoder output shape \n\n: {x.shape}")
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_bn_before_relu(self):
        BNs = self.backbone.get_bn_before_relu()
        BNs += self.aspp.get_bn_before_relu()
        BNs += self.decoder.get_bn_before_relu()

        return BNs

    def get_channel_num(self):
        channels = self.backbone.get_channel_num()
        channels += self.aspp.get_channel_num()
        channels += self.decoder.get_channel_num()

        return channels

    def extract_feature(self, input):
        feats, x, low_level_feat = self.backbone.extract_feature(input)
        feat, x = self.aspp.extract_feature(x)
        feats += feat
        # print(f"\n\\n\ndeeplab decoder output shape \n\n: {feat[-1].shape}")

        feat, x = self.decoder.extract_feature(x, low_level_feat)
        feats += feat
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)


        #img = torch.sum((feats[5])[0], axis=0).cpu().detach().numpy()
        #b = np.copy(img)
        #cv2.normalize(img,b,0,255,cv2.NORM_MINMAX)
        #b = cv2.applyColorMap(np.uint8(b), cv2.COLORMAP_JET)
        #cv2.imwrite('feature.jpeg', b)  

        return feats, x

