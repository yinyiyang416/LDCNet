import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from lib.lightrfb import LightRFB,test_conv
from lib.Res2Net_v1b import res2net50_v1b_26w_4s, ddfresnet
from ddf import DDFPack,DDFUpPack,DDFBlock



#结合高级feature和低级feature
class conbine_feature(nn.Module):
    def __init__(self,channels_in1 = 32,channels_in2 = 24, channels_out = 16):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(channels_in1, channels_out)
        self.up2_low = nn.Conv2d(channels_in2, channels_out, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(channels_out)
        self.up2_act = nn.PReLU(channels_out)
        self.refine = nn.Sequential(nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False), nn.BatchNorm2d(channels_out), nn.PReLU())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        refine_feature = self.refine(self.up2_act(high_fea + low_fea))

        return refine_feature


#结合高级feature和低级feature,使用cat
class conbine_feature_cat(nn.Module):
    def __init__(self,channels_in1 = 32,channels_in2 = 24, channels_out = 16):
        super(conbine_feature_cat, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(channels_in1, channels_out)
        self.up2_low = nn.Conv2d(channels_in2, channels_out, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(channels_out)
        self.up2_act = nn.PReLU(channels_out * 2)
        self.refine = nn.Sequential(nn.Conv2d(channels_out * 2, channels_out, 3, padding=1, bias=False), nn.BatchNorm2d(channels_out), nn.PReLU())

    def forward(self, low_fea, high_fea):
        high_fea = self.up2_high(high_fea)
        low_fea = self.up2_bn2(self.up2_low(low_fea))
        cat_fea = torch.cat((high_fea, low_fea),1)
        refine_feature = self.refine(self.up2_act(cat_fea))
        return refine_feature


#拆分卷积模块
class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)# torch.chunk(input, chunks, dim=0)，拆分张量
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output


class LDCNet(nn.Module):
    def __init__(self):
        super(LDCNet, self).__init__()
        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        self.High_RFB = LightRFB(channels_in=1024, channels_mid=512, channels_out=256)
        self.Low_RFB = LightRFB(channels_in=512, channels_mid=256, channels_out=128)
        self.decoder = conbine_feature(channels_in1 = 256,channels_in2 = 128, channels_out = 16)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        self.ddf1 = DDFPack(256)
        self.ddf2 = DDFPack(256)

    def load_backbone(self, pretrained_dict, logger):
        model_dict = self.state_dict()
        logger.info("load_state_dict!!!")
        for k, v in pretrained_dict.items():
            if (k in model_dict):
                logger.info("load:%s" % k)
            else:
                logger.info("jump over:%s" % k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        print(x.shape)
        if len(x.shape) == 4:  # Pretrain
            origin_shape = x.shape
            x = self.feature_extractor.conv1(x)
            x = self.feature_extractor.bn1(x)
            x = self.feature_extractor.relu(x)
            x = self.feature_extractor.maxpool(x)
            x1 = self.feature_extractor.layer1(x)
            low_feature = self.feature_extractor.layer2(x1)
            high_feature = self.feature_extractor.layer3(low_feature)

            high_feature = self.High_RFB(high_feature)
            low_feature = self.Low_RFB(low_feature)

            high_feature = self.ddf1(high_feature) + high_feature
            high_feature = self.ddf2(high_feature) + high_feature

            high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                         mode="bilinear",
                                         align_corners=False)
            out = self.decoder(low_feature, high_feature)
            out = torch.sigmoid(
                F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                              align_corners=False))
        else:
            assert("worng input size")
        return out



if __name__ == "__main__":
    a = torch.randn(5, 3, 256, 256).cuda()
    mobile = LDCNet().cuda()
    print(mobile(a).shape)
