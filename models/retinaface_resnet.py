from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from models.resnet import *
from data.config import cfg_re50 

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv1 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv2_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv2_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv3_1 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv3_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv1 = self.conv1(input)

        conv2_1 = self.conv2(input)
        conv2_2 = self.conv2_2(conv2_1)

        conv3_1 = self.conv2_2(conv2_1)
        conv3_2 = self.conv3_2(conv3_1)

        out = torch.cat([conv1, conv2_2, conv3_2], dim=1)
        out = F.relu(out)
        return out


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, phase, backbone, head, cfg=None):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        self.body = backbone

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.fpn_topdown1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.fpn_topdown2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.fpn_topdown3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)   

        self.ClassHead =  nn.ModuleList(head[0])
        self.BboxHead =  nn.ModuleList(head[1])
        self.LandmarkHead =  nn.ModuleList(head[2])


    def forward(self,inputs):
        of1, of2, of3, of4= self.body(inputs)
        ## fpn
        output1 = self.fpn_topdown1(of2)
        output2 = self.fpn_topdown2(of3)
        output3 = self.fpn_topdown3(of4)
        
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        
        # SSH
        feature1 = self.ssh1(output1)
        feature2 = self.ssh2(output2)
        feature3 = self.ssh3(output3)
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def xavier(self, param):
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()
    

def multiboxhead(fpn_num=3, inchannels=64,num_classes=2):
    classhead = []
    bboxhead = []
    landmarkhead = []
    for i in range(fpn_num):
        classhead.append(ClassHead(inchannels,num_classes))
        bboxhead.append(BboxHead(inchannels,num_classes))
        landmarkhead.append(LandmarkHead(inchannels,num_classes))
    return (classhead,bboxhead,landmarkhead)


def model_map(net_name='resnet50'):
    _dicts = { 'resnet18': resnet18,
              'resnet50': resnet50,
              'resnet101': resnet101, 'resnet152': resnet152}
    return _dicts[net_name]()


def build_net_resnet(phase, num_classes=2, net_name='resnet50', cfg=None):
    resnet = model_map(net_name)
    head = multiboxhead(fpn_num=3, inchannels=64,num_classes=2)
    model = RetinaFace(phase, resnet,head, cfg)
    return model


if __name__=='__main__':
    retinaface = build_net_resnet(phase='train', cfg=cfg_re50)
    print(retinaface)