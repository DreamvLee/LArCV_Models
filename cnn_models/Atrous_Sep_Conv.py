# Liquid Argon Computer Vision
# Encoder-Decoder with Atrous Seperable Convolution for Semantic Image Segmentation
# Based on DeepLabV3+ Architecture

# BibTex
##
# @article{deeplabv3plus2018,
#   title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
#   author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
#   journal={arXiv:1802.02611},
#   year={2018}
# }
##
# @inproceedings{mobilenetv22018,
#   title={Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation},
#   author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
#   booktitle={CVPR},
#   year={2018}
# }
##

#####
# Import Script
import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
from numbers import Integral

# python,numpy
import os
import sys
import commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import warnings

#####


class depthwise_separable_conv(nn.Module):  # from _sep_conv
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(depthwise_separable_conv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

        self.bypass = None
        self.bnpass = None
        if inplanes != planes or stride > 1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bnpass = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.bypass is not None:
            outbp = self.bypass(x)
            outbp = self.bnpass(outbp)
            out += outbp
        else:
            out += x

        out = self.relu(out)

        return out


class ASPP(nn.Module):  # from ASPP_Layer
    def __init__(self, inplanes, outplanes=16, nkernels=16, showsizes=False):
        super(ASPP, self).__init__()

        stride = 1
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nkernels = outplanes
        self.showsizes = showsizes

        # Block 1
        self.B1_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=stride, padding=0, dilation=1, bias=True)
        self.B1_bn = nn.BatchNorm2d(self.nkernels)
        self.B1_relu = nn.ReLU(inplace=True)

        # Block 2
        self.B2_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=True)
        self.B2_bn = nn.BatchNorm2d(self.nkernels)
        self.B2_relu = nn.ReLU(inplace=True)

        # Block 3
        self.B3_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=2, dilation=2, bias=True)
        self.B3_bn = nn.BatchNorm2d(self.nkernels)
        self.B3_relu = nn.ReLU(inplace=True)

        # Block 4
        self.B4_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=4, dilation=4, bias=True)
        self.B4_bn = nn.BatchNorm2d(self.nkernels)
        self.B4_relu = nn.ReLU(inplace=True)

        # Block 5
        self.B5_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=6, dilation=6, bias=True)
        self.B5_bn = nn.BatchNorm2d(self.nkernels)
        self.B5_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Atrous Spatial Pyramid Pooling
        # Block 1
        b1 = self.B1_conv(x)
        b1 = self.B1_bn(b1)
        b1 = self.B1_relu(b1)

        # Block 2
        b2 = self.B2_conv(x)
        b2 = self.B2_bn(b2)
        b2 = self.B2_relu(b2)

        # Block 3
        b3 = self.B3_conv(x)
        b3 = self.B3_bn(b3)
        b3 = self.B3_relu(b3)

        # Block 4
        b4 = self.B4_conv(x)
        b4 = self.B4_bn(b4)
        b4 = self.B4_relu(b4)

        # Block 5
        b5 = self.B5_conv(x)
        b5 = self.B5_bn(b5)
        b5 = self.B5_relu(b5)

        if self.showsizes:
            print "b1 dim:", b1.size()
            print "b2 dim:", b2.size()
            print "b3 dim:", b3.size()
            print "b4 dim:", b4.size()
            print "b5 dim:", b5.size()

        # Concatenation along the depth

        x = torch.cat((b1, b2, b3, b4, b5), 1)

        return x


class ASPP_post(nn.Module):  # from ASPP_combine
    def __init__(self, inplanes, outplanes):
        super(ASPP_post, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nkernels = outplanes

        self.ASPP_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=4, padding=0, bias=True)
        self.ASPP_bn = nn.BatchNorm2d(self.nkernels)
        self.ASPP_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.ASPP_conv(x)
        x = self.ASPP_bn(x)
        x = self.ASPP_relu(x)

        return x


class Atrous_Sep_Conv(nn.Module):

    def __init__(self, num_classes=3, in_channels=3, inplanes=16, showsizes=False):
        super(Atrous_Sep_Conv, self).__init__()

        # Class Variables
        stride = 1
        stride_mp = 2

        self.inplanes = inplanes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.showsizes = showsizes
        self.nMidflow = 9

        # Deep Convolutional Neural Network

        # Entry Flow
        # Block 1
        # Standard Convolutions | (32, 3x3, s=2), (48, 3x3, s=1)

        self.ent_conv1 = nn.Conv2d(in_channels, self.inplanes * 2, kernel_size=3, stride=stride * 2, padding=1, bias=True)
        self.ent_bn1 = nn.BatchNorm2d(self.inplanes * 2)
        self.ent_relu1 = nn.ReLU(inplace=True)

        self.ent_conv2 = nn.Conv2d(self.inplanes * 2, self.inplanes * 3, kernel_size=3, padding=1, stride=stride, bias=True)
        self.ent_bn2 = nn.BatchNorm2d(self.inplanes * 3)
        self.ent_relu2 = nn.ReLU(inplace=True)

        # Short Cut Connection 1 | (64, 1x1, s=2)

        self.short_cut1 = nn.Conv2d(self.inplanes * 3, out_channels=64, kernel_size=1, stride=stride)

        # Separable Convolutions | (64, 3x3, s=1), (64, 3x3, s=1), (64, 3x3, s=2)

        self.sepconv1 = self._sep_conv(self.inplanes * 3, self.inplanes * 4, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn1 = nn.BatchNorm2d(self.inplanes * 4)
        # self.sepconv_relu1 = nn.ReLU(inplace=True)

        self.sepconv2 = self._sep_conv(self.inplanes * 4, self.inplanes * 4, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn2 = nn.BatchNorm2d(self.inplanes * 4)
        # self.sepconv_relu2 = nn.ReLU(inplace=True)

        self.sepconv3 = self._sep_conv(self.inplanes * 4, self.inplanes * 4, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn3 = nn.BatchNorm2d(self.inplanes * 4)
        # self.sepconv_relu3 = nn.ReLU(inplace=True)

        # Block 2

        # Short Cut Connection 2 | (96, 1x1, s=1)

        self.short_cut2 = nn.Conv2d(self.inplanes * 4, out_channels=64, kernel_size=1, stride=stride * 2)

        # Separable Convolutions | (256, 3x3, s=1), (256, 3x3, s=1), (256, 3x3, s=2)

        self.sepconv4 = self._sep_conv(self.inplanes * 8, self.inplanes * 9, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn4 = nn.BatchNorm2d(self.inplanes * 9)
        # self.sepconv_relu4 = nn.ReLU(inplace=True)

        self.sepconv5 = self._sep_conv(self.inplanes * 9, self.inplanes * 9, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn5 = nn.BatchNorm2d(self.inplanes * 9)
        # self.sepconv_relu5 = nn.ReLU(inplace=True)

        self.sepconv6 = self._sep_conv(self.inplanes * 9, self.inplanes * 4, kernel_size=1, padding=0, stride=stride * 2)
        self.sepconv_bn6 = nn.BatchNorm2d(self.inplanes * 4)
        # self.sepconv_relu6 = nn.ReLU(inplace=True)

        # Block 3

        # Short Cut Connection 3 | (768, 1x1, s=2)

        self.short_cut3 = nn.Conv2d(self.inplanes * 4, out_channels=160, kernel_size=1, stride=stride * 2)

        # Separable Convolutions | (256, 3x3, s=1), (256, 3x3, s=1), (256, 3x3, s=2)

        self.sepconv7 = self._sep_conv(self.inplanes * 8, self.inplanes * 10, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn7 = nn.BatchNorm2d(self.inplanes * 10)
        # self.sepconv_relu7 = nn.ReLU(inplace=True)

        self.sepconv8 = self._sep_conv(self.inplanes * 10, self.inplanes * 10, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn8 = nn.BatchNorm2d(self.inplanes * 10)
        # self.sepconv_relu8 = nn.ReLU(inplace=True)

        self.sepconv9 = self._sep_conv(self.inplanes * 10, self.inplanes * 10, kernel_size=1, padding=0, stride=stride * 2)
        self.sepconv_bn9 = nn.BatchNorm2d(self.inplanes * 10)
        # self.sepconv_relu9 = nn.ReLU(inplace=True)

        # Middle Flow | Repeated 16 times
        # Separable Convolutions | (728, 3x3, s=1), (728, 3x3, s=1), (728, 3x3, s=1)

        self.sepconv10_l = [self._sep_conv(self.inplanes * 20, self.inplanes * 20, kernel_size=3, padding=1, stride=stride) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv10_l):
            self.__setattr__("sepconv10_%d" % (n), layer)

        self.sepconv_bn10_l = [nn.BatchNorm2d(self.inplanes * 20) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv_bn10_l):
            self.__setattr__("sepconv_bn10_%d" % (n), layer)
        # self.sepconv_relu10 = nn.ReLU(inplace=True)

        self.sepconv11_l = [self._sep_conv(self.inplanes * 20, self.inplanes * 20, kernel_size=3, padding=1, stride=stride) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv11_l):
            self.__setattr__("sepconv11_%d" % (n), layer)

        self.sepconv_bn11_l = [nn.BatchNorm2d(self.inplanes * 20) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv_bn11_l):
            self.__setattr__("sepconv_bn11_%d" % (n), layer)
        # self.sepconv_relu11 = nn.ReLU(inplace=True)

        self.sepconv12_l = [self._sep_conv(self.inplanes * 20, self.inplanes * 20, kernel_size=3, padding=1, stride=stride) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv12_l):
            self.__setattr__("sepconv12_%d" % (n), layer)

        self.sepconv_bn12_l = [nn.BatchNorm2d(self.inplanes * 20) for x in range(self.nMidflow)]
        for n, layer in enumerate(self.sepconv_bn12_l):
            self.__setattr__("sepconv_bn12_%d" % (n), layer)
        # self.sepconv_relu12 = nn.ReLU(inplace=True)

        # Exit Flow
        # Block 1
        # Short Cut Connection 4 | (1024, 1x1, s=2)

        self.short_cut4 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1, stride=stride * 2)

        # Separabel Convolutions | (728, 3x3, s=1), (1024, 3x3, s=1), (1024, 3x3, s=2)

        self.sepconv13 = self._sep_conv(in_channels=640, out_channels=512, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn13 = nn.BatchNorm2d(self.inplanes * 32)
        # self.sepconv_relu13 = nn.ReLU(inplace=True)

        self.sepconv14 = self._sep_conv(self.inplanes * 32, self.inplanes * 32, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn14 = nn.BatchNorm2d(self.inplanes * 32)
        # self.sepconv_relu14 = nn.ReLU(inplace=True)

        self.sepconv15 = self._sep_conv(self.inplanes * 32, self.inplanes * 32, kernel_size=1, padding=0, stride=stride * 2)
        self.sepconv_bn15 = nn.BatchNorm2d(self.inplanes * 32)
        # self.sepconv_relu15 = nn.ReLU(inplace=True)

        # Separabel Convolutions | (1536, 3x3, s=1), (1536, 3x3, s=1), (2048, 3x3, s=1)

        self.sepconv16 = self._sep_conv(self.inplanes * 64, self.inplanes * 64, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn16 = nn.BatchNorm2d(self.inplanes * 64)
        self.sepconv_relu16 = nn.ReLU(inplace=True)

        self.sepconv17 = self._sep_conv(self.inplanes * 64, self.inplanes * 64, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn17 = nn.BatchNorm2d(self.inplanes * 64)
        self.sepconv_relu17 = nn.ReLU(inplace=True)

        self.sepconv18 = self._sep_conv(self.inplanes * 64, self.inplanes * 64, kernel_size=3, padding=1, stride=stride)
        self.sepconv_bn18 = nn.BatchNorm2d(self.inplanes * 64)
        self.sepconv_relu18 = nn.ReLU(inplace=True)

        # Encoder
        # ASPP

        self.ASPP_enc_layer = self.ASPP_layer(self.inplanes * 64)
        self.ASPP_combine_layer = self.ASPP_combine(self.inplanes * 5, self.inplanes * 16)

        # self.ASPP_refine = nn.Conv2d(self.inplanes * 16, self.inplanes * 16, kernel_size=1, stride=stride * 4, padding=0, bias=True)

        # Decoder

        self.BiUp1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.BiUp2 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.reduce_channels = nn.Conv2d(self.inplanes * 64, self.inplanes * 16, kernel_size=1, stride=stride, padding=0, bias=True)
        self.reduce_bn = nn.BatchNorm2d(self.inplanes * 16)
        self.reduce_relu = nn.ReLU(inplace=True)

        self.refine_features = BasicBlock(self.inplanes * 32, num_classes, stride=stride)

        self.perceptron = nn.Conv2d(self.inplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # Loss Function

        self.softmax = nn.LogSoftmax(dim=1)  # should return [b,c=3,h,w], normalized over, c dimension

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _sep_conv(self, in_channels, out_channels, kernel_size, stride, padding):
        return depthwise_separable_conv(in_channels, out_channels, kernel_size, stride, padding)

    def ASPP_layer(self, inplanes):
        return ASPP(inplanes)

    def ASPP_combine(self, inplanes, outplanes):
        return ASPP_post(inplanes, outplanes)

    def forward(self, x):

        if self.showsizes:
            print "input:", x.size(), "is_cuda", x.is_cuda

        # DCNN
        # Entry Flow

        # Block 1
        # Standard Convolutions | (32, 3x3, s=2), (64, 3x3, s=1)
        x = self.ent_conv1(x)
        x = self.ent_bn1(x)
        x = self.ent_relu1(x)

        x = self.ent_conv2(x)
        x = self.ent_bn2(x)
        x = self.ent_relu2(x)

        if self.showsizes:
            print "x after entryflow:", x.size()

        # Short Cut Connection 1 | (128, 1x1, s=2)

        x0 = self.short_cut1(x)

        # Separable Convolutions | (128, 3x3, s=1), (128, 3x3, s=1), (128, 3x3, s=2)

        x = self.sepconv1(x)
        x = self.sepconv_bn1(x)
        # x = self.sepconv_relu1(x)

        x = self.sepconv2(x)
        x = self.sepconv_bn2(x)
        # x = self.sepconv_relu2(x)

        x = self.sepconv3(x)
        x = self.sepconv_bn3(x)
        # x = self.sepconv_relu3(x)

        # Concatenation 1 | Depth = 256

        if self.showsizes:
            print "x0 after short_cut1:", x0.size()
            print "x after sepconv3:", x.size()

        x = torch.cat((x, x0), 1)

        if self.showsizes:
            print "x after concat1:", x.size()

        # Block 2
        # Separable Convolutions | (256, 3x3, s=1), (256, 3x3, s=1), (256, 3x3, s=2)
        x = self.sepconv4(x)
        x = self.sepconv_bn4(x)
        # x = self.sepconv_relu4(x)

        x = self.sepconv5(x)
        x = self.sepconv_bn5(x)
        # x = self.sepconv_relu5(x)

        x = self.sepconv6(x)
        x = self.sepconv_bn6(x)
        # x = self.sepconv_relu6(x)

        # Short Cut Connection 2 | (256, 1x1, s=2)

        x0 = self.short_cut2(x0)

        # Concatenation 2 | Depth = 512

        if self.showsizes:
            print "x0 after short_cut2:", x0.size()
            print "x after sepconv6:", x.size()

        x = torch.cat((x0, x), 1)

        if self.showsizes:
            print "x after concat2:", x.size()

        # Block 3
        # Separable Convolutions | (256, 3x3, s=1), (256, 3x3, s=1), (256, 3x3, s=2)
        x = self.sepconv7(x)
        x = self.sepconv_bn7(x)
        # x = self.sepconv_relu7(x)

        x = self.sepconv8(x)
        x = self.sepconv_bn8(x)
        # x = self.sepconv_relu8(x)

        x = self.sepconv9(x)
        x = self.sepconv_bn9(x)
        # x = self.sepconv_relu9(x)

        # Short Cut Connection 3 | (728, 1x1, s=2)

        x0 = self.short_cut3(x0)

        if self.showsizes:
            print "x0 after short_cut3:", x0.size()
            print "x after sepconv9:", x.size()

        # Concatenation 3 | Depth = 1456

        x0 = torch.cat((x0, x), 1)

        if self.showsizes:
            print "x0 after concat3:", x0.size()

        # Middle Flow | Repeated 16 times | (728, 3x3, s=1), (728, 3x3, s=1), (728, 3x3, s=1)

        # x = self.sepconv10(x)
        # x = self.sepconv_bn10(x)
        # x = self.sepconv_relu10(x)

        for ii in range(self.nMidflow):
            x = self.sepconv10_l[ii](x0)
            x0 = x
            x = self.sepconv_bn10_l[ii](x)
            x = self.sepconv11_l[ii](x)
            x = self.sepconv_bn11_l[ii](x)
            x = self.sepconv12_l[ii](x)
            x = self.sepconv_bn12_l[ii](x)

        if self.showsizes:
            print "x after MidFlow:", x.size()

        # Concatenation 4 | Depth = 2184

        x0 = torch.cat((x0, x), 1)

        if self.showsizes:
            print "x0 after concat4:", x0.size()

        # Exit Flow
        # Block 1
        # Separable Convolutions | (728, 3x3, s=1), (1024, 3x3, s=1), (1024, 3x3, s=2)

        x = self.sepconv13(x0)
        x = self.sepconv_bn13(x)
        # x = self.sepconv_relu13(x)

        x = self.sepconv14(x)
        x = self.sepconv_bn14(x)
        # x = self.sepconv_relu14(x)

        x = self.sepconv15(x)
        x = self.sepconv_bn15(x)
        # x = self.sepconv_relu15(x)

        if self.showsizes:
            print "x after sepconv15:", x.size()

        # Short Cut Connection 4 | (1024, 1x1, s=2)

        x0 = self.short_cut4(x0)

        if self.showsizes:
            print "x0 after short_cut4:", x0.size()

        # Concatenation 5 | Depth = 2048

        x0 = torch.cat((x0, x), 1)

        if self.showsizes:
            print "x0 after concat5:", x0.size()

        # Block 2
        # Separabel Convolutions | (1536, 3x3, s=1), (1536, 3x3, s=1), (2048, 3x3, s=1)

        x = self.sepconv16(x0)
        x = self.sepconv_bn16(x)
        x = self.sepconv_relu16(x)

        x = self.sepconv17(x)
        x = self.sepconv_bn17(x)
        x = self.sepconv_relu17(x)

        # Low Level Feature Map

        x = self.sepconv18(x)
        x = self.sepconv_bn18(x)
        x = self.sepconv_relu18(x)

        if self.showsizes:
            print "x after sepconv18:", x.size()

        # Encoder
        # ASPP

        x_ASPP = self.ASPP_enc_layer(x)
        x_ASPP = self.ASPP_combine_layer(x_ASPP)

        # x_ASPP = self.ASPP_refine(x_ASPP)

        if self.showsizes:
            print "x_ASPP after ASPP_combine_layer:", x_ASPP.size()

        # Decoder

        x = self.reduce_channels(x)
        x = self.reduce_bn(x)
        x = self.reduce_relu(x)

        x_ASPP = self.BiUp1(x_ASPP)

        if self.showsizes:
            print "x after reduce_channels:", x.size()
            print "x_ASPP after BiUp1:", x_ASPP.size()

        x = torch.cat((x_ASPP, x), 1)

        if self.showsizes:
            print "x after final concat:", x.size()

        x = self.refine_features(x)

        if self.showsizes:
            print "x after refine_features", x.size()

        # x = self.perceptron(x)

        # if self.showsizes:
        #     print "x after perceptron", x.size()

        x = self.BiUp2(x)

        if self.showsizes:
            print "x after final BiUp2", x.size()

        x = self.softmax(x)

        return x
