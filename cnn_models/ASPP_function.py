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

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes=16, nkernels=16, rate1 = rate1, rate2 = rate2, rate3 = rate3, rate4 = rate4, pad1 = pad1, pad2 = pad2, pad3 = pad3, pad4 = pad4, showsizes=False):
        super(ASPP, self).__init__()

        stride = 1

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nkernels = nkernels
        self.showsizes = showsizes

        self.rate1 = rate1
        self.rate2 = rate2
        self.rate3 = rate3
        self.rate4 = rate4

        self.pad1 = pad1
        self.pad2 = pad2
        self.pad3 = pad3
        self.pad4 = pad4

        # Block 1
        self.B1_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=stride, padding=pad1, dilation=rate1, bias=True)
        self.B1_bn = nn.BatchNorm2d(self.nkernels)
        self.B1_relu = nn.ReLU(inplace=True)

        # Block 2
        self.B2_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=pad2, dilation=rate2, bias=True)
        self.B2_bn = nn.BatchNorm2d(self.nkernels)
        self.B2_relu = nn.ReLU(inplace=True)

        # Block 3
        self.B3_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=pad3, dilation=rate3, bias=True)
        self.B3_bn = nn.BatchNorm2d(self.nkernels)
        self.B3_relu = nn.ReLU(inplace=True)

        # Block 4
        self.B4_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=stride, padding=pad4, dilation=rate4, bias=True)
        self.B4_bn = nn.BatchNorm2d(self.nkernels)
        self.B4_relu = nn.ReLU(inplace=True)

        # Block 5
        self.B5_gp = torch.nn.MaxPool2d(kernel_size=3, stride=stride, padding=1, dilation=1, return_indices=False, ceil_mode=False)

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
        b5 = self.B5_gp(x)

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
        self.outplanes = outplanes

        self.ASPP_conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ASPP_bn = nn.BatchNorm2d(self.nkernels)
        self.ASPP_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Final 1x1 convolution for rich feature extraction
        x = self.ASPP_conv(x)
        x = self.ASPP_bn(x)
        x = self.ASPP_relu(x)

        return x
