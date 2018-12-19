# Imports
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn               as nn
import torch.backends.cudnn   as cudnn
import torch.distributed      as dist
import torchvision.transforms as transforms
import torchvision.datasets   as datasets
import torchvision.models     as models
import torch.nn.functional    as F
import warnings

# Model definitions
from ASPP_ResNet2  import ASPP_ResNet
from caffe_uresnet import UResNet

model1 = ASPP_ResNet(inplanes=16, in_channels=1, num_classes=3, showsizes=False)
model2 = UResNet(inplanes=16, input_channels=1, num_classes=3, showsizes=False)

# Sum the number of trainable parameters in a model
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Driver function
def main():

        params1    = count_parameters(model1)
        params2    = count_parameters(model2)
        paramDelta = abs(params2 - params1)

        print "Parameters in model1: ", params1
        print "Parameters in model2: ", params2
        print "Absolute difference:  ", paramDelta

if __name__ == "__main__":
        main()
