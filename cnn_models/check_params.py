#torch
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

from ASPP_ResNet2 import ASPP_ResNet

model = ASPP_ResNet(inplanes=16, in_channels=1, num_classes=3, showsizes=False)

#for params in model.parameters():
#        print params

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
        params = count_parameters(model)
        print params

if __name__ == "__main__":
        main()
