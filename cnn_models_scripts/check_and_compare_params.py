# Torch
import torch

# Models
from ASPP_ResNet1 import ASPP_ResNet
from caffe_uresnet import UResNet

model1 = ASPP_ResNet(inplanes=16, in_channels=1, num_classes=3, showsizes=False)
model2 = UResNet(inplanes=20, input_channels=1, num_classes=3, showsizes=False)

# Sum the number of trainable parameters in a model.
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Driver function
def main():

        params1 = count_parameters(model1.eval())
        params2 = count_parameters(model2.eval())
        paramDelta = abs( params2 - params1 )
        percentDiff = ( float( paramDelta) / params2 ) * 100

        print "Parameters in model1: ", params1
        print "Parameters in model2: ", params2
        print "Absolute difference:  ", paramDelta
        print "Percent difference: ",   percentDiff

if __name__ == "__main__":
        main()
