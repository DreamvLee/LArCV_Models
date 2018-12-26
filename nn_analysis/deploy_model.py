#!/bin/env python

##########################################
# Import Scripts
##########################################
import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
from   numbers import Integral

##########################################
# Python, Numpy
##########################################
import os
import sys
import commands
import shutil
import time
import traceback
import numpy as np

##########################################
# ROOT, LArCV
##########################################
import ROOT
from   larcv import larcv

##########################################
# Torch
##########################################
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
##########################################

##########################################
# Model Definitions:
# {1: ASPP_ResNet, 2: caffe_uresnet}
##########################################
from ASPP_ResNet2  import ASPP_ResNet
from caffe_uresnet import UResNet

##########################################
# Hardware environment
##########################################
MODEL                  = 2
GPUID                  = 1
GPUMODE                = True
RESUME_FROM_CHECKPOINT = True
RUNPROFILER            = False
CHECKPOINT_FILE        = " "

##########################################
# Functions and Classes:
##########################################
# SegData: class to hold batch data
# we expect LArCV1Dataset to fill this object
class SegData:
    def __init__(self):
        self.dim = None
        self.images = None # adc image
        self.labels = None # labels
        self.weights = None # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim


# Data interface
class LArCV1Dataset:
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration. see test.py.ipynb for example of configuration
        self.name = name
        self.cfgfile = cfgfile
        return

    def init(self):
        # create instance of data file interface
        self.io = larcv.ThreadDatumFiller(self.name)
        self.io.configure(self.cfgfile)
        self.nentries = self.io.get_n_entries()
        self.io.set_next_index(0)
        print "[LArCV1Data] able to create ThreadDatumFiller"
        return

    def getbatch(self, batchsize):
        self.io.batch_process(batchsize)
        time.sleep(0.1)
        itry = 0
        while self.io.thread_running() and itry<100:
            time.sleep(0.01)
            itry += 1
        if itry>=100:
            raise RuntimeError("Batch Loader timed out")

        # fill SegData object
        data = SegData()
        dimv = self.io.dim() # c++ std vector through ROOT bindings
        self.dim  = (dimv[0], dimv[1], dimv[2], dimv[3] )
        self.dim3 = (dimv[0], dimv[2], dimv[3] )

        # numpy arrays
        data.np_images     = np.zeros( self.dim,  dtype=np.float32 )
        data.np_labels     = np.zeros( self.dim3, dtype=np.int )
        data.np_weights    = np.zeros( self.dim3, dtype=np.float32 )
        data.np_images[:]  = larcv.as_ndarray(self.io.data()).reshape(    self.dim  )[:]
        data.np_labels[:]  = larcv.as_ndarray(self.io.labels()).reshape(  self.dim3 )[:]
        data.np_weights[:] = larcv.as_ndarray(self.io.weights()).reshape( self.dim3 )[:]
        data.np_labels[:] += -1

        # pytorch tensors
        data.images = torch.from_numpy(data.np_images)
        data.labels = torch.from_numpy(data.np_labels)
        data.weight = torch.from_numpy(data.np_weights)
        #if GPUMODE:
        #    data.images.cuda()
        #    data.labels.cuda(async=False)
        #    data.weight.cuda(async=False)


        # debug values
        #print "max label: ",np.max(data.labels)
        #print "min label: ",np.min(data.labels)

        return data

torch.cuda.device(GPUID)

# global variables
input_larcv_filename = "/media/hdd1/larbys/ssnet_dllee_trainingdata/val.root"
inputmeta            = larcv.IOManager(larcv.IOManager.kREAD )
inputmeta.add_in_file( input_larcv_filename )
inputmeta.initialize()
width = 512
height= 832

# output IOManager
# we only save flow and visi prediction results
if MODEL == 1:
    output_larcv_filename = "output_ASPP_ResNet2.root"
elif MODEL == 2:
    output_larcv_filename = "output_caffe_UResNet.root"
outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
outputdata.set_out_file( output_larcv_filename )
outputdata.initialize()

best_prec1 = 0.0  # best accuracy, use to decide when to save network weights

##########################################
# Main Function
##########################################
def main(MODEL = MODEL):

    global best_prec1
    # training parameters
    lr              = 2.0e-5
    momentum        = 0.9
    weight_decay    = 1.0e-3
    batchsize_valid = 2

    # Create model -- instantiate on the GPU
    if MODEL == 1:
        if GPUMODE:
            model = ASPP_ResNet(inplanes=16, in_channels=1, num_classes=3,
                                showsizes=False)
        else:
            model = ASPP_ResNet(inplanes=16,in_channels=1,num_classes=3)
    elif MODEL == 2:
        if GPUMODE:
            model = UResNet(inplanes=20, input_channels=1, num_classes=3,
                                showsizes=False)
        else:
            model = UResNet(inplanes=20,input_channels=1,num_classes=3)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    cudnn.benchmark = True

    # Load checkpoint and state dictionary
    # Map the checkpoint file to the CPU -- removes GPU mapping
    map_location = {"cuda:0":"cpu","cuda:1":"cpu"}
    checkpoint = torch.load( CHECKPOINT_FILE, map_location = map_location )

    # Debugging block:
    # print "Checkpoint file mapped to CPU."
    # print "Press return to load best prediction tensor."
    # raw_input()

    best_prec1 = checkpoint["best_prec1"]
    print "state_dict size: ", len(checkpoint["state_dict"])
    print " "
    for p,t in checkpoint["state_dict"].items():
        print p, t.size()

    # Debugging block:
    # print " "
    # print "Best prediction tensor loaded."
    # print "Press return to load state dictionary."
    # raw_input()

    # Map checkpoint file to the desired GPU
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda(GPUID)

    print " "
    print "State dictionary mapped to GPU: ",GPUID
    if MODEL == 1:
        modelString = "ASPP_ResNet"
    elif MODEL == 2:
        modelString = "caffe_uresnet"
    print "Press return to deploy:",modelString
    print "From checkpoint:",       CHECKPOINT_FILE
    raw_input()

    # switch to evaluate mode
    model.eval()

    # LOAD THE DATASET
    validcfg = """ThreadDatumFillerValid: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    false
  #InputFiles:   ["/mnt/raid0/taritree/ssnet_training_data/train02.root"]
  InputFiles:   ["/media/hdd1/larbys/ssnet_dllee_trainingdata/val.root"]
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }

  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: false
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    with open("segfiller_valid.cfg",'w') as fvalid:
        print >> fvalid,validcfg

    iovalid = LArCV1Dataset("ThreadDatumFillerValid","segfiller_valid.cfg" )
    iovalid.init()
    iovalid.getbatch(batchsize_valid)

    NENTRIES = iovalid.io.get_n_entries()
    NENTRIES=10 #debug
    print "Number of entries in input: ",NENTRIES

    ientry = 0
    nbatches = NENTRIES/batchsize_valid
    if NENTRIES%batchsize_valid!=0:
        nbatches += 1

    for ibatch in range(nbatches):
        data = iovalid.getbatch(batchsize_valid)

        # convert to pytorch Variable (with automatic gradient calc.)
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda(GPUID))
            labels_var = torch.autograd.Variable(data.labels.cuda(GPUID),requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(GPUID),requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)

        # compute output
        output = model(images_var)
        ev_out_wire = outputdata.get_data(larcv.kProductImage2D, "wire")
        wire_t = images_var.data.cpu().numpy()
        weight_t = weight_var.data.cpu().numpy()
        # get predictions from gpu (turns validation routine into images)
        labels_np = output.data.cpu().numpy().astype(np.float32)
        labels_np = 10**labels_np

        for ib in range(batchsize_valid):
            if ientry>=NENTRIES:
                break
            inputmeta.read_entry(ientry)

            ev_meta   = inputmeta.get_data(larcv.kProductImage2D,"wire")
            outmeta   = ev_meta.Image2DArray()[2].meta()

            img_slice0 = labels_np[ib,0,:,:]
            nofill_lcv  = larcv.as_image2d_meta( img_slice0, outmeta )
            ev_out    = outputdata.get_data(larcv.kProductImage2D,"class0")
            ev_out.Append( nofill_lcv )

            img_slice1 = labels_np[ib,1,:,:]
            fill_lcv = larcv.as_image2d_meta( img_slice1, outmeta )
            ev_out = outputdata.get_data(larcv.kProductImage2D, "class1")
            ev_out.Append( fill_lcv)

            wire_slice=wire_t[ib,0,:,:]
            wire_out = larcv.as_image2d_meta(wire_slice,outmeta)
            ev_out_wire.Append( wire_out )

            #weight_slice=weight_t[ib,0,:,:]
            #weights_out = larcv.as_image2d_meta(weight_slice,outmeta)
            #ev_out_weights.Append( weights_out )

            outputdata.set_id(1,1,ibatch*batchsize_valid+ib)
            outputdata.save_entry()
            ientry += 1


    # save results
    outputdata.finalize()

if __name__ == '__main__':
    main()
