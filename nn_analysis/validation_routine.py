# High statistics validation for CNN models
#      trained on LArCV1 dataset
#
# Tufts University | Department of Physics
# High Energy Physics Research Group
# Liquid Argon Computer Vision
#
##########################################
# Import Scripts
##########################################
import torch.nn              as nn
import torch                 as torch
import cv2                   as cv
import torch.utils.model_zoo as model_zoo
import csv
import math
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
import numpy  as np
import pandas as pd

##########################################
# ROOT, LArCV
##########################################
import ROOT
from   larcv import larcv

##########################################
# Torch
##########################################
import torch
import torch.nn               as nn
import torch.nn.parallel
import torch.backends.cudnn   as cudnn
import torch.distributed      as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets   as datasets
import torchvision.models     as models
import torch.nn.functional    as F
import warnings

##########################################
# Model Definitions:
# {1: ASPP_ResNet, 2: caffe_uresnet}
##########################################
from ASPP_ResNet2  import ASPP_ResNet
from caffe_uresnet import UResNet

##########################################
# Hardware environment
##########################################
MODEL                   = 1
GPUID                   = 1
GPUMODE                 = True
RESUME_FROM_CHECKPOINT  = True
RUNPROFILER             = False
CHECKPOINT_FILE         = "/media/hdd1/kai/ASPP_/training_set3_fa18/Nov07/model_ASPP_best_Nov07_13-12-57.tar"
OUTPUT_CSV_DIR          = "/media/hdd1/kai/ASPP_/training_set3_fa18/"

##########################################
# Functions and Classes:
##########################################
# SegData: class to hold batch data
# we expect LArCV1Dataset to fill this object

class SegData:
    def __init__(self):
        self.dim     = None
        self.images  = None # adc image
        self.labels  = None # labels
        self.weights = None # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim

##########################################
# Data Interface
# See test.py.ipynb for example config.
##########################################
class LArCV1Dataset:
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration.
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

        # Fill SegData object
        data               = SegData()
        dimv               = self.io.dim() # c++ std vector via ROOT bindings
        self.dim           = (dimv[0], dimv[1], dimv[2], dimv[3] )
                             # batch, channel, height, width
                             # prediction tensor
        self.dim3          = (dimv[0], dimv[2], dimv[3] )
                             # batch, height, width
                             # target 'truth' tensor

        # Numpy arrays
        data.np_images     = np.zeros( self.dim,  dtype=np.float32 )
        data.np_labels     = np.zeros( self.dim3, dtype=np.int )
        data.np_weights    = np.zeros( self.dim3, dtype=np.float32 )
        data.np_images[:]  = larcv.as_ndarray(self.io.data()).reshape(    self.dim  )[:]
        data.np_labels[:]  = larcv.as_ndarray(self.io.labels()).reshape(  self.dim3 )[:]
        data.np_weights[:] = larcv.as_ndarray(self.io.weights()).reshape( self.dim3 )[:]
        data.np_labels[:] += -1

        # Torch tensors
        data.images        = torch.from_numpy(data.np_images)
        data.labels        = torch.from_numpy(data.np_labels)
        data.weight        = torch.from_numpy(data.np_weights)

        return data

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False

    def forward(self,predict,target,pixelweights):
        """
        predict:      (b,c,h,w) tensor with output from logsoftmax
        target:       (b,h,w) tensor with correct class
        pixelweights: (b,h,w) tensor with weights for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(pixelweights)
        # reduce for below is false, so returns (b,h,w)
        pixelloss = F.nll_loss(predict,target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)
        pixelloss *= pixelweights
        return torch.mean(pixelloss)

##########################################
# Global variables
##########################################
best_prec1 = 0.0  # best accuracy metric

##########################################
# Main Function
##########################################
def main(MODEL = MODEL):

    global best_prec1

    # TODO: Write loop that extracts checkpoint filename
    #       from text file and performs 1 iteration of
    #       the validation routine per checkpoint

    # Loop prototype:
    # with open('checkpoints.txt', 'r') as data:
    #     for line in data:
    #         checkpoint = line.rstrip()
    #         CHECKPOINT_FILE = checkpoint

    # Create model -- instantiate on the GPU
    if MODEL == 1:
        if GPUMODE:
            model = ASPP_ResNet(inplanes=16, in_channels=1, num_classes=3,
                                showsizes=False)
        else:
            model = ASPP_ResNet(inplanes=16, in_channels=1,num_classes=3)
    elif MODEL == 2:
        if GPUMODE:
            model = UResNet(inplanes=20, input_channels=1, num_classes=3,
                                showsizes=False)
        else:
            model = UResNet(inplanes=20, input_channels=1, num_classes=3)

    # Load checkpoint and state dictionary
    # Map the checkpoint file to the CPU -- removes GPU mapping
    map_location = {"cuda:0":"cpu","cuda:1":"cpu"}
    checkpoint = torch.load( CHECKPOINT_FILE, map_location = map_location )

    best_prec1 = checkpoint["best_prec1"]
    print "state_dict size: ", len(checkpoint["state_dict"])
    print " "
    for p,t in checkpoint["state_dict"].items():
        print p, t.size()

    # Map checkpoint file to the desired GPU
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda(GPUID)

    print "#############################################################"
    print " "
    print "State dictionary mapped to GPU: ",GPUID
    if MODEL == 1:
        modelName = 'ASPP_ResNet'
        modelinfo = 'ASPP_ResNet_Validation_Data_'
    if MODEL == 2:
        modelName = 'caffe_uresnet'
        modelinfo = 'caffe_uresnet_Validation_Data_'
    print "Current model:                  ", modelName
    print " "
    print "Using checkpoint file:          ", CHECKPOINT_FILE
    print " "
    print "Saving to directory:            ", OUTPUT_CVS_DIR
    print "#############################################################"
    print "######### Press return to launch validation routine #########"
    print "#############################################################"
    raw_input()

    # Define loss function (criterion) -- no optimizer for validation
    if GPUMODE:
        criterion = PixelWiseNLLLoss().cuda(GPUID)
    else:
        criterion = PixelWiseNLLLoss()

    ########################
    # Validation Parameters
    ########################
    batchsize_valid        = 5
    iter_per_valid         = 1
    print_freq             = 1
    nbatches_per_itervalid = 2000
    itersize_valid         = (batchsize_valid) * (nbatches_per_itervalid)
    validbatches_per_print = 5
    ########################

    cudnn.benchmark = True

    ##########################################
    # Validation routine configuration string
    ##########################################
    validcfg = """ThreadDatumFillerValid: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: false
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
      EnableMirror: true
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    with open("segfiller_valid.cfg",'w') as fvalid:
        print >> fvalid, validcfg

    iovalid = LArCV1Dataset("ThreadDatumFillerValid","segfiller_valid.cfg" )
    iovalid.init()

    accuracies = []
    for i in range(9):
        accuracies.append(0)

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

            # Get total number of pixels from the validation set
            pixel_bucket = validate(iovalid, batchsize_valid, model, criterion,
                                nbatches_per_itervalid, validbatches_per_print,
                                print_freq)
                                # Data, batchSize, NN_Model, Loss_Func

    #########################################################
    # Compute the final accuracies over all validation images
    #########################################################
    pixel_acc = []
    for i in range(5):
        pixel_acc.append(0)

    # Pixels for non-zero accuracy calculation
    total_corr_non_zero = pixel_bucket[2] + pixel_bucket[4]
    total_non_zero_pix  = pixel_bucket[3] + pixel_bucket[5]

    # Total background accuracy
    pixel_acc[0] = ( float(pixel_bucket[0] / (pixel_bucket[1] + 1.0e-8))
                             * 100 )
    # Total track accuracy
    pixel_acc[1] = ( float(pixel_bucket[2] / (pixel_bucket[3] + 1.0e-8))
                             * 100 )
    # Total shower Accuracy
    pixel_acc[2] = ( float(pixel_bucket[4] / (pixel_bucket[5] + 1.0e-8))
                             * 100 )
    # Non-Zero Pixel Accuracy
    pixel_acc[3] = ( float(total_corr_non_zero / (total_non_zero_pix + 1.0e-8))
                             * 100 )
    # Total pixel accuracy
    pixel_acc[4] = ( float(pixel_bucket[6] / (pixel_bucket[7] + 1.0e-8))
                             * 100 )
    print "FIN"
    print "PROFILER"
    print "#############################################################"
    print "############### Validation Routine Results: #################"
    print "#############################################################"
    print " "
    print "Accuracies computed over total number of pixels"
    print "in the validation set for model checkpoint:"
    print CHECKPOINT_FILE
    print " "
    print "Batch size:                ", batchsize_valid
    print "Number of batches:         ", nbatches_per_itervalid
    print "Images in validation set:  ", (batchsize_valid *
                                          nbatches_per_itervalid)
    print " "
    print "#############################################################"
    print "###############     Total Pixel Counts:      ################"
    print "#############################################################"
    print " "
    print "total_bkgrnd_correct:         ", pixel_bucket[0]
    print "total_bkgrnd_pix:             ", pixel_bucket[1]
    print "total_trk_correct:            ", pixel_bucket[2]
    print "total_trk_pix:                ", pixel_bucket[3]
    print "total_shwr_correct:           ", pixel_bucket[4]
    print "total_shwr_pix:               ", pixel_bucket[5]
    print "total_corr:                   ", pixel_bucket[6]
    print "total_pix:                    ", pixel_bucket[7]
    print " "
    print "#############################################################"
    print "###############         Accuracies:          ################"
    print "#############################################################"
    print " "
    print "Total background accuracy:    ", pixel_acc[0]
    print "Total non-zero accuracy:      ", pixel_acc[3]
    print "Total shower accuracy:        ", pixel_acc[2]
    print "Total track accuracy:         ", pixel_acc[1]
    print "Total accuracy:               ", pixel_acc[4]
    print " "

    # Create CSV file and write results to it
    filename  = OUTPUT_CSV_DIR + modelinfo + '{:%m_%d_%Y}' + '.csv'
    today     = pd.Timestamp('today')

    if os.path.isfile(filename.format(today)):
        flag = 'a'
    else:
        flag = 'w'

    with open(filename.format(today), flag) as vD:
        valWriter = csv.writer(vD, delimiter=',')
        valWriter.writerow([' '])
        valWriter.writerow(['Model:                     ', modelName])
        valWriter.writerow(['Checkpoint File:           ', CHECKPOINT_FILE])
        valWriter.writerow(["Total pixel count:"])
        valWriter.writerow(["total_bkgrnd_correct:      ", pixel_bucket[0]])
        valWriter.writerow(["total_bkgrnd_pix:          ", pixel_bucket[1]])
        valWriter.writerow(["total_trk_correct:         ", pixel_bucket[2]])
        valWriter.writerow(["total_trk_pix:             ", pixel_bucket[3]])
        valWriter.writerow(["total_shwr_correct:        ", pixel_bucket[4]])
        valWriter.writerow(["total_shwr_pix:            ", pixel_bucket[5]])
        valWriter.writerow(["total_corr:                ", pixel_bucket[6]])
        valWriter.writerow(["total_pix:                 ", pixel_bucket[7]])
        valWriter.writerow(["Accuracies:"])
        valWriter.writerow(["Total background accuracy: ", pixel_acc[0]])
        valWriter.writerow(["Total non-zero accuracy:   ", pixel_acc[3]])
        valWriter.writerow(["Total shower accuracy:     ", pixel_acc[2]])
        valWriter.writerow(["Total track accuracy:      ", pixel_acc[1]])
        valWriter.writerow(["Total accuracy:            ", pixel_acc[4]])

##########################################
# Validation function
##########################################
def validate(val_loader, batchsize, model, criterion, nbatches, print_freq,
             iiter):

    # Pixel buckets
    pixel_list     = []
    pixel_bucket   = []
    acc_list       = []

    # Initialize Arrays
    for i in range(0, 8):
        pixel_list.append(0)
        pixel_bucket.append(0)

    # Switch to evaluate mode
    model.eval()
    end = time.time()

    # Validation Loop
    for i in range(0,nbatches):

        data = val_loader.getbatch(batchsize)

        # Convert to pytorch Variable (with automatic gradient calc.)
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda(GPUID))
            labels_var = torch.autograd.Variable(data.labels.cuda(GPUID),
                                                 requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(GPUID),
                                                 requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)

        # Compute output
        output = model(images_var)
        loss   = criterion(output, labels_var, weight_var)

        # Call pixelwise accuracy function
        pixelwise_vec = pixelwise_accuracy(output.data, labels_var.data,
                                            images_var.data)

        # Update pixel bucket
        pixel_bucket[0] += pixelwise_vec[0] # total_bkgrnd_correct
        pixel_bucket[1] += pixelwise_vec[1] # total_bkgrnd_pix
        pixel_bucket[2] += pixelwise_vec[2] # total_trk_correct
        pixel_bucket[3] += pixelwise_vec[3] # total_trk_pix
        pixel_bucket[4] += pixelwise_vec[4] # total_shwr_correct
        pixel_bucket[5] += pixelwise_vec[5] # total_shwr_pix
        pixel_bucket[6] += pixelwise_vec[6] # total_corr
        pixel_bucket[7] += pixelwise_vec[7] # total_pix

        # Print update information to terminal
        print "Pixel_bucket updated..."
        print " "
        print "Iteration: ",i," / ", nbatches
        print " "

        # Debugging block
        # t_np = labels_var
        # t_np = labels_var.data.cpu().numpy()
        #
        # t_np0 = (t_np[0,:,:] == 0)
        # t_np0 = t_np0.astype(np.float32)
        #
        # t_np1 = (t_np[0,:,:] == 1)
        # t_np1 = t_np1.astype(np.float32)
        #
        # t_np2 = (t_np[0,:,:] == 2)
        # t_np2 = t_np2.astype(np.float32)

        # Debugging block
        # cv.imwrite( "Class 2.png", t_np0*1000 )
        # cv.imwrite( "Class 1.png", t_np1*1000 )
        # cv.imwrite( "Class 0.png", t_np2*1000 )
        # print "After CV imwrite: PRESS RETURN:"
        # raw_input()

    return pixel_bucket

##########################################
# Pixelwise accuracy function
##########################################
# Computes accuracy over entire set of
# validation images.
# args: model output tensor
#       truth tensor
#       image labels
# returns: 8 element array of pixel values
##########################################
def pixelwise_accuracy(output, target, imgdata):
    # Aggregate accuracy calculation & accuracies for individual labels
        # Output parameter is the [B, C, H, W] model prediction tensor
        # Target parameter is the [B, H, W] 'truth' tensor
        # imgData parameter contains the corresponding pixel labels

    # Needs to be on the GPU
    maxk        = 1
    batch_size  = target.size(0)                      # Dim 0 -> batch size
    _, pred     = output.max( 1, keepdim = False)     # On gpu
    targetex    = target.resize_( pred.size() ) # Resized to be symmetrical
                                                # with prediction tensor
    correct     = pred.eq( targetex )           # Performs elementwise &&
                                                # Produces symm. tensor
    # Elementwise counts
    num_per_class         = {}
    corr_per_class        = {}
    # Background pixels
    total_bkgrnd_correct  = 0
    total_bkgrnd_pix      = 0
    # Track pixels
    total_trk_correct     = 0
    total_trk_pix         = 0
    # Shower pixels
    total_shwr_correct    = 0
    total_shwr_pix        = 0
    # Totals
    total_corr            = 0
    total_pix             = 0

    # Loop over the classes
    for c in range(output.size(1)):    # range: [0:bkrnd, 1:trk, 2:shwr]
        classmat = targetex.eq(int(c)) # .eq() performs elementwise &&
                                       # for each position c; (0==F, 1==T)

        num_per_class[c]  = classmat.sum() # total predicted class pixels

        corr_per_class[c] = (correct*classmat).sum() # mask by class matrix
                                                     # followed by summation
                                                     # type: tuple {index: int}
        if c  == 0:
            total_bkgrnd_pix     += num_per_class [c]
            total_bkgrnd_correct += corr_per_class[c]
        elif c == 1:
            total_trk_pix        += num_per_class [c]
            total_trk_correct    += corr_per_class[c]
        elif c == 2:
            total_shwr_pix       += num_per_class [c]
            total_shwr_correct   += corr_per_class[c]

        total_pix                += num_per_class [c]
        total_corr               += corr_per_class[c]

    # Result vector
    res = []
    for i in range(0, 8):
        res.append(0)

    # Result assignments
    res[0] = total_bkgrnd_correct
    res[1] = total_bkgrnd_pix
    res[2] = total_trk_correct
    res[3] = total_trk_pix
    res[4] = total_shwr_correct
    res[5] = total_shwr_pix
    res[6] = total_corr
    res[7] = total_pix

    return res

if __name__ == '__main__':
    main()
