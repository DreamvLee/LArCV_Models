import matplotlib.pyplot as plt
import numpy as np
import csv
import glob

### Uncomment to strip headers from CSV files ###
# TODO: Convert this into a function?

# File Paths
inputDir = '/Users/KsEuro/Dropbox/Academics/Research/deeplearnphysics/\
pytorch/fall_2018/caffe_uresnet/training_set3_fa18/loss_curves/'
outDir  = inputDir
path    = inputDir + '*.csv'

for fileName in glob.glob(path):
    with open(fileName, 'rb') as input_:
        reader = csv.reader(input_)
        tBFile  = fileName[-50:]
        outFile = inputDir + 'edit_' + tBFile
        with open(outFile, 'wb') as output_:
            writer = csv.writer(output_, delimiter=',')
            next(reader) # skip col headers
            for row in reader:
                writer.writerow(row)

### Plot data from CSV files using numpy ###
# TODO: Finish this portion of the script
