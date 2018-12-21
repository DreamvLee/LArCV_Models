#
# filename: tb_plot.py
# purpose:  This script will iterate over a set of CSV files in a given folder
#           (presumed to be TensorBoard output) and generate high resolution
#           plots with labeled axes. Data shape is expected to be [1001, 3].
#           The csv headers are skipped and only columns 1 and 2 are loaded
#           into a numpy array.
# author:   Kai Kharpertian
# dept:     Tufts University - Department of Physics
# date:     December 21, 2018
#
###############################
# Import Script
###############################
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob

###############################
# Plot Data
###############################
def tb_data_plot(inputDir, path):
    """ Iterates through all csv's in folder given by 'path'
        and generates 300 dpi plots with labeled axes. File name
        generated as a substring from the input csv file name. """
    for fileName in glob.glob(path):
        tBFile  = fileName[-50:]
        outFile = inputDir + tBFile
        x = np.loadtxt(open(outFile, 'r'), delimiter = ",", skiprows = (1), usecols = (1, 2))
        fig   = plt.figure()
        ax    = fig.add_subplot(111)
        title = 'Run ID: ' + tBFile[4:15] + ' ' + tBFile[36:46]
        ax.set_title(title)
        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Loss / Accuracy')
        plt.plot(x[:,0], x[:,1])
        saveFig = inputDir + '_' + title + '.png'
        plt.savefig(saveFig, dpi=300)

###############################
# Main Function
###############################
def main():
    inputDir = input("Provide file path or drag in folder from GUI:")
    inputDir = inputDir.rstrip() + '/'
    path     = inputDir + '*.csv'
    tb_data_plot(inputDir, path)

if __name__ == '__main__':
    main()
