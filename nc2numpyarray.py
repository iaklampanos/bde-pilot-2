#!/usr/bin/env python

import sys
from netCDF4 import Dataset
import os
import numpy as np


def extract_np_array(ncfilename):
    dataset = Dataset(ncfilename, 'r')
    c137 = dataset.variables['C137'][:]
    i131 = dataset.variables['I131'][:]
    c137 = np.sum(c137, axis=0).reshape(501, 501)
    i131 = np.sum(i131, axis=0).reshape(501, 501)
    return {'c137': c137, 'i131': i131}

def main():
    for l in sys.argv[1:]:
        os.system('bunzip2 -k ' + l)
        ncfilename = l[:l.rfind('.')]
        print ncfilename

        nparr = extract_np_array(ncfilename)
        for s in nparr:
            npyfilename = ncfilename[:ncfilename.rfind('.')] + '-' + s + '.npy'
            np.save(npyfilename, nparr[s])

        os.system('rm ' + ncfilename)
    

if __name__ == "__main__":
    main()
