#!/usr/bin/env python

"""
Create a training set for supervised learning based on convnets.
This presumes a c147.npz-type data file already exists, as it is
created by corr_disp2weather.py.
"""

import sys
import os
import numpy as np
from scipy.misc import imresize
from datetime import datetime
from disputil import display_array
from numpy.random import rand

DATAFILE = 'c137_v2.npz'
SPECIES = 'c137'
NUM_OF_SAMPLES = 10  # number of random-noise samples per original sample
TARGET_DISP_DIMS = (167, 167)
WEATHER_DIST = 0.1  # the weather distortion percentage
DISPER_DIST = 0.1   # the dispersion distortion percentage


def log(s, label='INFO', metadata=True):
    if metadata:
        sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' +
                         str(s) + '\n')
    else:
        sys.stdout.write(str(s) + '\n')
    sys.stdout.flush()


def main():
    # load the base training data
    log('Loading ' + DATAFILE + '...')
    c137 = np.load(DATAFILE)[SPECIES]
    log('Done.')
    training_set = []

    for si, origsmpl in enumerate(c137):
        dispersion = origsmpl[3]  #.reshape((501, 501))
        newdispersion = imresize(dispersion, TARGET_DISP_DIMS, mode='F')
        # newdispersion = newdispersion.reshape(
        #                             TARGET_DISP_DIMS[0]*TARGET_DISP_DIMS[1])
        weather = origsmpl[4]
        origin = origsmpl[0]
        origin_index = origsmpl[1]
        thedate = origsmpl[2]
        
        for i in range(NUM_OF_SAMPLES):
            rw = (rand(3, 3, 64, 64) - 0.5) * 2.0 * WEATHER_DIST
            rd = (rand(167, 167) - 0.5) * 2.0 * DISPER_DIST
            w = weather + rw * weather
            d = newdispersion
            nzidx = d.nonzero()
            d[nzidx] = d[nzidx] + rd[nzidx] * d[nzidx]
            display_array(weather[2][1])
            display_array(w[2][1])
            display_array(newdispersion)
            display_array(d)
            break
        break
            # tuple = [origin, origin_index, thedate, d, w]
            #
            # training_set.append(tuple)
        
        log('Finished sample ' + str(si) + '/' + str(len(c137)))
    training_set = np.array(training_set)
    log('Saving...')
    np.save('supervised_training_c137', training_set)
    log('Done.')

if __name__ == '__main__':
    main()
