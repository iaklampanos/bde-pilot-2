"""
   CLASS INFO
   -------------------------------------------------------------------------------------------
     Detection contains methods that have to do with manipulating and using (lat,lon) detection
     points and comparing them with the dispersion of various stations in order to generate a score
     of each one, leading to estimating the source.
   -------------------------------------------------------------------------------------------
"""

from Dataset_transformations import Dataset_transformations
import numpy as np
import dataset_utils as utils
from netCDF4 import Dataset
from scipy.ndimage.filters import gaussian_filter
import scipy
from sklearn.preprocessing import maxabs_scale

RESIZE_DIM = 167

class Detection(object):

    def __init__(self, dispersion, filelat, filelon, llat, llon):
        self._conc = dispersion # Dispersion grid
        self._filelat = filelat # "Standard" latitude range of WRF
        self._filelon = filelon # "Standard" longitude range of WRF
        self._lats = llat # Detection points latitude
        self._lons = llon # Detection points longitude

    def get_indices(self):
        # Creates (lat,lon) detection points
        lat_idx = []
        lon_idx = []
        # For each point latitude from UI map find closest actual latitude
        for lat in self._lats:
            lat_idx.append(np.argmin(np.abs(self._filelat - lat)))
        # For each point longitude from UI map find closest actual longitude
        for lon in self._lons:
            lon_idx.append(np.argmin(np.abs(self._filelon - lon)))
        self._lat_idx = lat_idx
        self._lon_idx = lon_idx

    def create_detection_map(self,resize=False):
        # Creates detection grid which is a (x,y) grid full of zeros, except where the
        # Detection points are.
        pollutant_array = self._conc
        # Initialize grid
        det_map = np.zeros(shape=(pollutant_array.shape[0] ,pollutant_array.shape[1]))
        readings = [(self._lat_idx[k],self._lon_idx[k]) for k,i in enumerate(self._lat_idx)]
        # Mark detection points
        for r in readings:
                det_map[r] = 1
        # Apply filter for better estimation
        det_map = gaussian_filter(det_map,0.3)
        if resize:
            det_map = scipy.misc.imresize(det_map, (RESIZE_DIM, RESIZE_DIM))
        # Scale
        det_map = maxabs_scale(det_map)
        self._det_map = det_map

    def calc(self):
        # Finds whether detection points and dispersion are overlapping
        # if there is no overlap then score = 0
        nonzero_det = np.nonzero(self._det_map)
        nonzero_points = [(nonzero_det[0][i],nonzero_det[1][i]) for i in range(0,len(nonzero_det[0]))]
        score = 0
        for i in range(0,len(nonzero_points)):
            score += self._conc[nonzero_points[i]]
        return score

    # def KL(self):
    #     nonzero_conc = np.nonzero(self._conc)
    #     nonzero_points = [(nonzero_conc[0][i],nonzero_conc[1][i]) for i in range(0,len(nonzero_conc[0]))]
    #     det = []
    #     for i in range(0,len(nonzero_points)):
    #         det.append(self._det_map[nonzero_points[i]])
    #     det = np.add(self._det_map, 1e-12)
    #     conc = np.add(self._conc, 1e-12)
    #     return scipy.stats.entropy(conc.flatten(),det.flatten())

    def cosine(self):
        # Calculate cosine distance
        det = self._det_map
        conc = maxabs_scale(self._conc)
        return scipy.spatial.distance.cosine(conc.flatten(),det.flatten())
