from Dataset_transformations import Dataset_transformations
import numpy as np
import dataset_utils as utils
from netCDF4 import Dataset
from scipy.ndimage.filters import gaussian_filter
import scipy

class Detection(object):

    def __init__(self, dispersion, filelat, filelon, llat, llon):
        self._conc = dispersion
        self._filelat = filelat
        self._filelon = filelon
        self._lats = llat
        self._lons = llon

    def get_indices(self):
        lat_idx = []
        lon_idx = []
        for lat in self._lats:
            lat_idx.append(np.argmin(np.abs(self._filelat - lat)))
        for lon in self._lons:
            lon_idx.append(np.argmin(np.abs(self._filelon - lon)))
        self._lat_idx = lat_idx
        self._lon_idx = lon_idx

    def create_detection_map(self):
        pollutant_array = self._conc
        det_map = np.zeros(shape=(pollutant_array.shape[0] ,pollutant_array.shape[1]))
        for lat in self._lat_idx:
            for lon in self._lon_idx:
                det_map[lat,lon] = 1
        det_map = gaussian_filter(det_map,0.3)
        self._det_map = det_map

    def calc(self):
        nonzero_det = np.nonzero(self._det_map)
        nonzero_points = [(nonzero_det[0][i],nonzero_det[1][i]) for i in range(0,len(nonzero_det[0]))]
        score = 0
        for i in range(0,len(nonzero_points)):
            score += self._conc[nonzero_points[i]]
        return score

    def KL(self):
        det = np.add(self._det_map,1e-12)
        conc = np.add(self._conc,1e-12)
        return scipy.stats.entropy(det.flatten(),conc.flatten())

    def cosine(self):
        conc = []
        nonzero_det = np.nonzero(self._det_map)
        nonzero_points = [(nonzero_det[0][i],nonzero_det[1][i]) for i in range(0,len(nonzero_det[0]))]
        score = 0
        for i in range(0,len(nonzero_points)):
            conc.append(self._conc[nonzero_points[i]])
        conc = np.add(conc,1e-12)
        detnon = np.add(self._det_map[np.nonzero(self._det_map)],1e-12)
        return scipy.spatial.distance.cosine(conc.flatten(),detnon.flatten())
