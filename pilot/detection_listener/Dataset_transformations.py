"""
   CLASS INFO
   -------------------------------------------------------------------------------------------
     Dataset_transformations contains every method that has to do with processing or altering a dataset's structure. Regularyl used in netCDF data for easy transformation of N-dim arrays to 2D and normalization.
   -------------------------------------------------------------------------------------------
"""
import numpy as np
from Dataset import Dataset
from sklearn.preprocessing import minmax_scale

class Dataset_transformations(Dataset):

    def __init__(self, items, items_iterator, similarities=None):
        super(Dataset_transformations, self).__init__(
            items, items_iterator, similarities)

    def twod_transformation(self):
        # Check if every netcdf subset timeslots are the same
        timeslot_len = []
        for nc in self._items:
            timeslot_len.append(nc.shape[1])
        if not(all(map(lambda x: x == timeslot_len[0], timeslot_len))):
            raise ValueError(
                'Item subsets must have the same time length')
        # If every netcdf has the same timeslot length
        # reshape every variable into (timeslot,level*lat*lon)
        var_list = []
        timeslot_len = timeslot_len[0]
        for nc in self._items:
            var_num = nc.shape[0]
            timeslot_num = nc.shape[1]
            level_num = nc.shape[2]
            lat_num = nc.shape[3]
            lon_num = nc.shape[4]
            if var_num > 1:
                for var in range(0, var_num):
                    var_list.append(nc[var, :].reshape(
                        timeslot_num, level_num * lat_num * lon_num))
            else:
                var_list.append(nc.reshape(
                    timeslot_num, level_num * lat_num * lon_num))
        # create a big table of [timeslots,features]
        appended_shape = 0
        for v in var_list:
            appended_shape += v.shape[1]
        appended = np.zeros(shape=(timeslot_len, 1))
        for v in var_list:
            appended = np.concatenate((appended, v), axis=1)
        appended = np.delete(appended, 0, axis=1)
        self._items = np.transpose(appended)
        self._x = self._items.shape[0]
        self._y = self._items.shape[1]

    def normalize(self):
        # self._items = minmax_scale(self._items)
        for j in range(0, self._items.shape[1]):
            mean = self._items[:, j].mean()
            self._items[:, j] = np.subtract(self._items[:, j], mean)
            self._items[:, j] = np.divide(
                self._items[:, j], np.sqrt(np.var(self._items[:, j])+10))
        # print np.min(self._items), np.max(self._items), np.std(self._items)
