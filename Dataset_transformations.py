import numpy as np
from Dataset import Dataset
from sklearn.preprocessing import normalize

class Dataset_transformations(Dataset):

    def __init__(self, items, items_iterator, dims, similarities=None):
        super(Dataset_transformations, self).__init__(
            items, items_iterator, dims, similarities)

    def set_encoded(self, encoded):
        self._encoded_items = encoded

    def set_nnet_output(self, output):
        self._reconstructed_items = output

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
        for j in range(0, self._items.shape[1]):
            mean = self._items[:, j].mean()
            self._items[:, j] = np.subtract(self._items[:, j], mean)
            self._items[:, j] = np.divide(
                self._items[:, j], np.std(self._items[:, j]))
        # print np.min(self._items), np.max(self._items), np.std(self._items)

    def shift(self):
        # self._delta = 1e-8
        self._items = self._items.flatten()
        self._items_min = np.min(self._items)
        self._items = np.add(self._items, -1 * np.min(self._items))
        self._items_max = np.max(self._items)
        self._items = np.divide(self._items, np.max(self._items))
        # self._items = np.add(self._items, self._delta)
        self._items = self._items.reshape(self._x, self._y)
        # print np.min(self._items), np.max(self._items), np.std(self._items)

    def reverse(self):
        self._items = np.subtract(self._items, self._delta)
        self._items = np.multiply(self._items, self._items_max)
        self._items = np.add(self._items, self._items_min)
        self._items = self._items.reshape(self._x, self._y)
        # print np.min(self._items), np.max(self._items), np.std(self._items)

    def conv_process(self, items):
        X_train = np.transpose(self.get_items())
        X_train = X_train.reshape(items.shape[2], items.shape[
                                  0], items.shape[4], items.shape[5])
        X_train = X_train.astype(np.float32)
        X_out = X_train.reshape((X_train.shape[0], -1))
        return X_train,X_out
