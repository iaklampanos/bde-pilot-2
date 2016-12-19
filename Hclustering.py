import numpy as np
from sklearn.cluster import AgglomerativeClustering
from netcdf_subset import netCDF_subset
from Clustering import Clustering

class Hclustering(Clustering):

    def __init__(self, cluster_dict, dataset_dict):
        cdict_fields = ['n_clusters', 'normalize', 'season', 'affinity', 'method',
                        'therm_season', 'multilevel', 'size_desc', 'size_div']
        datadict_fields = ['dataset', 'levels', 'sub_vars',
                           'lvlname', 'timename', 'time_unit', 'time_cal', 'ncar_lvls']
        Clustering.__init__(self, cluster_dict, dataset_dict,
                            cdict_fields, datadict_fields)
        self._affinity = cluster_dict['affinity']
        self._method = cluster_dict['method']

    # Perform clustering and retrieve dataset clustered in n_clusters (every
    # variable individually clustered, if you want to cluster multiple
    # variables you should use link_multivar)
    def link_var(self, time_idx=None):
        # Check pressure levels
        if len(self._netcdf_subset._pressure_levels) != 1:
            if not(self._multilevel):
                raise ValueError('Multilevel is disabled, if you want multilevel '
                                 + 'clustering then re run with multilevel=True')
        # get data from netcdf subset
        if time_idx is None:
            var_list = self._netcdf_subset.extract_data(
                self._netcdf_subset.lvl_pos())
        else:
            var_list = self._netcdf_subset.extract_timeslotdata(time_idx,
                                                                self._netcdf_subset.lvl_pos())
        for v in var_list:
            # create place holder variable where the grid is flattened
            var_data = np.ndarray(
                shape=(v.shape[0], v[0][:].flatten().shape[0]))
            for i in range(0, v.shape[0]):
                var_data[i] = v[i][:].flatten()
            print var_data.shape
            # for normalization purposes we get the column mean and subtract it
            if self._normalize:
                for j in range(0, var_data.shape[1]):
                    mean = var_data[:, j].mean()
                    var_data[:, j] = np.subtract(var_data[:, j], mean)
            # perform parallel Kmeans clustering
            V = AgglomerativeClustering(n_clusters=self._n_clusters,
                                        affinity=self._affinity, linkage=self._method).fit(var_data).labels_
        return self.get_clut_list(V)

    # Perform clustering and retrieve dataset clustered in n_clusters (for
    # multiple variables)
    def link_multivar(self, time_idx=None):
        # Check pressure levels
        if len(self._netcdf_subset._pressure_levels) != 1:
            if not(self._multilevel):
                raise ValueError('Multilevel is disabled, if you want multilevel '
                                 + 'clustering then re run with multilevel=True')
        if time_idx is None:
            var_list = self._netcdf_subset.extract_data(
                self._netcdf_subset.lvl_pos())
        else:
            var_list = self._netcdf_subset.extract_timeslotdata(time_idx,
                                                                self._netcdf_subset.lvl_pos())
        uv = self.preprocess_multivar(var_list)
        # for normalization purposes we get the column mean and subtract it
        if self._normalize:
            for j in range(0, uv.shape[1]):
                mean = uv[:, j].mean()
                uv[:, j] = np.subtract(uv[:, j], mean)
        UV = AgglomerativeClustering(n_clusters=self._n_clusters,
                                     affinity=self._affinity, linkage=self._method).fit(uv).labels_
        return self.get_clut_list(UV)

    # Clustering for certain season (i.e winter,spring,summer,autumn)
    def link_4svar(self):
        times = self._netcdf_subset.get_times()
        time_idx = self.get_seasons(times, self._season)
        return self.link_var(time_idx=time_idx)

    # Clustering for certain biseason (hot/cold)
    def link_2svar(self):
        times = self._netcdf_subset.get_times()
        time_idx = self.get_biseasons(times, self._therm_season)
        return self.link_var(time_idx=time_idx)

    # Clustering for certain season (i.e winter,spring,summer,autumn)
    # for multiple variables
    def link_multi4svar(self):
        times = self._netcdf_subset.get_times()
        time_idx = self.get_seasons(times, self._season)
        return self.link_multivar(time_idx=time_idx)

    # Clustering for certain biseason (hot/cold) for multiple variables
    def link_multi2svar(self):
        times = self._netcdf_subset.get_times()
        time_idx = self.get_biseasons(times, self._therm_season)
        return self.link_multivar(time_idx=time_idx)
