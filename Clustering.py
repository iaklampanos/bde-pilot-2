import numpy as np
from netcdf_subset import netCDF_subset
import scipy.stats

class Clustering(object):

    def __init__(self, cluster_dict, dataset_dict, cdict_fields, datadict_fields):
        self._cdict_fields = cdict_fields
        self._datadict_fields = datadict_fields
        if not(isinstance(cluster_dict, dict)) or not(isinstance(dataset_dict, dict)):
            raise TypeError('KClustering object needs 2 dictionaries, ' +
                            'Cluster_dictionary:{n_clusters,normalize,season,therm_season,' +
                            'multilevel,size_desc,size_div} and Dataset_dictionary')
        for cfield in self._cdict_fields:
            if cfield not in cluster_dict:
                raise ValueError(
                    cfield + ' is needed in the cluster dictionary')
        for dfield in self._datadict_fields:
            if dfield not in dataset_dict:
                raise ValueError(dfield + ' is needed in dataset dictionary')
        self._netcdf_subset = netCDF_subset(dataset_dict['dataset'],
                                            dataset_dict['levels'],
                                            dataset_dict['sub_vars'],
                                            dataset_dict['lvlname'],
                                            dataset_dict['timename'],
                                            dataset_dict['time_unit'],
                                            dataset_dict['time_cal'],
                                            dataset_dict['ncar_lvls'])
        # Number of clusters
        self._n_clusters = cluster_dict['n_clusters']
        # Normalize data switch
        self._normalize = cluster_dict['normalize']
        # Use season divided dataset(winter,spring,summer,autumn)
        self._season = cluster_dict['season']
        # Use binary season dataset (hot/cold)
        self._therm_season = cluster_dict['therm_season']
        # Use multiple levels for clustering
        self._multilevel = cluster_dict['multilevel']
        # Hourslots used for cluster descriptors
        self._size_desc = cluster_dict['size_desc']
        # Number that divides size_desc for cluster descriptor output
        # timeframes
        self._size_div = cluster_dict['size_div']

    # Method used for link_var/multivar AND saved linkages
    def get_clut_list(self, V):
        clut_list = []
        clut_indices = []
        # create list of indices for every cluster
        for nc in range(0, self._n_clusters):
            clut_indices.append(np.where(V == nc)[0])
        print 'Clustering distirbution'
        print '---------------------'
        clut_list.append(clut_indices)
        for pos, c in enumerate(clut_list):
            obv_dev = []
            for nc in range(0, self._n_clusters):
                obv_dev.append((nc, len(c[nc])))
            sort_obd_dev = sorted(obv_dev, key=lambda x: x[1], reverse=True)
            print sort_obd_dev
        return clut_list, V, sort_obd_dev

    # Preprocessing netcdf data before clustering
    def preprocess_multivar(self, var_list):
        # place holder for ndarrays
        temp_v_list = []
        for pos, v in enumerate(var_list):
            # create ndarray
            temp_v_list.append(np.ndarray(
                shape=(v.shape[0], v[0][:].flatten().shape[0])))
            # flatten the grid
            for i in range(0, v.shape[0]):
                temp_v_list[pos][i] = v[i][:].flatten()
        gather_data = np.concatenate(temp_v_list)
        # create placeholder for ndarray(total_hourslots,(V1,V2...Vn))
        uv = np.ndarray(shape=(gather_data.shape[
                        0] / len(var_list), gather_data.shape[1] * len(var_list)))
        # populate placeholder
        for pos, idx in enumerate(uv):
            iters = []
            iters.append(pos)
            for it in range(1, len(var_list)):
                iters.append(iters[it - 1] + uv.shape[0])
            uv[pos] = gather_data[iters].flatten()
        print uv.shape
        del gather_data
        return uv

    # Create a season map and divide the dataset into seasons
    def get_seasons(self, times, season_ch):
        seasons_idx = []
        for t in times:
            if t.month == 12 or t.month == 1 or t.month == 2:
                seasons_idx.append('winter')
            elif t.month == 3 or t.month == 4 or t.month == 5:
                seasons_idx.append('spring')
            elif t.month == 6 or t.month == 7 or t.month == 8:
                seasons_idx.append('summer')
            elif t.month == 9 or t.month == 10 or t.month == 11:
                seasons_idx.append('autumn')
        winter_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'winter']
        spring_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'spring']
        summer_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'summer']
        autumn_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'autumn']
        if season_ch == 'winter':
            return winter_idx
        elif season_ch == 'summer':
            return summer_idx
        elif season_ch == 'spring':
            return spring_idx
        elif season_ch == 'autumn':
            return autumn_idx

    # Create season map and divide dataset into cold and hot seasons
    def get_biseasons(self, times, season_therm):
        seasons_idx = []
        cold_d = [9, 10, 11, 12, 1, 2]
        hot_d = [3, 4, 5, 6, 7, 8]
        for t in times:
            if t.month in cold_d:
                seasons_idx.append('cold')
            elif t.month in hot_d:
                seasons_idx.append('hot')
        cold_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'cold']
        hot_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'hot']
        if season_therm == 'cold':
            return cold_idx
        elif season_therm == 'hot':
            return hot_idx

    def single_cluster_tofile(self, out_path, cluster_label, clut_list):
        for pos, c in enumerate(clut_list):
            print 'Creating file for Variable ', self._netcdf_subset._subset_variables[pos]
            print 'Cluster label is ', cluster_label
            self._netcdf_subset.write_timetofile(out_path +
                                                 '/var_' + self._netcdf_subset._subset_variables[pos] +
                                                 '_cluster' +
                                                 str(cluster_label) + '.nc',
                                                 self._netcdf_subset.lvl_pos(), c[cluster_label])

    def multi_cluster_tofile(self, out_path, cluster_label, clut_list):
        for pos, c in enumerate(clut_list):
            print 'Creating file for mixed variables. Cluster label is ', cluster_label
            self._netcdf_subset.write_timetofile(out_path +
                                                 '/var_mixed_cluster' +
                                                 str(cluster_label) + '.nc',
                                                 self._netcdf_subset.lvl_pos(), c[cluster_label])

    def cluster_descriptor_max(self, out_path, max_ret_list):
        for pos, c in enumerate(max_ret_list):
            self._netcdf_subset.write_timetofile(out_path + '/cluster_descriptor_meanmax'
                                                 + str(pos) + '.nc', self._netcdf_subset.lvl_pos(), range(c[0], c[1]),
                                                 c_desc=True)

    #From this function and now on we need the original dataset and not the modified one
    def middle_cluster_tofile(self, out_path, max_ret_list):
        for pos, c in enumerate(max_ret_list):
            mid_start_plus = (c[len(c) - 1] - c[0] + 1) / \
                2 - self._size_desc / 2
            mid_start = c[0] + mid_start_plus
            self._netcdf_subset.exact_copy_file(out_path +
                                                '/cluster_descriptor' +
                                                str(pos) + '.nc',
                                                range(mid_start, mid_start + self._size_desc))

    def cluster_descriptor_middle(self, out_path, max_ret_list):
        for pos, c in enumerate(max_ret_list):
            if (c[1] - c[0]) >= self._size_desc:
                self._netcdf_subset.exact_copy_mean(out_path +
                                                    '/cluster_descriptor_meanmiddle' + str(pos) +
                                                    '.nc', range(c[0], c[0] + self._size_desc), self._size_desc, self._size_div)
            else:
                self._netcdf_subset.exact_copy_mean(out_path +
                                                    '/cluster_descriptor_meanmiddle' + str(pos) +
                                                    '.nc', range(c[0], c[1]), len(range(c[0],c[1])), self._size_div)

    def euc_dist(self, start_date, end_date, clut_list):
        if len(clut_list) != 1:
            raise ValueError('List of clusters must contain only a single variable '
                             + 'or a single list for multiple variables')
        times = self._netcdf_subset.get_times()
        times = times.tolist()
        t1_pos = times.index(start_date)
        t2_pos = times.index(end_date)
        z_case = self._netcdf_subset.extract_timeslotdata(
            range(t1_pos, t2_pos), self.lvl_pos())
        z_case = np.mean(z_case[0], axis=0)
        ec_dist = []
        for pos, c in enumerate(clut_list[0]):
            mid_start_plus = (c[len(c) - 1] - c[0] + 1) / \
                2 - self._size_desc / 2
            mid_start = c[0] + mid_start_plus
            z_c = self._netcdf_subset.extract_timeslotdata(
                range(mid_start, mid_start + self._size_desc), self.lvl_pos())
            z_c = np.mean(z_c[0], axis=0)
            ec_dist.append(np.linalg.norm(z_case - z_c))
        return ec_dist

    def kl_divergence(self,current_weather,cfile_list,var_list):
        kl = []
        for c in cfile_list:
            self._netcdf_subset = c
            cv = preprocess_multivar(var_list)
            cv_sum = np.sum(cv)
            cv = np.divide(cv,cv_sum)
            curr_sum = np.sum(current_weather)
            current_weather = np.divide(current_weather,curr_sum)
            kl.append(scipy.stats.entropy(current_weather, cv))
        return kl
