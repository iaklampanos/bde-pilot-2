from netcdf_subset import netCDF_subset
import numpy as np

class Meta_clustering(object):

    def __init__(self, dataset_dict, cluster_dict):
        datadict_fields = ['dataset', 'levels', 'sub_vars',
                           'lvlname', 'timename', 'time_unit', 'time_cal', 'ncar_lvls']
        cdict_fields = ['normalize', 'multilevel']
        for dfield in datadict_fields:
            if dfield not in dataset_dict:
                raise ValueError(dfield + ' is needed in dataset dictionary')
        for cfield in cdict_fields:
            if cfield not in cluster_dict:
                raise ValueError(
                    cfield + ' is needed in the cluster dictionary')
        self._netcdf_subset = netCDF_subset(dataset_dict['dataset'],
                                            dataset_dict['levels'],
                                            dataset_dict['sub_vars'],
                                            dataset_dict['lvlname'],
                                            dataset_dict['timename'],
                                            dataset_dict['time_unit'],
                                            dataset_dict['time_cal'],
                                            dataset_dict['ncar_lvls'])
        self._normalize = cluster_dict['normalize']
        self._multilevel = cluster_dict['multilevel']

    def prepare_c_list_for_metrics(self, clut_list):
        if (len(clut_list) != 1):
             raise ValueError('List of clusters must contain only a single ' +
                            'variable or a single list for multiple variables')
        ret_list = []
        reshape_dim = 0
        if self._multilevel:
            for lvl in self._netcdf_subset._pressure_levels:
                for var in self._netcdf_subset._subset_variables:
                    reshape_dim += self._netcdf_subset._dataset[var].shape[2]* \
                    self._netcdf_subset._dataset[var].shape[3]
        else:
             for var in self._netcdf_subset._subset_variables:
                 reshape_dim += self._netcdf_subset._dataset[var].shape[2]* \
                 self._netcdf_subset._dataset[var].shape[3]
        for c in clut_list[0]:
             var_list = self._netcdf_subset.extract_timeslotdata(c.tolist(),
                                                 self._netcdf_subset.lvl_pos())
             temp_arr = np.array(0)
             for v in var_list:
                 temp_arr = np.append(temp_arr,v)
             temp_arr = np.delete(temp_arr,0)
             temp_arr = temp_arr.reshape(len(c),reshape_dim)
             if self._normalize:
                for j in range(0,temp_arr.shape[1]):
                    mean = temp_arr[:,j].mean()
                    temp_arr[:,j] = np.subtract(temp_arr[:,j],mean)
             ret_list.append(temp_arr)
        return ret_list

    # Calculate precision,recall and F-scores for two clustering outcomes
    def calculate_prf(clut_list1,clut_list2):
        if len(clut_list1)!=1 and len(clust_list2)!=1:
              raise ValueError('List of clusters must contain only a single ' +
                             'variable or a single list for multiple variables')
        P = []
        R = []
        F = []
        for c in clut_list1[0]:
              C_precision = []
              C_recall = []
              C_f = []
              for cc in clut_list2[0]:
                  p = len(np.intersect1d(c,cc))/float(len(cc))
                  r = len(np.intersect1d(c,cc))/float(len(c))
                  if len(np.intersect1d(c,cc)) != 0:
                      C_precision.append(p)
                      C_recall.append(r)
                      C_f.append(2*p*r/(p+r))
                  else:
                      C_precision.append(0)
                      C_recall.append(0)
                      C_f.append(0)
              P.append(C_precision)
              R.append(C_recall)
              F.append(C_f)
        return P,R,F

    # Calculate Cohesion(WSS) by Sum squared error and Separation
    # measured by between cluster sum of squares
    def calculate_clut_metrics(self,cluster_data,normalize=False):
        wss_c = []
        bss_c = []
        mag = []
        if normalize:
            for cluster in cluster_data:
                mag.append(np.linalg.norm(cluster))
            max_mag = max(mag)
            cluster_norm = []
            for cluster in cluster_data:
                cluster_norm.append(np.divide(cluster,max_mag))
        flatten_clust = np.array([0])
        if normalize:
            for c in cluster_norm:
                flatten_clust = np.append(flatten_clust,c)
        else:
            for c in cluster_data:
                flatten_clust = np.append(flatten_clust,c)
        flatten_clust = np.delete(flatten_clust,0)
        m = np.mean(flatten_clust,axis=0)
        if normalize:
           for cluster in cluster_norm:
                mi = np.mean(np.matrix(cluster),axis=0)
                wss = np.power((np.subtract(cluster,mi)),2)
                bss = np.multiply(len(cluster),np.power(np.subtract(m,mi),2))
                wss_c.append(np.divide(np.sum(wss),len(cluster)))
                bss_c.append(np.divide(np.sum(bss),len(cluster)))
        else:
            for cluster in cluster_data:
                mi = np.mean(np.matrix(cluster),axis=0)
                wss = np.power((np.subtract(cluster,mi)),2)
                bss = np.multiply(len(cluster),np.power(np.subtract(m,mi),2))
                wss_c.append(np.sum(wss))
                bss_c.append(np.sum(bss))
        WSS = np.sum(wss_c)
        BSS = np.sum(bss_c)
        TOTAL = WSS+BSS
        return WSS,BSS,TOTAL
