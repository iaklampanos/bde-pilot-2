import numpy as np
from Dataset import Dataset
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn import metrics
import dataset_utils as utils
# import oct2py
from sklearn.neighbors.kde import KernelDensity
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()
    
class Clustering(Dataset):

    def __init__(self, dataset, n_clusters, n_init,
                 max_iter=300, features_first=False, similarities=None):
        super(Clustering, self).__init__(
            dataset.get_items(), dataset.get_items_iterator(),
            dataset.get_dims(), dataset.get_similarities())
        self._features_first = features_first
        data = self.get_items()
        if self._features_first:
            self._items = np.transpose(self.get_items())
        self._n_clusters = n_clusters
        self._n_init = n_init
        self._max_iter = max_iter
        self._clustering_dist = None
        self._labels = None
        self._centroids = None
        self._index_list = None  # result of k-means ( list of lists of int )
        self._link = None

    def kmeans(self):
        data = self.get_items()
        # print data.shape
        self._link = KMeans(n_clusters=self._n_clusters, n_init=self._n_init,
                            max_iter=self._max_iter, n_jobs=-1).fit(data)
        self._labels = self._link.labels_
        self._centroids = self._link.cluster_centers_
        self.get_clut_list(self._labels)

    def batch_kmeans(self, max_no_imprv):
        self._max_no_imprv = max_no_imprv
        data = self.get_items()
        init_size = self.get_items_iterator()
        self._link = MiniBatchKMeans(n_clusters=self._n_clusters, init_size=init_size, n_init=self._n_init,
                                     max_iter=self._max_iter, max_no_improvement=self._max_no_imprv).fit(data)
        self._labels = self._link.labels_
        self._centroids = self._link.cluster_centers_
        self.get_clut_list(self._labels)

    def hierachical(self, affinity, linkage):
        self._affinity = affinity
        self._linkage = linkage
        data = self.get_items()
        self._link = AgglomerativeClustering(
            n_clusters=self._n_clusters, affinity=self._affinity, linkage=self._linkage).fit(data)
        self._labels = self._link.labels_
        self._leaves = self._link.leaves_
        self.get_clut_list(self._labels)

    # TODO: Change name - refactor callers
    def create_descriptors(self, frames):
        data = self.get_items()
        clut_list = self._index_list[0]
        c_desc = []
        for c in clut_list:
            cluster_data = data[c]
            kj = KMeans(n_clusters=frames, n_init=self._n_init,
                        max_iter=self._max_iter, n_jobs=-1).fit(cluster_data).labels_
            avg = []
            for j in range(0, frames):
                idx = [idx for idx, frame in enumerate(
                    kj) if frame == j]
                avg.append(c[idx])
            c_desc.append(avg)
        self._descriptors = c_desc
    
    # TODO: Change name - see comment above
    def create_kmeans_descriptors(self, frames):
        create_descriptors(self, frames)
    
    def create_density_descriptors(self, frames, times):
        """
        Create max-density cluster descriptors.
        Params:
        times: list of datetime objects
        frames: int - the number of 6hr-snapshots required
        """
        bwdth = 32*2
        snap_duration_hrs = 6
        # Transpose all dates to a fixed hypothetical year and calculate hour offsets
        refdate = datetime(2020, 1, 1)
        times_f = np.array([ int((d.replace(year=2020) - 
                           refdate).total_seconds() / 3600.0) for d in times ])
        data = self.get_items() # samples, features
        clusters = self._index_list[0]
        c_descriptors = []  # the cluster descriptors

        frames_filled = frames_total = 0
        # print len(times), len(times_f), len(data), np.max(times_f)
        for c in clusters:
            c_desc = [] # the descriptor of the current cluster
            indexes = np.array([x for x in c])
            cdata = data[indexes]
            ctimes_f = times_f[indexes]
            X_plot = np.linspace(0, times_f[-1], times_f[-1])[:, np.newaxis]
            kde = KernelDensity(kernel='gaussian', bandwidth=bwdth).fit(
                ctimes_f[:, np.newaxis])
            dens = np.exp(kde.score_samples(X_plot))
            
            # Choose the descriptor around the point where density is max
            max_den_i = np.argmax(dens)

            # find positions of interest
            cent_pos = max_den_i - (max_den_i % snap_duration_hrs)
            if max_den_i % snap_duration_hrs > snap_duration_hrs / 2:
                cent_pos += snap_duration_hrs
            start_time_offset = cent_pos - (frames / 2) * snap_duration_hrs  ##
            end_time_offset = start_time_offset + frames * snap_duration_hrs ##
            
            pos = []  # list of time offsets
            for k in range(frames):
                frames_total += 1
                pos.append(start_time_offset + k * snap_duration_hrs)
                cindices = np.where(np.in1d(ctimes_f, pos[k]))[0] # indices in cluster data list where the offsets occur - it may be []
                
                if len(cindices) > 0:
                    gindices = indexes[cindices]
                    # print times[indexes[cindices]]  # checking real times
                    c_desc.append(list(gindices))
                    # c_desc.append(np.mean(cdata[cindices], 0))           # ***
                else:
                    c_desc.append(None)
            
            # Deal with None by duplicating neighbouring snapshots (shouldn't occur often...)
            for k in range(frames):
                if c_desc[k] is None and k > 0:
                    c_desc[k] = c_desc[k-1]
                    if c_desc[k] is not None: frames_filled += 1
            for k in reversed(range(frames)):
                if c_desc[k] is None and k < frames - 1:
                    c_desc[k] = c_desc[k+1]
                    if c_desc[k] is not None: frames_filled += 1
            
            # for displaying data - need to uncommend *** above
            # from disputil import display_array
            # for tmp in c_desc:
            #     img = np.array(tmp).reshape((64,64))
            #     display_array(img)
            
            # for displaying the descriptor ranges in the hypothetical year
            plt.axvline(x=start_time_offset, color='r', linestyle='--')
            plt.axvline(x=end_time_offset, color='r', linestyle='--')
            plt.plot(X_plot, dens, 'k-')
            plt.show() 
            
            # print c_desc
            c_descriptors.append(c_desc)
        # print np.array(c_desc).shape
        log('Frames filled from neighbours: ' + str(frames_filled) + '/' + 
            str(frames_total))
        self._descriptors = c_descriptors

    def get_clut_list(self, V):
        clut_list = []
        clut_indices = []
        for nc in range(0, self._n_clusters):
            clut_indices.append(np.where(V == nc)[0])
        # print 'Clustering distirbution'
        # print '-----------------------'
        clut_list.append(clut_indices)
        for pos, c in enumerate(clut_list):
            obv_dev = []
            for nc in range(0, self._n_clusters):
                obv_dev.append((nc, len(c[nc])))
            sort_obd_dev = sorted(obv_dev, key=lambda x: x[1], reverse=True)
            # print sort_obd_dev
        self._clustering_dist = sort_obd_dev
        self._index_list = clut_list

    # def plot_cluster_distirbution(self, outp=None):
    #     lens = []
    #     oc = oct2py.Oct2Py()
    #     for i in self._clustering_dist:
    #         lens.append(i[1])
    #     oc.push('lens', lens)
    #     oc.push('xlens', range(0, self._n_clusters))
    #     if outp is None:
    #         oc.eval('plot(xlens,lens)',plot_width='2048', plot_height='1536')
    #     else:
    #         oc.eval('plot(xlens,lens)',
    #                 plot_dir=outp, plot_name='clustering_frequency', plot_format='jpeg',
    #                 plot_width='2048', plot_height='1536')

    def centroids_distance(self, dataset,features_first=False):
        items = dataset.get_items()
        if features_first:
            items = np.transpose(items)
        dists = [(x,np.linalg.norm(self._centroids[x]-items)) for x in range(0, self._n_clusters)]
        dists = sorted(dists, key=lambda x: x[1], reverse=False)
        return dists

    def desc_date(self,nc_subset):
        desc_date = []
        for pos,i in enumerate(self._descriptors):
            gvalue = nc_subset._dataset.variables[nc_subset._time_name][i[0][0]]
            sim_date = ""
            for gv in gvalue:
                sim_date += gv
            desc_date.append(sim_date)
        self._desc_date = desc_date
        return desc_date

    def CH_evaluation(self):
        return metrics.calinski_harabaz_score(self.get_items(), self._labels)

    def ari(self, labels_true):
        return metrics.adjusted_rand_score(labels_true, self._labels)

    def nmi(self, labels_true):
        return metrics.adjusted_mutual_info_score(labels_true, self._labels)

    def save(self, filename='Clustering_object.zip'):
        utils.save(filename, self)

    def load(self, filename='Clustering_object.zip'):
        self = utils.load(filename)

from Dataset_transformations import Dataset_transformations
from netcdf_subset import netCDF_subset
import numpy as np        
if __name__ == '__main__':
    data_dict = netCDF_subset(
        'test_modified.nc', [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    items = [data_dict.extract_data()]
    items = np.array(items)
    #print items.shape
    ds = Dataset_transformations(items, 1000, items.shape)
    ds.twod_transformation()
    ds.normalize()
    times = data_dict.get_times()
    clust_obj = Clustering(ds, n_clusters=14, n_init=1, features_first=True)
    clust_obj.kmeans()
    # print clust_obj._labels.shape
    clust_obj.create_density_descriptors(12, times)
    # clust_obj.create_descriptors(12)
    print np.array(clust_obj._descriptors).shape
    print clust_obj._descriptors