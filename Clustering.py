import numpy as np
from Dataset import Dataset
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn import metrics
import dataset_utils as utils
import oct2py

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
        self._index_list = None
        self._link = None

    def kmeans(self):
        data = self.get_items()
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
        self.get_clut_list(self._link)
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

    def create_descriptors(self,frames):
        data = self.get_items()
        clut_list = self._index_list[0]
        c_desc = []
        for c in clut_list:
            cluster_data = data[c]
            kj = KMeans(n_clusters=frames, n_init=self._n_init,
                                max_iter=self._max_iter, n_jobs=-1).fit(cluster_data).labels_
            avg = []
            for j in range(0,frames):
                idx = [idx for idx, frame in enumerate(
                    kj) if frame == j]
                avg.append(c[idx])
            c_desc.append(avg)
        self._descriptors = c_desc


    def get_clut_list(self, V):
        clut_list = []
        clut_indices = []
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
        self._clustering_dist = sort_obd_dev
        self._index_list = clut_list

    def plot_cluster_distirbution(self, outp):
        oc = oct2py.Oct2Py()
        for i in self._clustering_dist:
            lens.append(i[1])
        oc.push('lens', lens)
        oc.push('xlens', range(1, self._n_clusters))
        oc.eval('plot(xlens,lens)',
                plot_dir=outp, plot_name='clustering_frequency', plot_format='jpeg',
                plot_width='2048', plot_height='1536')

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
