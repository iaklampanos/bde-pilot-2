import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
from netcdf_subset import netCDF_subset,calculate_clut_metrics,calculate_prf
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
import cPickle
from matplotlib.mlab import PCA as mlabPCA

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input',required=True,type=str,
                        help='input file')
    parser.add_argument('-o', '--output',type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input','output')
    inp,outp = getter(opts)
    dsin = Dataset(inp,"r")
    level2 = [500]
    vs2 = ['z']
    n_sub2 = netCDF_subset(dsin,level2,vs2,'level','time')
    vals = np.load('../c_6_z_values.npy')
    vals2 = []
    for v in vals:
        print v.shape
        q = v.reshape(len(v),v.shape[1]*v.shape[2])
        vals2.append(q)
    for v in vals2:
        print v.shape
    vals2 = np.array(vals2)
    vv = np.array([0])
    for v in vals2:
        vv = np.append(v,vv)
    vv = np.delete(vv,0)
    vv = vv.reshape(2920,47107)
    del(vals)
    del(vals2)
    #kmeans = KMeans(n_clusters=6,random_state=0).fit(vv)
    with open('kmeans_6_clusters_var_z.pkl', 'rb') as fid:
         kmeans=cPickle.load(fid)
    print len(kmeans.labels_)
    print kmeans.labels_
    print kmeans.cluster_centers_
    print len(kmeans.cluster_centers_)
    for c in kmeans.cluster_centers_:
        print len(c)
    clut_list = n_sub2.get_clusters_saved(kmeans.labels_,6,'kmeans')
    print len(clut_list)
    print len(clut_list[0])
    print clut_list[0][0]
    WSS,BSS,TOTAL = calculate_clut_metrics(n_sub2.prepare_c_list_for_metrics(clut_list))
    print WSS
    print BSS
    #mlab_pca = mlabPCA(vv.T)
    #pca = PCA(n_components=500, svd_solver='full')
    #pca.fit(vv)
    #print len(pca.components_)
    #print type(sklearn_transf)
    #print len(sklearn_transf)
    #print sklearn_transf
