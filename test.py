import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
from netcdf_subset import netCDF_subset,calculate_clut_metrics,calculate_prf
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as sklearnPCA
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
    #level = [500]
    #vs = ['u','v']
    #n_sub = netCDF_subset(dsin,level,vs,'level','time')
    #clut_list = n_sub.link_var('average','cosine',10)
    #n_sub.link_var('average','cosine',10)
    #print n_sub.lvl_pos()
    #for i in clut_list:
    #    avg_list = []
    #    for pos,c in enumerate(i):
    #        avg_list.append(n_sub.calculate_clut_avg_cent(pos,clut_list))
    #    print avg_list
    #n_sub.link_multivar('average','cosine',10)
    #n_sub.cluster_tofile(outp,2,clut_list)
    #clut_list,UV = n_sub.link_multivar('average','cosine',10)
    #n_sub.multi_cluster_tofile(outp,3,clust_list)
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
    print vv.shape
    del(vals)
    del(vals2)
    #mlab_pca = mlabPCA(vv.T)

    sklean_pca = sklearnPCA(n_components=5)
    sklearn_transf = sklean_pca.fit_transform(vv.T)
    print len(sklearn_transf)
    print sklearn_transfs
    #clut_list2,Z = n_sub2.link_var('average','cosine',5)
    #n_sub2.find_continuous_timeslots(clut_list2)
    #wss,bss,total = calculate_clut_metrics(n_sub2.prepare_c_list_for_metrics(clut_list2))
    #for i in range(0,5):
    #    n_sub2.single_cluster_tofile(outp,i,clut_list2)
    #plt.plot(range(0,Z.shape[0]),Z[:,2])
    #plt.show()
    #WSS =[]
    #BSS=[]
    #TOTAL = []
    #for i in range(2,21):
    #    clut_list2,Z = n_sub2.link_var('average','cosine',i)
    #    print calculate_prf(clut_list,clut_list2)
    #    clusters = []
    #    for c in clut_list2[0]:
    #        temp_arr = np.array(n_sub2.extract_timedata(c.tolist(),n_sub2.lvl_pos()))
    #        temp_arr = temp_arr.reshape(len(c),5335)
    #        clusters.append(temp_arr)
        #print clusters[0].shape
        #np.savetxt('qq.txt',clusters[0])
        #print clusters[0]
        #print 'Cluster number...... ',i
        #print 'WSS BSS TOTAL'
    #    wss,bss,total = calculate_clut_metrics(clusters)
    #    WSS.append(wss)
    #    BSS.append(bss)
    #    TOTAL.append(total)
    #print 'WSS'
    #print WSS
    #print 'BSS'
    #print BSS
    #print 'TOTAL'
    #print total
    #plt.plot(range(0,UV.shape[0]),UV[:,2],'r--')
    #plt.show()
    #n_sub.write_tofile(outp)
    #print n_sub.get_time()
