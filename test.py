import datetime
import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
from netcdf_subset import netCDF_subset,calculate_clut_metrics,calculate_prf,ncar_to_ecmwf_type,find_time_slot
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
import cPickle
from matplotlib.mlab import PCA as mlabPCA
import oct2py

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
    dorg = Dataset('/mnt/disk1/thanasis/data/1986_1987.nc','r')
    #level2 = [500,550,600,650,700,750,775,800,825,850,875,900]
    oc = oct2py.Oct2Py()
    level2 = [700]
    vs2 = ['UU','VV']
    n_sub2 = netCDF_subset(dsin,level2,vs2,'num_metgrid_levels','Times')
    clut_list,Z,c_dist = n_sub2.link_multivar(6,'kmeans')
    max_ret_list = n_sub2.find_continuous_timeslots(clut_list=clut_list)
    n_sub = netCDF_subset(dorg,level2,vs2,'num_metgrid_levels','Times')
    #indices = n_sub.middle_cluster_tofile(outp,max_ret_list)
    """
    h = n_sub2.get_time_diagram(datetime.datetime(1986,1,1,0,0),clut_list)
    for i,hh in enumerate(h):
        oc.push('x',hh)
        a = oc.eval('x*4;')
        oc.push('a',a)
        a = oc.eval('a+1;')
        oc.push('a',a)
        y = oc.zeros(1,2920)
        oc.push('y',y)
        oc.eval('y(a)=1;')
        oc.eval('bar(y)',
                    plot_dir=outp,plot_name='cluster'+str(int(i))+'_timebar',plot_format='jpeg',
                    plot_width='2048',plot_height='1536')
    """
    t1 = find_time_slot(1987,7,10,18,0)
    t2 = find_time_slot(1987,7,13,12,0)
    print n_sub2.euc_dist(t1,t2,clut_list)
