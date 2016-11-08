import numpy as np
from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from netCDF4 import Dataset

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
    level = [500]
    vs = ['u','v']
    n_sub = netCDF_subset(dsin,level,vs,'level','time')
    #clut_list = n_sub.link_var('average','cosine',10)
    #n_sub.link_var('average','cosine',10)
    #print n_sub.lvl_pos()
    #for i in clut_list:
    #    avg_list = []
    #    for pos,c in enumerate(i):
    #        avg_list.append(n_sub.calculate_clut_avg_cent(pos,clut_list))
    #    print avg_list
    n_sub.link_multivar('average','cosine',10)
   # n_sub.cluster_tofile(outp,2,clut_list)
    #n_sub.write_tofile(outp)
    #print n_sub.get_time()
