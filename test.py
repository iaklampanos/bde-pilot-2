import numpy as np
from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
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
    level = [500,1000]
    vs = ['u','v']
    n_sub = netCDF_subset(dsin,level,vs)
    print n_sub.lvl_pos()
    n_sub.write_tofile(outp)
