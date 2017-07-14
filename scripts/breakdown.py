import sys

BDE_PILOT_SRC = '..'

sys.path.append(BDE_PILOT_SRC)

from netcdf_subset import netCDF_subset
from netCDF4 import Dataset, num2date, date2num
from operator import attrgetter
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='netcdf file')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='netcdf file')
    opts = parser.parse_args()
    getter = attrgetter('input','output')
    inp,outp = getter(opts)
    if '/' not in outp:
        outp = outp + '/'
    nc_mod = netCDF_subset(inp, levels=[500,700,900],sub_vars=['UU','VV','TT'],
                 lvlname='num_metgrid_levels', timename='Times',
                 time_unit='hours since 1900-01-01 00:00:00',
                 time_cal='gregorian', ncar_lvls=None)
    dates = nc_mod.get_times()
    ltimes = len(nc_mod._dataset[nc_mod._time_name][:])
    times = range(ltimes)
    tidx = times[0::12]
    for pos,i in enumerate(tidx):
        try:
            nc_mod.exact_copy_file(outp+str(pos)+'.nc',range(tidx[pos],tidx[pos+1]))
        except IndexError:
            continue
