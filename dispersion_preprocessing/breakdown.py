from netcdf_subset import netCDF_subset
from netCDF4 import Dataset, num2date, date2num
from operator import attrgetter
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-im', '--input_mod', required=True, type=str,
                        help='modified file')
    parser.add_argument('-ir', '--input_real', required=True, type=str,
                        help='real file')
    opts = parser.parse_args()
    getter = attrgetter('input_mod','input_real')
    inp,inpr = getter(opts)
    dsin = Dataset(inp,'r')
    dsinr = Dataset(inpr,'r')
    nc_mod = netCDF_subset(dsin, levels=[500,700,900],sub_vars=['UU','VV','TT'],
                 lvlname='num_metgrid_levels', timename='Times',
                 time_unit='hours since 1900-01-01 00:00:00',
                 time_cal='gregorian', ncar_lvls=None)
    nc = netCDF_subset(dsinr, levels=[500,700,900],sub_vars=['UU','VV','TT'],
                 lvlname='num_metgrid_levels', timename='Times',
                 time_unit='hours since 1900-01-01 00:00:00',
                 time_cal='gregorian', ncar_lvls=None)
    dates = nc_mod.get_times()
    ltimes = len(nc_mod._dataset[nc_mod._time_name][:])
    times = range(ltimes)
    tidx = times[0::13]
    for pos,i in enumerate(tidx):
        try:
            nc.exact_copy_file('./nc/'+dates[i].strftime("%Y-%m-%d_%H:%M:%S")+'.nc'
                               ,range(tidx[pos],tidx[pos+1]))
        except IndexError:
            continue
