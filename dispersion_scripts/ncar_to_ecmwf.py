from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    dataset = Dataset(inp, 'r')
    dsout = Dataset(outp, 'w',format='NETCDF3_CLASSIC')
    for dname, dim in dataset.dimensions.iteritems():
        dsout.createDimension(dname, len(
            dim) if not dim.isunlimited() else None)
    for v_name, varin in dataset.variables.iteritems():
        if v_name == 'UU' or v_name == 'VV':
            outVar = dsout.createVariable(
                v_name, varin.datatype, ('Time',
                                         'num_metgrid_levels', 'south_north', 'west_east'))
            outVar.setncatts({k: varin.getncattr(k)
                              for k in varin.ncattrs()})
            rang = range(0, 64)
            if v_name == 'UU':
                outVar[:] = varin[:, :, :, rang]
            else:
                outVar[:] = varin[:, :, rang, :]
        elif v_name == 'Times':
            outVar = dsout.createVariable(v_name, 'int32', ('Time',))
            outVar.setncatts({'units': 'hours since 1900-01-01 00:00:00',
                              'long_name': 'time', 'calendar': 'gregorian'})
            nvarin = []
            for var in varin[:]:
                str = ""
                for v in var:
                    str += v
                nvarin.append(str)
            nums = []
            for var in nvarin:
                under_split = var.split('_')
                date_split = under_split[0].split('-')
                time_split = under_split[1].split(':')
                date_object = datetime.datetime(int(date_split[0]), int(date_split[1]), int(
                    date_split[2]), int(time_split[0]), int(time_split[1]))
                d2n = date2num(
                    date_object, 'hours since 1900-01-01 00:00:00', 'gregorian')
                nums.append(int(d2n))
            nums = np.array(nums)
            print nums[:]
            outVar[:] = nums[:]
        else:
            outVar = dsout.createVariable(
                v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k)
                              for k in varin.ncattrs()})
            outVar[:] = varin[:]
    outVar = dsout.createVariable(
        'num_metgrid_levels', 'int32', ('num_metgrid_levels',))
    outVar.setncatts(
        {u'units': u'millibars', u'long_name': u'pressure_level'})
    outVar[:] = np.array([0, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,
                          175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
                          650, 700, 750, 775, 800, 825, 850, 875, 900, 925,
                          950, 975, 1000])[:]
    dsout.close()
