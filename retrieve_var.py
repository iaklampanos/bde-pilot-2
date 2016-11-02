import numpy as np
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input',
                        help='input file')
    parser.add_argument('-l', '--level',required=True,
                        help='pressure level')
    parser.add_argument('-o', '--output',
                        help='output file')
    opts = parser.parse_args()

    getter = attrgetter('level')
    level = getter(opts)
    level = int(level)
    dsin = Dataset('data.nc')
    
    
    arr = np.array(dsin.variables['level'])
    print arr
    checklevel = False
    pos = 0
    for c,i in enumerate(arr):
        if i == level:
            checklevel = True
            pos = c
    u = dsin.variables['u'][:]
    lvl = dsin.variables['level'][:]
    #u = dsin.variables['u'][:,21:22,:,:]
    lui = np.argmin( np.abs( u - lvl[pos] ) )
    li = np.argmin( np.abs( u - level[pos+1] ) )
    usub = dsin.variables['u'][:,lui:li,:,:]
    dsout = Dataset("crop.nc", "w")
    #Copy dimensions
    for dname, the_dim in dsin.dimensions.iteritems():
        #print dname, len(the_dim)
        if dname != 'level':
           dsout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
        else:
           dsout.createDimension(dname, 1 if not the_dim.isunlimited() else None)
    for dname, the_dim in dsout.dimensions.iteritems():
        print dname, len(the_dim)
    var_std = ['time','level','latitude','longitude']
    # Copy variables
    for v_name, varin in dsin.variables.iteritems():
        if v_name == 'u':
            outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
        # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            #outVar[:] = usub
            outVar[:] = u
        elif v_name in var_std:
            print v_name,varin.datatype,varin.dimensions
            outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
            
            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            if v_name == 'level':
                outVar[:] = varin[pos:pos+1]
            else:
                outVar[:] = varin[:]
    # close the output file
    dsout.close()
