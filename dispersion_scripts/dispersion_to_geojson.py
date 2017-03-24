from netCDF4 import Dataset
import numpy as np
import os
import subprocess

p = ''
pa = []
for ppa in pa:
    path = p+ppa+'/'
    for file in os.listdir(path):
        if file.endswith(".nc"):
            print file
            dataset = Dataset(path+file,'r')
            dsout = Dataset(path+'TEST.nc','w',format='NETCDF3_CLASSIC')
            c137 = dataset.variables['C137'][:]
            i131 = dataset.variables['I131'][:]
            c137 = np.sum(c137,axis=0).reshape(501,501)
            i131 = np.sum(i131,axis=0).reshape(501,501)
            for gattr in dataset.ncattrs():
                gvalue = dataset.getncattr(gattr)
                dsout.setncattr(gattr, gvalue)
            for dname, dim in dataset.dimensions.iteritems():
                if dname == 'time':
                    dsout.createDimension(dname,1 if not dim.isunlimited() else None)
                else:
                    dsout.createDimension(dname, len(
                        dim) if not dim.isunlimited() else None)
            print dsout.dimensions
            for v_name, varin in dataset.variables.iteritems():
                if v_name == 'C137':
                    outVar = dsout.createVariable(
                                v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k)
                                      for k in varin.ncattrs()})
                    outVar[:] = c137[:]
                elif v_name == 'I131':
                    outVar = dsout.createVariable(
                                v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k)
                                      for k in varin.ncattrs()})
                    outVar[:] = i131[:]
                else:
                    try:
                        outVar = dsout.createVariable(
                                    v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k)
                                          for k in varin.ncattrs()})
                        outVar[:] = varin[:]
                    except:
                        outVar[:] = varin[0]
            dsout.close()
            os.chdir(path)
            os.system('gdal_translate NETCDF:\\"TEST.nc\\":C137 '+file.split('.')[0]+'_c137.tiff')
            os.system('gdal_translate NETCDF:\\"TEST.nc\\":I131 '+file.split('.')[0]+'_i131.tiff')
            os.system('rm TEST.nc')
