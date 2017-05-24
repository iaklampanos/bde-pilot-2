from netCDF4 import Dataset
import numpy as np
import os
import datetime

list = sorted(os.listdir('descriptors'))
for l in list:
    no_nc = l.split('.')[0]
    date = datetime.datetime.strptime(no_nc,'%Y-%m-%d_%H:%M:%S')
    dates = []
    dates.append(datetime.datetime.strftime(date,'%Y-%m-%d_%H:%M:%S'))
    for i in xrange(13):
        date = date+datetime.timedelta(hours=6)
        dates.append(datetime.datetime.strftime(date,'%Y-%m-%d_%H:%M:%S'))
    print dates,len(dates)
    dataset = Dataset('./descriptors/'+l,'r')
    s1_d = []
    for d in dates:
        s1 = []
        for j in d:
            s1.append(j)
        s1_d.append(s1)
    # print np.array(s1_d,dtype='|S1')
    dsout = Dataset(l, 'w',format='NETCDF3_CLASSIC')
    for dname, dim in dataset.dimensions.iteritems():
        dsout.createDimension(dname, len(
            dim) if not dim.isunlimited() else None)
    for gattr in dataset.ncattrs():
        gvalue = dataset.getncattr(gattr)
        dsout.setncattr(gattr, gvalue)
    for v_name, varin in dataset.variables.iteritems():
        if v_name == 'Times':
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:]= s1_d[:]
        else:
                outVar = dsout.createVariable(
                        v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[:]
    dsout.close()
    # dataset = Dataset('check.nc','r')
    # #print dataset.variables['Times'][:]
