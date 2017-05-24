from netCDF4 import Dataset,date2num
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import json
from operator import attrgetter
from argparse import ArgumentParser


parser = ArgumentParser(description='Extract variables from netcdf file')
parser.add_argument('-i', '--input', required=True, type=str,
                help='input file')
opts = parser.parse_args()
getter = attrgetter('input')
file = getter(opts)
path = ''
out = ''
print file
dataset = Dataset(path+file, 'r')
dsout = Dataset(path+'test2.nc', 'w',format='NETCDF3_CLASSIC')
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

dataset = Dataset(path+'test2.nc', 'r')
dsout = Dataset(path+'test3.nc','w')
for dname, dim in dataset.dimensions.iteritems():
    dsout.createDimension(dname, len(
        dim) if not dim.isunlimited() else None)
dsout.createDimension('latitude',4096)
dsout.createDimension('longitude',4096)
for v_name, varin in dataset.variables.iteritems():
    if v_name == 'UU' or v_name == 'VV':
        outVar = dsout.createVariable(
            v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k)
                          for k in varin.ncattrs()})
        outVar[:] = varin[:]
    elif v_name == 'XLAT_M':
        outVar = dsout.createVariable(
            'latitude', 'float32', ('latitude',))
        outVar.setncatts({k: varin.getncattr(k)
                          for k in varin.ncattrs()})
        outVar[:] = varin[0,:,:].flatten()
    elif v_name == 'XLONG_M':
        outVar = dsout.createVariable(
            'longitude', 'float32', ('longitude',))
        outVar.setncatts({k: varin.getncattr(k)
                          for k in varin.ncattrs()})
        outVar[:] = varin[0,:,:].flatten()
dsout.close()

import matplotlib.pyplot as plt
import json
from netCDF4 import Dataset,date2num
import numpy as np
import datetime
dataset = Dataset(path+'test3.nc', 'r')
u = dataset.variables['UU'][:].reshape(494,4096)
v = dataset.variables['VV'][:].reshape(494,4096)
lat = dataset.variables['latitude'][:]
lon = dataset.variables['longitude'][:]
u = np.sum(u,axis=0)
v = np.sum(v,axis=0)
uv = np.vstack((u,v))
uv = np.divide(uv,np.max(uv))
x1 = lon
y1 = lat
x1 = [float(i) for i in lon]
y1 = [float(i) for i in lat]
x2 = []
y2 = []
arr = []
import math
from geojson import Feature,Point,MultiPoint,MultiLineString
for i in range(0,uv.shape[1]):
    x2.append(float(x1[i]+uv[0][i]))
    y2.append(float(y1[i]+uv[1][i]))
arr = []
for i in range(0,uv.shape[1]):
    L1 = math.sqrt((x1[i]-x2[i])*(x1[i]-x2[i])+(y2[i]-y1[i])*(y2[i]-y1[i]))
    L2 = float(L1 / 3.5)
    x3 = x2[i] + (L2/L1)*((x1[i]-x2[i])*math.cos((math.pi/6))+(y1[i]-y2[i])*math.sin((math.pi)/6))
    y3 = y2[i] + (L2/L1)*((y1[i]-y2[i])*math.cos((math.pi/6))-(x1[i]-x2[i])*math.sin((math.pi)/6))
    x4 = x2[i] + (L2/L1)*((x1[i]-x2[i])*math.cos((math.pi/6))-(y1[i]-y2[i])*math.sin((math.pi)/6))
    y4 = y2[i] + (L2/L1)*((y1[i]-y2[i])*math.cos((math.pi/6))+(x1[i]-x2[i])*math.sin((math.pi)/6))
    a = (x1[i],y1[i])
    b = (x2[i],y2[i])
    c = (x3,y3)
    d = (x4,y4)
    temp = []
    temp.append(a)
    temp.append(b)
    temp2 = []
    temp2.append(b)
    temp2.append(c)
    temp3 = []
    temp3.append(b)
    temp3.append(d)
    arr.append(temp)
    arr.append(temp2)
    arr.append(temp3)
feature = Feature(geometry=MultiLineString(arr))
# print feature
# import json
with open(out+file+'.json', 'w') as outfile:
    json.dump(feature, outfile)
dataset.close()
os.system('rm /mnt/disk1/thanasis/data/wrf/nc/test*')
