from netCDF4 import Dataset
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import Basemap

inp = '11_train.nc'
nc = Dataset(inp,'r')
lats = nc_fid.variables['XLAT_M']
lons = nc_fid.variables['XLONG_M']

time = nc_fid.variables['time']
data_dict = netCDF_subset(inp, [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
items = [data_dict.extract_piece(range(20,25),range(0,64),range(0,64))]

m = Basemap(width=5000000,height=3500000,
 resolution='l',projection='stere', lat_0 = 60, lon_0 = 70, lat_ts = 40)
m.drawcoastlines()
m.drawcountries()
lons, lats = np.meshgrid(lons, lats)
x, y = m(lons, lats)

clevs = np.arange(400.,604.,4.)
cs = m.contour(x, y, items[0] * .1, clevs, linewidths=1.5, colors = 'k')
plt.clabel(cs, inline=1, fontsize=15, color='k', fmt='%.0f')

pcl = m.pcolor(x,y,np.squeeze(hgt[0]*.1))
cbar = m.colorbar(pcl, location='bottom', pad="10%")
cbar.set_label("hPa")

plt.title('500 hPa Geopotential Height')
plt.show()
