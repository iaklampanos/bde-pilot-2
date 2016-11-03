import numpy as np
from netCDF4 import Dataset


class netCDF_subset(object):
      dataset = None #initial netcdf dataset path
      pressure_levels = None #pressure level of interest
      subset_variables = None #variables of interest
      
      #Constructor
      def __init__(self,dataset,levels,sub_vars):
          #Init original dataset
          self.dataset = dataset
          #Multiple levels
          self.pressure_levels = levels
          #Multiple vars
          self.subset_variables = sub_vars
      
      #Find pressure level position in dataset
      def lvl_pos(self):
          idx_list = []
          arr = np.array(self.dataset.variables['level']).tolist()
          for lvl in self.pressure_levels:
              idx_list.append(arr.index(lvl))
          return idx_list
      
      #Retrieve variables for a specific level (defined in Class attributes)
      def extract_data(self):
          sub_pos = self.lvl_pos()
          var_list = []
          for v in self.subset_variables:
                var_list.append(self.dataset.variables[v][:,sub_pos,:,:])
          return var_list
      
      #Export results to file
      def write_tofile(self,out_path):
          dsout = Dataset(out_path,'w')
          dim_vars = []
          var_list = self.extract_data()
          for dname, dim in self.dataset.dimensions.iteritems():
              dim_vars.append(dname)
              if dname != 'level':
                 dsout.createDimension(dname, len(dim) if not dim.isunlimited() else None)
              else:
                 dsout.createDimension(dname, len(self.pressure_levels) if not dim.isunlimited() else None) 
          for v_name, varin in self.dataset.variables.iteritems():
              if v_name in self.subset_variables:
                  for pos_v,v in enumerate(self.subset_variables):
                      if v_name == v:
                          outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
                          outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                          outVar[:] = var_list[pos_v]
              elif v_name in dim_vars:
                  outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
                  outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                  if v_name == 'level':
                     outVar[:] = self.pressure_levels
                  else:
                     outVar[:] = varin[:]
          dsout.close()
                 
