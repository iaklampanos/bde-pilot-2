import numpy as np
from netCDF4 import Dataset,num2date
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree


class netCDF_subset(object):
      dataset = None #initial netcdf dataset path
      level_name = None
      time_name = None
      pressure_levels = None #pressure level of interest
      subset_variables = None #variables of interest
      
      #Constructor
      def __init__(self,dataset,levels,sub_vars,lvlname,timename):
          #Init original dataset
          self.dataset = dataset
          #Multiple levels
          self.pressure_levels = levels
          #Multiple vars
          self.subset_variables = sub_vars
          self.level_name = lvlname
          self.time_name = timename
      
      #Find pressure level position in dataset
      def lvl_pos(self):
          idx_list = []
          arr = np.array(self.dataset.variables[self.level_name]).tolist()
          for lvl in self.pressure_levels:
              idx_list.append(arr.index(lvl))
          return idx_list
      
      #Retrieve variables for a specific level (defined in Class attributes)
      def extract_data(self,sub_pos):
          var_list = []
          for v in self.subset_variables:
                var_list.append(self.dataset.variables[v][:,sub_pos,:,:])
          return var_list
      
      #Retrieve variables for a specific level and time (used in clusters to file)
      def extract_timedata(self,time_pos,sub_pos):
          var_list = []
          for v in self.subset_variables:
                var_list.append(self.dataset.variables[v][time_pos,sub_pos,:,:])
          return var_list

      def calculate_clut_avg_cent(self,cluster_label,clut_list):
          for pos,c in enumerate(clut_list):
              print 'Calculating average for Variable ',self.subset_variables[pos]
              print 'Cluster label is ',cluster_label
              var_cluster_state = self.extract_timedata(c[cluster_label],self.lvl_pos())
              return np.average(var_cluster_state)
      
      #Perform clustering and retrieve dataset clustered in n_clusters (for multiple variables)
      def link_multivar(self,method,metrics,n_clusters):
          var_list = self.extract_data(self.lvl_pos())
          clut_list = []
          temp_v_list = []
          for pos,v in enumerate(var_list):
              temp_v_list.append(np.ndarray(shape=(v.shape[0],v[0][:].flatten().shape[0])))
              for i in range(0,v.shape[0]):
                  temp_v_list[pos][i] = v[i][:].flatten()
          gather_data = np.concatenate(temp_v_list)
          uv = np.ndarray(shape=(gather_data.shape[0]/len(var_list),gather_data.shape[1]*2))
          for pos,idx in enumerate(uv):
              uv[pos] = np.append(gather_data[pos],gather_data[pos+uv.shape[0]])
          print uv.shape
          del gather_data
          print uv[1]
          print len(uv[0])
          UV = linkage(uv,method,metrics)
          cutree = np.array(cut_tree(UV,n_clusters=n_clusters).flatten())
          clut_indices = []
          for nc in range(0,n_clusters):
              clut_indices.append(np.where(cutree == nc)[0])
          clut_list.append(clut_indices)
          print 'Cluster distirbution'
          print '---------------------'
          for pos,c in enumerate(clut_list):
              obv_dev = []
              for nc in range(0,n_clusters):
                  obv_dev.append((nc,len(c[nc])))
              print sorted(obv_dev,key=lambda x:x[1],reverse=True)
          return clut_list
                   
      #Perform clustering and retrieve dataset clustered in n_clusters (every var individually)
      def link_var(self,method,metrics,n_clusters):
          var_list = self.extract_data(self.lvl_pos())
          clut_list = []
          for v in var_list:
              var_data = np.ndarray(shape=(v.shape[0],v[0][:].flatten().shape[0]))
              for i in range(0,v.shape[0]):
                  var_data[i] = v[i][:].flatten()
              print var_data.shape
              V = linkage(var_data,method,metrics)
              cutree = np.array(cut_tree(V, n_clusters=n_clusters).flatten())
              clut_indices = []
              for nc in range(0,n_clusters):
                  clut_indices.append(np.where(cutree == nc)[0])
              clut_list.append(clut_indices)
              print 'Cluster distirbution'
              print '---------------------'
              for pos,c in enumerate(clut_list):
                  print 'Variable ',self.subset_variables[pos]
                  obv_dev = []
                  for nc in range(0,n_clusters):
                      obv_dev.append((nc,len(c[nc])))
                  print sorted(obv_dev,key=lambda x:x[1],reverse=True)
                  #for nc in range(0,n_clusters):
                  #    print 'Cluster ',nc
                  #    print '--------------------------'
                  #    unit = self.dataset.variables['time'].units
                  #    cal = self.dataset.variables['time'].calendar
                  #    times = self.dataset.variables['time'][c[nc]]
                  #    print num2date(times,unit,cal)                      
          return clut_list
      

          
      #Write a single cluster to a file
      def cluster_tofile(self,out_path,cluster_label,clut_list):
          for pos,c in enumerate(clut_list):
              print 'Creating file for Variable ',self.subset_variables[pos]
              print 'Cluster label is ',cluster_label
              self.write_timetofile(out_path+'/var_'+self.subset_variables[pos]+'_cluster'+str(cluster_label)+'.nc',self.lvl_pos(),c[cluster_label])
                  
            
      #Export results to file from attibute dataset
      def write_tofile(self,out_path):
          dsout = Dataset(out_path,'w')
          dim_vars = []
          var_list = self.extract_data(self.lvl_pos())
          for dname, dim in self.dataset.dimensions.iteritems():
              dim_vars.append(dname)
              if dname != self.level_name:
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
                  if v_name == self.level_name:
                     outVar[:] = self.pressure_levels
                  else:
                     outVar[:] = varin[:]
          dsout.close()
      
      #Export variables for specific lvl and time period
      def write_timetofile(self,out_path,lvl_pos,time_pos):
          dsout = Dataset(out_path,'w')
          dim_vars = []
          var_list = self.extract_timedata(time_pos,lvl_pos)
          for dname, dim in self.dataset.dimensions.iteritems():
              dim_vars.append(dname)
              if dname == self.level_name:
                 dsout.createDimension(dname, len(self.pressure_levels) if not dim.isunlimited() else None) 
              elif dname == self.time_name:
                 dsout.createDimension(dname, len(time_pos) if not dim.isunlimited() else None)
              else:
                 dsout.createDimension(dname, len(dim) if not dim.isunlimited() else None)
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
                  if v_name == self.level_name:
                     outVar[:] = self.pressure_levels
                  elif v_name == self.time_name:
                     outVar[:] = self.dataset.variables[self.time_name][time_pos]
                  else:
                     outVar[:] = varin[:]
          dsout.close()
                 
           
