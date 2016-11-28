from netcdf_subset import netCDF_subset,calculate_clut_metrics,calculate_prf,ncar_to_ecmwf_type
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
import numpy as np
import oct2py
import os

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input',required=True,type=str,
                        help='input file')
    parser.add_argument('-r1','--range1',required=True,type=int,
                        help='')
    parser.add_argument('-r2','--range2',required=True,type=int,
                        help='')
    parser.add_argument('-a','--algorithm',required=True,type=str)
    parser.add_argument('-o', '--output',type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input','range1','range2','algorithm','output')
    inp,rang1,rang2,alg,outp = getter(opts)
    rang = range(rang1,rang2)
    oc = oct2py.Oct2Py()
    oc.push('x',rang)
    dsin = Dataset(inp,"r")
    level2 = [300]
    vs2 = ['UU','VV','GHT']
    n_sub2 = netCDF_subset(dsin,level2,vs2,'num_metgrid_levels','Times')
    dir_name = ""
    for pos,v_name in enumerate(vs2):
        if pos!=0:
           dir_name += '_'+str(v_name)
        else:
           dir_name += str(v_name)
    met_path = outp+'/'+dir_name
    for pos,lvl_name in enumerate(level2):
        if pos!=0:
           dir_name += '_'+str(lvl_name)
        else:
           dir_name += str(lvl_name)
    lvl_path = met_path+'/'+dir_name
    link_path = lvl_path+'/saved_linkages'
    dist_path = lvl_path+'/cluster_distirbution'
    if not(os.path.isdir(met_path)):
       os.mkdir(met_path)
    if not(os.path.isdir(lvl_path)):
       os.mkdir(lvl_path)
    if not(os.path.isdir(link_path)):
       os.mkdir(link_path)
    if not(os.path.isdir(dist_path)):
        os.mkdir(dist_path)
    WSS = []
    BSS = []
    for i in rang:
        print i
        if len(vs2)==1:
            if len(level2)==1:
               clut_list,linkage,c_dist = n_sub2.link_var(n_clusters=i,algorithm=alg)
            else:
               clut_list,linkage,c_dist = n_sub2.link_var(n_clusters=i,algorithm=alg,multilevel=True)
        else:
           if len(level2)==1:
              clut_list,linkage,c_dist = n_sub2.link_multivar(n_clusters=i,algorithm=alg)
           else:
              clut_list,linkage,c_dist = n_sub2.link_multivar(n_clusters=i,algorithm=alg,multilevel=True)
        oc.push('x_dist',range(0,i))
        oc.push('y_dist',[int(j[1]) for j in c_dist])
        oc.eval("plot(x_dist,y_dist),title(\'Clustering distirbution\')",
                    plot_dir=dist_path,plot_name='cluster'+str(int(i))+'_distirbution',plot_format='jpeg',
                    plot_width='2048',plot_height='1536')
        np.save(link_path+'/cluster'+str(int(i))+"_linkage.npy",linkage)
        wss,bss,total = calculate_clut_metrics(n_sub2.prepare_c_list_for_metrics(clut_list))
        WSS.append(wss)
        BSS.append(bss)
    oc.push('wss',WSS)
    oc.push('bss',BSS)
    oc.eval("subplot(2,1,1),plot(x,wss),title(\'WSS\'),subplot(2,1,2),plot(x,bss),title(\'BSS\')",
            plot_dir=lvl_path,plot_name=dir_name+'_wss_bss_metrics',plot_format='jpeg',
            plot_width='2048',plot_height='1536')
