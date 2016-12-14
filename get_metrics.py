from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from Kclustering import Kclustering
from Hclustering import Hclustering
from Meta_clustering import Meta_clustering
import datetime
import numpy as np
import oct2py
import os

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='output path')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    seasons = ['winter', 'spring', 'summer', 'autumn', 'cold', 'hot']
    algorithm = 'kmeans'
    season = 'cold'
    rang = range(4,21)
    oc = oct2py.Oct2Py()
    oc.push('x', rang)
    dsin = Dataset(inp, "r")
    lvl_v = [[300], [500], [700], [900]]
    for l in lvl_v:
        plevels = l
        #plevels = [500]
        sub_var = ['UU','VV']
        data_dict = {'dataset': dsin, 'levels': plevels,
                     'sub_vars': sub_var, 'lvlname': 'num_metgrid_levels',
                     'timename': 'Times', 'time_unit': 'hours since 1900-01-01 00:00:0.0',
                     'time_cal': 'gregorian', 'ncar_lvls': None}
        dir_name = ""
        for pos, v_name in enumerate(sub_var):
            if pos != 0:
                dir_name += '_' + str(v_name)
            else:
                dir_name += str(v_name)
        if season in seasons:
            s_path = outp + '/' + season
            if not(os.path.isdir(s_path)):
                os.mkdir(s_path)
            met_path = s_path + '/' + dir_name
        else:
            met_path = outp + '/' + dir_name
        for pos, lvl_name in enumerate(plevels):
            if pos != 0:
                dir_name += '_' + str(lvl_name)
            else:
                dir_name += str(lvl_name)
        lvl_path = met_path + '/' + dir_name
        link_path = lvl_path + '/saved_linkages'
        dist_path = lvl_path + '/cluster_distirbution'
        bar_path = lvl_path + '/time_bars'
        if not(os.path.isdir(met_path)):
            os.mkdir(met_path)
        if not(os.path.isdir(lvl_path)):
            os.mkdir(lvl_path)
        if not(os.path.isdir(link_path)):
            os.mkdir(link_path)
        if not (os.path.isdir(bar_path)):
            os.mkdir(bar_path)
        if not(os.path.isdir(dist_path)):
            os.mkdir(dist_path)
        WSS = []
        BSS = []
        for i in rang:
            print i
            if algorithm == 'kmeans':
                c_dict = {'n_clusters': i, 'normalize': True, 'season': None,
                          'therm_season': 'cold', 'multilevel': False, 'size_desc': 24, 'size_div': 3}
                kc = Kclustering(c_dict,data_dict)
                clut_list, linkage, c_dist = kc.link_multi2svar()
                mc_dict = {'normalize'}
                mc = Meta_clustering(data_dict,c_dict)
                wss,bss,total = mc.calculate_clut_metrics(mc.prepare_c_list_for_metrics(clut_list))
                tdia = mc._netcdf_subset.get_time_diagram(datetime.datetime(1986,1,1,0,0),clut_list)
                c_bar_path = bar_path + '/c'+str(i)
                if not(os.path.isdir(c_bar_path)):
                   os.mkdir(c_bar_path)
                for ti,td in enumerate(tdia):
                    oc.push('td',td)
                    a = oc.eval('td*4;')
                    oc.push('a',a)
                    a = oc.eval('a+1;')
                    oc.push('a',a)
                    y = oc.zeros(1,len(mc._netcdf_subset.get_times()))
                    oc.push('y',y)
                    oc.eval('y(a)=1;')
                    oc.eval('bar(y)',plot_dir=c_bar_path,plot_name='cluster'+str(int(ti))+'_timebar',plot_format='jpeg',
                            plot_width='2048',plot_height='1536')
            else:
                c_dict = {'n_clusters': i, 'normalize': True, 'affinity': 'cosine', 'method': 'average',
                          'season': None, 'therm_season': 'cold', 'multilevel': False, 'size_desc': 24, 'size_div': 3}
                hc = Hclustering(c_dict,data_dict)
                clut_list, linkage, c_dist = hc.multi2svar()
                mc = Meta_clustering(data_dict,c_dict)
                wss,bss,total = mc.calculate_clut_metrics(mc.prepare_c_list_for_metrics(clut_list))
                tdia = mc._netcdf_subset.get_time_diagram(datetime.datetime(1986,1,1,0,0),clut_list)
                c_bar_path = bar_path + '/c'+str(i)
                if not(os.path.isdir(c_bar_path)):
                   os.mkdir(c_bar_path)
                for ti,td in enumerate(tdia):
                    oc.push('td',td)
                    a = oc.eval('td*4;')
                    oc.push('a',a)
                    a = oc.eval('a+1;')
                    oc.push('a',a)
                    y = oc.zeros(1,len(linkage))
                    oc.push('y',y)
                    oc.eval('y(a)=1;')
                    oc.eval('bar(y)',plot_dir=c_bar_path,plot_name='cluster'+str(int(i))+'_timebar',plot_format='jpeg',
                            plot_width='2048',plot_height='1536')
            oc.push('x_dist', range(0, i))
            oc.push('y_dist', [int(j[1]) for j in c_dist])
            oc.eval("plot(x_dist,y_dist),title(\'Clustering distirbution\')",
                    plot_dir=dist_path, plot_name='cluster' + str(int(i)) + '_distirbution', plot_format='jpeg',
                    plot_width='2048', plot_height='1536')
            np.save(link_path + '/cluster' +
                    str(int(i)) + "_linkage.npy", linkage)
            WSS.append(wss)
            BSS.append(bss)
        oc.push('wss', WSS)
        oc.push('bss', BSS)
        oc.eval("subplot(2,1,1),plot(x,wss),title(\'WSS\'),subplot(2,1,2),plot(x,bss),title(\'BSS\')",
                plot_dir=lvl_path, plot_name=dir_name + '_wss_bss_metrics', plot_format='jpeg',
                plot_width='2048', plot_height='1536')
