import os
from string import Template
from operator import attrgetter
from argparse import ArgumentParser
import datetime

class MyTemplate(Template):
   delimiter = '!@#'
   idpattern = '[a-z0-9]*'

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    opts = parser.parse_args()
    getter = attrgetter('input')
    inp = getter(opts)
    filelist = sorted(os.listdir(inp))
    #lfiles = range(len(filelist))
    #fidx = lfiles[0::13]
    for pos,f in enumerate(filelist):
        parts = f.split('_')
        print parts
        date = parts[0]
        hours = parts[1]
        new_hour = parts[1].split('.')[0]
        h = hours.split(':')[0]
        nwps = open('template.sh', 'r')
        nwps_new = open('ncsplitcluster.sh', 'w')
        template = MyTemplate(nwps.read())
        di = {'date':date,'h':h,'file':inp+'/'+f}
        nwps_new.write(template.substitute(di))
        nwps_new.close()
        nwps.close()
        os.system('chmod +x ncsplitcluster.sh')
        os.system('./ncsplitcluster.sh')
        date = parts[0]+'_'+new_hour
        nwrf = open('tempnamelist.input','r')
        nwrf_new = open('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/namelist.input','w')
        template = Template(nwrf.read())
        dst = datetime.datetime.strptime(date,'%Y-%m-%d_%H:%M:%S')
        edst = dst+datetime.timedelta(days=3)
        dif = edst-dst
        di = {'run_hours':dif.days*24+dif.seconds//3600,'start_year':int(dst.year),'start_month':int(dst.month),'start_day':int(dst.day),'start_hour':int(dst.hour),'end_year':int(edst.year),'end_month':int(edst.month),'end_day':int(edst.day),'end_hour':int(edst.hour)}
        nwrf_new.write(template.substitute(di))
        nwrf_new.close()
        nwrf.close()
        os.chdir('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/')
        os.system('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/./real.exe')
        os.environ['OMP_NUM_THREADS'] = '4'
        os.system('echo $OMP_NUM_THREADS')
        os.system('mpirun -np 4 /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/./wrf.exe >> /dev/null')
        os.system('rm /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/met_em*')
        #os.system('rm /home/wrf/data/nc/*')
