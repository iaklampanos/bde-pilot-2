from string import Template
import os
import datetime

if __name__ == '__main__':
    onlyfiles = [f for f in os.listdir('/home/wrf/data/grib/') if os.path.isfile(os.path.join('/home/wrf/data/grib/', f))]
    onlyfiles = sorted(onlyfiles)
    std = onlyfiles[0]
    std_year = std[0:4]
    std_month = std[4:6]
    std_day = std[6:8]
    std_hour = std[8:12]
    ed = onlyfiles[len(onlyfiles)-1]
    ed_year = ed[0:4]
    ed_month = ed[4:6]
    ed_day = ed[6:8]
    ed_hour = ed[8:12]
    os.chdir('/home/wrf/Build_WRF/LIBRARIES/WPS/')
    os.system('csh /home/wrf/Build_WRF/LIBRARIES/WPS/link_grib.csh '+
                                                       '/home/wrf/data/grib/')
    nwps = open('/home/wrf/data/namelist.wps', 'r')
    nwps_new = open('/home/wrf/Build_WRF/LIBRARIES/WPS/namelist.wps', 'w')
    template = Template(nwps.read())
    std_date = str(std_year+'-'+std_month+'-'+std_day+'_'+std_hour+':00:00')
    end_date = str(ed_year+'-'+ed_month+'-'+ed_day+'_'+ed_hour+':00:00')
    di = {'start_date':std_date,'end_date':end_date}
    nwps_new.write(template.substitute(di))
    nwps_new.close()
    nwps.close()
    dst = datetime.datetime(int(std_year),int(std_month),int(std_day),int(std_hour))
    edst = datetime.datetime(int(ed_year),int(ed_month),int(ed_day),int(ed_hour))
    os.system('/home/wrf/Build_WRF/LIBRARIES/WPS/./ungrib.exe')
    os.system('/home/wrf/Build_WRF/LIBRARIES/WPS/./metgrid.exe')
    os.system('rm /home/wrf/Build_WRF/LIBRARIES/WPS/FILE*')
    os.system('rm /home/wrf/Build_WRF/LIBRARIES/WPS/GRIBFILE*')
    os.system('mv /home/wrf/Build_WRF/LIBRARIES/WPS/met_em*'+
                                   ' /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/')
    nwrf = open('/home/wrf/data/namelist.input','r')
    nwrf_new = open('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/namelist.input','w')
    template = Template(nwrf.read())
    dif = edst-dst
    di = {'run_hours':dif.days*24+dif.seconds//3600,'start_year':std_year,'start_month':std_month,'start_day':std_day,'start_hour':std_hour,'end_year':ed_year,'end_month':ed_month,'end_day':ed_day,'end_hour':ed_hour}
    nwrf_new.write(template.substitute(di))
    nwrf_new.close()
    nwrf.close()
    os.chdir('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/')
    os.system('/home/wrf/Build_WRF/LIBRARIES/WRFV3/run/./real.exe')
    os.environ['OMP_NUM_THREADS'] = '18'
    os.system('echo $OMP_NUM_THREADS')
    os.system('mpirun -np 4 /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/./wrf.exe')
    os.system('rm /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/met_em*')
    os.system('rm /home/wrf/data/grib/*')
