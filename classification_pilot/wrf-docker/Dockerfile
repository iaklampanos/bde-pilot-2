FROM rdccosmo/wrf-arwpost
USER root
RUN apt-get update
RUN apt-get install -y nano sudo npm
RUN mkdir /home/wrf/data
RUN mkdir /home/wrf/data/grib
RUN sed -i "s/geog_data_path = .*/geog_data_path = '\/home\/wrf\/data\/'/" /home/wrf/Build_WRF/LIBRARIES/WPS/namelist.wps
VOLUME /home/wrf/data
COPY geo_em.d01.nc /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY METGRID.TBL /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY Vtable /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY grib/ /home/wrf/data/grib/
COPY namelist.input /home/wrf/data/
COPY namelist.wps /home/wrf/data/
COPY process_dataset.py /home/wrf/data/
