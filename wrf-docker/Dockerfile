FROM rdccosmo/wrf-arwpost
USER root
RUN apt-get update
RUN apt-get install -y nano sudo npm
RUN apt-get install -y nco
RUN apt-get install -y libtool-bin
RUN apt-get install -y gdb
RUN apt-get install -y openmpi-bin
RUN mkdir /home/wrf/data
RUN mkdir /home/wrf/data/grib
RUN mkdir /home/wrf/data/nc/
RUN mkdir /home/wrf/Build_WRF/LIBRARIES/DIPCOT/
RUN mkdir /home/wrf/Build_WRF/LIBRARIES/FILMAKER/
RUN sed -i "s/geog_data_path = .*/geog_data_path = '\/home\/wrf\/data\/'/" /home/wrf/Build_WRF/LIBRARIES/WPS/namelist.wps
VOLUME /home/wrf/data
COPY geo_em.d01.nc /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY METGRID.TBL /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY Vtable /home/wrf/Build_WRF/LIBRARIES/WPS/
COPY grib/ /home/wrf/data/grib/
COPY tempnamelist.input /home/wrf/data/
COPY namelist.wps /home/wrf/data/
COPY process_dataset.py /home/wrf/data/
COPY ncsplitcluster.sh /home/wrf/data/
COPY template.sh /home/wrf/data/
COPY preprocess_scripts/ /home/wrf/data/

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

