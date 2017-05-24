FROM bde2020/sextant

ENV PORT 8080
ENV PORT2 5000


RUN apt-get update && \
    apt-get install -y git python-pip libpq-dev python-dev libhdf5-serial-dev netcdf-bin libnetcdf-dev gdal-bin libgdal-dev python-gdal libblas-dev liblapack-dev libatlas-base-dev gfortran && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /pilot_data


ADD sc5.sh /sc5.sh
RUN chmod +x /*.sh

EXPOSE $PORT
EXPOSE $PORT2

CMD ["/bin/bash", "/sextant.sh"]
