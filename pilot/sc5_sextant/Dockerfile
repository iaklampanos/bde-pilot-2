FROM bde2020/sextant

ENV PORT 8080


RUN apt-get update && \
    apt-get install -y git python-pip libpq-dev python-dev libhdf5-serial-dev netcdf-bin libnetcdf-dev gdal-bin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD sc5.sh /sc5.sh
RUN chmod +x /*.sh

EXPOSE $PORT

CMD ["/bin/bash", "/sextant.sh"]
