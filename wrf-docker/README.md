# Modified WRF-Docker used in SC5-pilot-3

## Setup
```
$ cd wrf-docker
$ docker build . -t wrf
```
## Run
```
$ docker run -d -P --name wrf wrf
$ docker port wrf 22
```

## File Information
#### WPS additional files
```
geo_em.d01.nc
METGRID.TBL
namelist.wps
Vtable
```
Configuration files used in WPS for our experiments, that are not included in the dockerized version.

#### WRF additional files
```
tempnamelist.input
ncsplitcluster.sh
template.sh
```
Configuration files used in WRF for our experiments, that are not included in the dockerized version.

#### Additional script files
Some additional scripts are included for specific processes but are not required in order to run WPS/WRF
* ```process_dataset.py``` is the main script used for converting NCAR grib files to netCDF4 (retrieves files from /home/wrf/data/grib/ runs WPS and then WRF
* ```split.py``` main script used when NetCDF data are available and we want to only run WRF (it is called split because we need to split the files using ```ncsplitcluster.sh```)
* ```breakdown.py``` breaks down a NetCDF file containing multiple timeframes to 3day periods
* ```cdo_merge_ncar.py``` merges gribs that contain multiple pressure levels and surface level into one grib (expects 2 txt files with list of surface and pressure level grib files)
* ```cluster_fix.py``` used on descriptors created from kmeans clusters in order convert them into netCDF3 format
* ```ncar_to_ecmwf.py``` used after WPS on combined netCDF files in order to output a NetCDF with 'ECMWF type' structrure


# Original based docker md
https://github.com/rdccosmo/containered-wrf-data
> #rdccosmo/containered-wrf-data
This project aims to dockerize the [WRF (Weather Research and Forecast Model)](http://www.wrf-model.org/index.php).
>
>It follows the instructions from this [tutorial](http://www2.mmm.ucar.edu/wrf/OnLineTutorial/compilation_tutorial.php).
>It is part of a series of projects each providing a Dockerfile with parts of the instructions:
> 1. [rdccosmo/containered-wrf-base](https://github.com/rdccosmo/containered-wrf-base)
> 2. [rdccosmo/containered-wrf-hdf5](https://github.com/rdccosmo/containered-wrf-hdf5)
> 3. [rdccosmo/containered-wrf-netcdf](https://github.com/rdccosmo/containered-wrf-netcdf)
> 4. [rdccosmo/containered-wrf](https://github.com/rdccosmo/containered-wrf)
> 5. [rdccosmo/containered-wps](https://github.com/rdccosmo/containered-wps)
> 6. [rdccosmo/containered-wrf-arwpost](https://github.com/rdccosmo/containered-wrf-arwpost)
> 7. [rdccosmo/containered-wrf-data](https://github.com/rdccosmo/containered-wrf-data)
```
