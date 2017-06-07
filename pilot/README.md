# BDE SC5 PILOT V2

#### Setup
```sh
$ cd sc5_sextant
$ docker build . -t sc5_sextant
$ cd ..
$ docker-compose up -d
$ docker -D exec -it sc5_sextant bash /sc5.sh
$ docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/dfs_init.py
```

#### Ingest data
```sh
$ docker cp <netcdf_weather_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /bde-pilot-2/pilot/detection_listener
#
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_weather.py -i <netcdf_weather_files>/
docker cp <netcdf_dispersion_files>/ sc5_sextant:/pilot_data/
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_cluster.py -i <netcdf_dispersion_files>/ -m '<clustering_method>' -d '<descriptor>' -hp <hdfs_path>
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_model.py -i <model_template.zip> -m "<clustering_method>"
```

#### Info
- <netcdf_weather_files>: NetCDF files containing 3 days worth of six hours time frames. These files are used as the current weather in order to perform source estimation.
- ```sh
        Recommended data sources: ECMWF,NCAR
        Recommended structure: ERA-Interim
    ```
- <netcdf_dispersion_files>: NetCDF files that contain dispersions for different clustering methods,descriptors and configurations. Usually these files are the output of an atmospheric dispersion model i.e HYSPLIT,DIPCOT

- <model_template.zip>: zip files that contain the NeuralNetwork model that is used for source estimation. Usually exported from model_template class and neural network scripts.

- <clustering_method>: Clustering configuration i.e shallow_ae (Single autoencoder), deep_ae (Stacked autoencoders)

- <descriptor>: descriptor used for clustering_method i.e km2 (double kmeans), dense (density-based descriptors)
