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
# ls -d /pilot_data/<netcdf_weather_files_dir>/* >> weather.txt
# cat weather.txt | xargs -n <# of files per processor> -P <# of processors> python ingest_weather.py
# exit
```
#### Ingest neural network models
```sh
$ docker cp <model_template.zip> sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /bde-pilot-2/pilot/detection_listener
# python ingest_model.py -i /pilot_data/<model_template.zip> -m "<clustering/classification method>" -ht "<html_repr>"
# exit
```

#### Ingest cluster dispersions
```sh
$ docker cp <netcdf_dispersion_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /bde-pilot-2/pilot/detection_listener
# python ingest_cluster.py -i <netcdf_dispersion_files_dir>/ -m '<clustering_method>' -d '<descriptor>' -hp <hdfs_path>
# exit
```

#### Ingest classification dispersions
```sh
$ docker cp <netcdf_dispersion_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /bde-pilot-2/pilot/detection_listener
# ls -d /pilot_data/<netcdf_dispersion_files_dir>/* >> classes.txt
# cat classes.txt | xargs -n <# of files per processor> -P <# of processors> python ingest_class.py
# exit
```

#### RUN
```sh
$ docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/listener.py
```

#### Info
- <netcdf_weather_files>: NetCDF files containing 3 days worth of six hours time frames. These files are used as the current weather in order to perform source estimation.
  - ```sh
        Recommended data sources: ECMWF,NCAR
        Recommended structure: ERA-Interim
    ```
- <netcdf_dispersion_files>: NetCDF files that contain dispersions for different clustering methods,descriptors and configurations. Usually these files are the output of an atmospheric dispersion model i.e HYSPLIT,DIPCOT

- <model_template.zip>: zip files that contain the NeuralNetwork model that is used for source estimation. Usually exported from model_template class and neural network scripts.

- <clustering_method>: Clustering configuration i.e shallow_ae (Single autoencoder), deep_ae (Stacked autoencoders), etc.

- <descriptor>: descriptor used for clustering_method i.e km2 (double kmeans), dense (density-based descriptors)
