# BDE SC5 PILOT #3

The pilot is carried out by NCSR-D in the frame of SC5 Climate Action, Environment, Resource Efficiency and Raw Materials.

The pilot demonstrates the following workflow: A (potentially hazardous) substance is released in the atmosphere that results to increased readings in one or more monitoring stations. The user accesses a user interface provided by the pilot to define the locations of the monitoring stations as well as a timeseries of the measured values (e.g. gamma dose rate). The platform initiates
- a weather matching algorithm, that is a search for similarity of the current weather and the pre-computed weather patterns, as well as
- a dispersion matching algorithm, that is a search for similarity of the current substance dispersion patterns with the precomputed ones.
- Semanticly-aware querying to enrich dispersion patterns with additional information such as affected population and nearby hospitals
- A uniform perspective of heterogeneous data stored in heterogeneous data management and processing infrastructures using SemaGrow.

# Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=ZtjZhNviEwo
" target="_blank"><img src="https://img.youtube.com/vi/ZtjZhNviEwo/0.jpg"
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


#### Setup
```sh
$ cd sc5_sextant-docker
$ docker build . -t sc5_sextant
$ cd ..
$ docker-compose up -d
$ cd osm-docker
$ docker-compose up -d
$ cd ../geonames-docker
$ docker-compose up -d
```

#### RUN
```sh
After data ingestion
$ docker -D exec -it sc5_sextant bash /pilot-sc5-cycle3/backend/sc5_backend.sh
```

#### Info
- ```netcdf_weather_files```: NetCDF files containing 3 days worth of six hours time frames. These files are used as the current weather in order to perform source estimation.
  - ```sh
        Recommended data sources: ECMWF,NCAR
        Recommended structure: ERA-Interim
    ```
- ```netcdf_dispersion_files```: NetCDF files that contain dispersions for different clustering methods,descriptors and configurations. Usually these files are the output of an atmospheric dispersion model i.e HYSPLIT,DIPCOT.

- ```model_template.zip```: zip files that contain the NeuralNetwork model that is used for source estimation. Usually exported from model_template class and neural network scripts.

- ```clustering_method```: Clustering configuration i.e shallow_ae (Single autoencoder), deep_ae (Stacked autoencoders), etc.

- ```descriptor``` : descriptor used for clustering_method i.e km2 (double kmeans), dense (density-based descriptors).

- ```html_repr```: String that presents the estimation method to the end user.
