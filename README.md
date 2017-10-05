# BDE SC5 PILOT #3

#### Setup
```sh
$ cd sc5_sextant-docker
$ docker build . -t sc5_sextant
$ cd ..
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
