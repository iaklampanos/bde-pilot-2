#### Ingest neural network models
```sh
$ docker cp <model_template.zip> sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /pilot-sc5-cycle3/data_ingest
# python ingest_model.py -i /pilot_data/<model_template.zip> -m "<clustering/classification method>" -ht "<html_repr>"
# exit
```

#### Ingest weather files
```sh
$ docker cp <netcdf_weather_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /pilot-sc5-cycle3/data_ingest
# ls -d /pilot_data/<netcdf_weather_files_dir>/* >> weather.txt
# cat weather.txt | xargs -n <# of files per processor> -P <# of processors> python ingest_weather.py
# exit
```

#### Ingest cluster dispersions files
```sh
$ docker cp <netcdf_dispersion_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /pilot-sc5-cycle3/data_ingest
# python ingest_cluster.py -i <netcdf_dispersion_files_dir>/ -m '<clustering_method>' -d '<descriptor>' -hp <hdfs_path>
# exit
```

#### Ingest classification dispersions files
```sh
$ docker cp <netcdf_dispersion_files_dir>/ sc5_sextant:/pilot_data/
$ docker -D exec -it sc5_sextant bash
# cd /pilot-sc5-cycle3/data_ingest
# ls -d /pilot_data/<netcdf_dispersion_files_dir>/* >> classes.txt
# cat classes.txt | xargs -n <# of files per processor> -P <# of processors> python ingest_class.py
# exit
```
