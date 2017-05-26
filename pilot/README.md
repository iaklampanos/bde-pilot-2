# BDE SC5 PILOT V2

### Run

```sh
cd sc5_sextant
docker build . -t sc5_sextant
cd ..
docker-compose up
docker -D exec -it sc5_sextant bash /sc5.sh
```

### Ingest data
```sh
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/dfs_init.py
docker cp <netcdf_weather_files>/ sc5_sextant:/pilot_data/
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_weather.py -i <netcdf_weather_files>/
docker cp <netcdf_dispersion_files>/ sc5_sextant:/pilot_data/
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_cluster.py -i <netcdf_dispersion_files>/ -m '<clustering_method>' -d '<descriptor>' -hp <hdfs_path>
docker -D exec -it sc5_sextant python /bde-pilot-2/pilot/detection_listener/ingest_model.py -i <model_template.zip> -m "<clustering_method>"
```
