TIFF_IN=


clean::
	rm *.tiff
	rm *.xml

png::
	gdalwarp -t_srs EPSG:4326 $(TIFF_IN) temp1.tiff
	gdal_translate -a_nodata 0 temp1.tiff temp2.tiff
	gdal_polygonize.py -f "GeoJSON" temp2.tiff `echo $(TIFF_IN)|awk -F "." '{print $$1}'`.json
	rm temp1.tiff
	rm temp2.tiff

