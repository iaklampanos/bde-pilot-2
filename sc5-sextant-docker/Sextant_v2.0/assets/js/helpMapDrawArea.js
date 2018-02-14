var mapFilter;
var draw;
var trigger = false;

var source = new ol.source.Vector({
    wrapX: false
});

var base = new ol.layer.Tile({
    source: new ol.source.MapQuest({
        layer: 'sat'
    })
});

var vector = new ol.layer.Vector({
    source: source,
    style: new ol.style.Style({
        fill: new ol.style.Fill({
            color: 'rgba(255, 255, 255, 0.2)'
        }),
        stroke: new ol.style.Stroke({
            color: '#ffcc33',
            width: 2
        }),
        image: new ol.style.Circle({
            radius: 7,
            fill: new ol.style.Fill({
                color: '#ff6347'
            })
        })
    })
});

function addInteraction() {
    var geometryFunction, maxPoints;
    var value = 'LineString';
    maxPoints = 2;
    geometryFunction = function(coordinates, geometry) {
        if (!geometry) {
            geometry = new ol.geom.Polygon(null);
        }
        var start = coordinates[0];
        var end = coordinates[1];
        geometry.setCoordinates([
            [start, [start[0], end[1]], end, [end[0], start[1]], start]
        ]);
        return geometry;
    };

    draw = new ol.interaction.Draw({
        source: source,
        type: /** @type {ol.geom.GeometryType} */ (value),
        geometryFunction: geometryFunction,
        maxPoints: maxPoints
    });
    mapFilter.addInteraction(draw);

    draw.on('drawstart', function(evt) {
        vector.getSource().clear();
    });
}

function mapClear(){
  vector.getSource().clear();
  clearDispersion();
  clearWindDir();
  removeSelect();
  drawStations();
  drawNetworks();
  addSelect();
}

function pointClear(){
   vector.getSource().clear();
   clearDispersion();
   removeSelect();
   drawStations();
   drawNetworks();
   addSelect();
}

function PaddInteractionMainMap() {
    if (trigger) {
        trigger = false;
        document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(185, 106, 139, 0.7)';
        // document.getElementById('map_canvas').style.display = 'block';
        // document.getElementById('map_canvas2').style.display = 'none';
        mapFilter.removeInteraction(draw);
    } else {
        clearDispersion();
        trigger = true;
        document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(38, 166, 154, 0.7)';
        // vector.getSource().clear();
        removeSelect();
        var value = 'Point';

        draw = new ol.interaction.Draw({
            source: source,
            type: /** @type {ol.geom.GeometryType} */ (value),
            //geometryFunction: geometryFunction,
            //maxPoints: maxPoints
        });
        mapFilter.addInteraction(draw);
        draw.on('drawend', function(evt) {
            var feature = evt.feature;
            var p = feature.getGeometry();
            feature.setId("detection"+ol.proj.transform(p.getCoordinates(), 'EPSG:3857', 'EPSG:4326'))
            // ol.proj.transform(p.getCoordinates(), 'EPSG:3857', 'EPSG:4326');
        });

    }
}

function mapF() {
  vector.getSource().clear();
  document.getElementById('map_canvas').style.display = 'none';
  document.getElementById('map_canvas2').style.display = 'block';
  //Initialize map
  var baseType = getBaseMapType();
  //var currentView = map.getView().getCenter();
  //var currentZoom = map.getView().getZoom();
  mouseControl = new ol.control.MousePosition({
      coordinateFormat: ol.coordinate.createStringXY(4),
      projection: 'EPSG:4326',
      target: document.getElementById('coordinates'),
      undefinedHTML: '&nbsp;'
  });

  var scaleLineControl = new ol.control.ScaleLine();

  mapFilter = new ol.Map({
      layers: [baseType, vector],
      target: 'map_canvas2',
      view: map.getView(),
      controls: ol.control.defaults().extend([mouseControl, scaleLineControl])
  });
  //mapFilter.getView().setZoom(currentZoom);
}


function drawCircle(){
  var value = 'Circle';
  var geometryFunction = function(coordinates, opt_geometry) {
      var extent = ol.extent.boundingExtent(coordinates);
      var geometry = opt_geometry || new ol.geom.Polygon(null);
      geometry.setCoordinates([[
        ol.extent.getBottomLeft(extent),
        ol.extent.getBottomRight(extent),
        ol.extent.getTopRight(extent),
        ol.extent.getTopLeft(extent),
        ol.extent.getBottomLeft(extent)
      ]]);
      return geometry;
  };
  removeSelect();
  draw = new ol.interaction.Draw({
          source: source,
          type: /** @type {ol.geom.GeometryType} */ (value),
          geometryFunction: geometryFunction
  });
  mapFilter.addInteraction(draw);
  draw.on('drawend', function(evt) {
        var extent = evt.feature.getGeometry().getExtent();
        var style =  new ol.style.Style({
                fill: new ol.style.Fill({
                    color: 'rgba(255, 255, 255, 0.2)'
                }),
                stroke: new ol.style.Stroke({
                    color: '#ffcc33',
                    width: 1
                }),
                image: new ol.style.Circle({
                    radius: 5,
                    fill: new ol.style.Fill({
                        color: '#ff6347'
                    })
                })
        });
        vector.getSource().forEachFeatureIntersectingExtent(extent,function(feature) {
          try{
              var id = feature.getId();
              var div = document.getElementById(id);
              div.style.display = 'block';
              feature.setStyle(style);
              feature.setId("detection_"+id);
            }
            catch(err) {
              ;
            }
        });
        mapFilter.removeInteraction(draw);
        addSelect();
      });
}

function addInteractionMainMap() {
    if (trigger) {
        trigger = false;
        document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(185, 106, 139, 0.7)';
        document.getElementById('map_canvas').style.display = 'block';
        document.getElementById('map_canvas2').style.display = 'none';
    } else {
        trigger = true;
        document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(38, 166, 154, 0.7)';

        vector.getSource().clear();
        document.getElementById('map_canvas').style.display = 'none';
        document.getElementById('map_canvas2').style.display = 'block';

        //Initialize map
        var baseType = getBaseMapType();
        var points = new OpenLayers.Layer.PointGrid({
            isBaseLayer: true,
            dx: 10,
            dy: 10
        });
        points.setMaxFeatures(501);
        //var currentView = map.getView().getCenter();
        //var currentZoom = map.getView().getZoom();
        // mapFilter = new ol.Map({
        //     layers: [baseType, vector, points],
        //     target: 'map_canvas2',
        //     view: map.getView()
        // });
        //mapFilter.getView().setZoom(currentZoom);

        var geometryFunction, maxPoints;
        var value = 'LineString';
        maxPoints = 2;
        geometryFunction = function(coordinates, geometry) {
            if (!geometry) {
                geometry = new ol.geom.Polygon(null);
            }
            var start = coordinates[0];
            var end = coordinates[1];
            geometry.setCoordinates([
                [start, [start[0], end[1]], end, [end[0], start[1]], start]
            ]);
            return geometry;
        };

        draw = new ol.interaction.Draw({
            source: source,
            type: /** @type {ol.geom.GeometryType} */ (value),
            geometryFunction: geometryFunction,
            maxPoints: maxPoints
        });
        mapFilter.setSize([250, 250]);
        mapFilter.addInteraction(draw);


        draw.on('drawstart', function(evt) {
            vector.getSource().clear();
        });

    }
}

function clearMapInteraction() {
    map.removeInteraction(draw);
}

function getBaseMapType() {
    switch (baseMapType) {
        case 'OSM':
            return baseOSM;
        case 'bing':
            return bingMap;
        case 'aerial':
            return bingAerialLabels;
        case 'road':
            return bingRoads;
    }
}


function clearDispersion(){
  mapFilter.getLayers().forEach(function(layer) {
    try{
        if (layer.get('title').indexOf('dispersion') !== -1) {
            mapFilter.removeLayer(layer);
        }
      } catch(e) {
        // pass
      }
  });
}

function clearWindDir(){
  mapFilter.getLayers().forEach(function(layer) {
      if (layer.get('title') == 'wind_direction') {
          mapFilter.removeLayer(layer);
      }
  });
}

function getDrawnDispersionId(){
  var id;
  mapFilter.getLayers().forEach(function(layer) {
    try{
        if (layer.get('title').indexOf('dispersion') !== -1) {
            var tokens = layer.get('title').split("_");
            id = parseInt(tokens[1]);
        }
      } catch(e) {
        // pass
      }
  });
  return id;
}

function clearPopGrid(){
  vector.getSource().forEachFeature(function(feature) {
    try{
        var id = feature.getId();
        if (id.indexOf('POP_') !== -1){
           vector.getSource().removeFeature(feature);
        }
      }
    catch(err){ ; }
    });
}

function clearHospGrid(){
  vector.getSource().forEachFeature(function(feature) {
    try{
        var id = feature.getId();
        if (id.indexOf('HOSP_') !== -1){
           vector.getSource().removeFeature(feature);
        }
      }
    catch(err){ ; }
    });
}
