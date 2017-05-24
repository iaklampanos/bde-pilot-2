var mapFilter;
var draw;
var trigger = false;

var source = new ol.source.Vector({wrapX: false});

var base = new ol.layer.Tile({
    source: new ol.source.MapQuest({layer: 'sat'})
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
          color: '#ffcc33'
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

function addInteractionMainMap() {  
	if (trigger) {
		trigger = false;
		document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(185, 106, 139, 0.7)';
		document.getElementById('map_canvas').style.display = 'block';
		document.getElementById('map_canvas2').style.display = 'none';
	}
	else {
		trigger = true;
		document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(38, 166, 154, 0.7)';
		
		vector.getSource().clear();
		document.getElementById('map_canvas').style.display = 'none';
		document.getElementById('map_canvas2').style.display = 'block';
	
		//Initialize map
		var baseType = getBaseMapType();
		//var currentView = map.getView().getCenter();
		//var currentZoom = map.getView().getZoom();
		mapFilter = new ol.Map({
	        layers: [baseType, vector],
	        target: 'map_canvas2',
	        view: map.getView()
	    });
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
	switch(baseMapType) {
	case 'OSM': return baseOSM;
	case 'bing': return bingMap;
	case 'aerial': return bingAerialLabels;
	case 'road': return bingRoads;
	}
}