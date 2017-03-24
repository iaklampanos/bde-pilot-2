var evtSource;

function loadMapChangeset() {
	$('#modalChangeset').on('shown.bs.modal', function () {
		trigger = false;
		document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(185, 106, 139, 0.7)';
		
		resetChangesetMapForm();
		initSearchMapChangeset();
		document.getElementById('searchBoxChangeSet').value = "";
	});
}

function initSearchMapChangeset() {
	//document.getElementById('searchMapExtentFormChangeset').style.display = 'block';
	//document.getElementById('drawExtentButtonChangeset').disabled = true;
	
	//Initialize map
	//var currentView = map.getView().getCenter();
	var currentZoom = map.getView().getZoom();
	mapFilter = new ol.Map({
        layers: [bingAerialLabels, vector],
        target: 'mapExtentChangeset',
        view: map.getView()
    });
	//mapFilter.getView().setZoom(currentZoom-1);
	
	document.getElementsByClassName('ol-zoom')[0].style.top = '10px';
	document.getElementsByClassName('ol-zoom')[0].style.left = '10px';
   
    addInteraction();
    
    document.getElementById('map_canvas').style.display = 'block';
	document.getElementById('map_canvas2').style.display = 'none';
}

function resetChangesetMapForm() {	
	//document.getElementById('searchMapExtentFormChangeset').style.display = 'none';
	//document.getElementById('drawExtentButtonChangeset').disabled = false;
	
	var divRef = document.getElementById('mapExtentChangeset');
	while (divRef.firstChild) {
		divRef.removeChild(divRef.firstChild);
	}
	
	document.getElementById('changesetSearchForm').reset();
	//clearChangesetFilterForm();
	
	//vector.getSource().clear();
}

function getChangesetResults() {
	var searchTitle = document.getElementById('changesetTitle').value;
	if (searchTitle == "") {
		searchTitle = 'null';
	}
	
	var eventDate = document.getElementById('startDateChangeset').value;
    var referenceDate = document.getElementById('endDateChangeset').value;
    
    if (eventDate == "") {
    	eventDate = 'null';
    }
    else {
    	eventDate = new Date(eventDate);
    	eventDate = eventDate.toISOString();
    }
    
    if (referenceDate == "") {
    	referenceDate = 'null';
    }
    else {
    	referenceDate = new Date(referenceDate);
    	referenceDate = referenceDate.toISOString();
    }
    
    var user = document.getElementById('changesetUsername').value;
    if (user == "") {
    	user = 'null';
    }
    var pass = document.getElementById('changesetPassword').value;
    if (pass == "") {
    	pass = 'null';
    }
    var polar = document.getElementById('polarization').value;
    if (polar == "") {
    	polar = 'null';
    }
    
	var extent = 'null';
	if (vector.getSource().getFeatures().length > 0) {
		extent = vector.getSource().getExtent();
		var extent4326 = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
		extent = mapExtentToWKTLiteralWGS84(extent4326);	
	}
	
	if (searchTitle == 'null') {
		document.getElementById('alertMsgNoLayerName').style.display = 'block';
        setTimeout(function() {$('#alertMsgNoLayerName').fadeOut('slow');}, fadeTime);
	}
	else {
		if (extent == 'null') {
			document.getElementById('alertMsgBBOXNoValue').style.display = 'block';
	        setTimeout(function() {$('#alertMsgBBOXNoValue').fadeOut('slow');}, fadeTime);
		}
		else {		
			console.log('START: '+eventDate);
			console.log('END: '+referenceDate);
			console.log('EXTENT: '+extent);
			
			showSpinnerChangeset(colorSpin);	
			var newElementMsg = document.getElementById('alertMsgChangeset');			  
		    newElementMsg.innerHTML = "<strong>Changeset Detection: </strong> Fetching Images.";
		    document.getElementById('alertMsgChangeset').style.display = 'block';
			
		    
		    var url = '?extent='+encodeURIComponent(extent);
		    
			if (referenceDate != 'null') {
				url = url + '&reference_date='+encodeURIComponent(referenceDate);
			}
			if (eventDate != 'null') {
				url = url + '&event_date='+encodeURIComponent(eventDate);
			}
			
			url = url + '&username='+encodeURIComponent(user)+'&password='+encodeURIComponent(pass);
			
			if (polar != 'null') {
				url = url + '&polarization='+encodeURIComponent(polar);
			}
			else {
				url = url + '&polarization=HH';
			}
		    console.log(url);
		    $.ajax({
		        type: 'GET',
		        url: rootURL + '/imageAggregator'+url,
		        headers: {
		        	//'Accept-Charset' : 'utf-8',
		        	'Content-Type'   : 'text/plain; charset=utf-8',
		        },
		        timeout: 0,
		        searchTitle: searchTitle,
		        success: parseChangeset,
		        error: printError
		    });
			/*if(typeof(EventSource) !== "undefined") {
				var lastMsg = false;				
			    var source = new EventSource(url);
			    source.onmessage = function(event) {
			    	var newElementMsg = document.getElementById('alertMsgChangeset');
			    	
			    	if (!lastMsg) {
			    		newElementMsg.className = "alert alert-info";
					    
					    if (event.data == 'HTTP 401 Unauthorized') {
					    	newElementMsg.className = "alert alert-danger";
					    	newElementMsg.innerHTML = "<strong>Error!</strong> Unauthorized access to Copernicus SCIHUB.";
					    }
					    else if (event.data == 'No images were found for the specified parameters.') {
					    	newElementMsg.className = "alert alert-danger";
					    	newElementMsg.innerHTML = "<strong>Error!</strong> " + event.data;
					    }
					    else {
						    newElementMsg.innerHTML = "<strong>Please wait!</strong> " + event.data;
					    }
					    
					    document.getElementById('alertMsgChangeset').style.display = 'block';
			    	}	   
				    
			    	if (event.data == 'HTTP 401 Unauthorized') {
			    		source.close();
				    	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
				    	hideSpinner();
				    	hideSpinnerChangeset();
			    	}
			    	 
			    	if (event.data == 'No images were found for the specified parameters.') {
				    	source.close();
					   	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
					   	hideSpinner();
					   	hideSpinnerChangeset();
				    } 
			    	
			    	//Check flag to close the SSE
				    if (lastMsg) {
				    	source.close();
				    	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
				    	hideSpinner();
				    	hideSpinnerChangeset();
				    	
				    	parseChangeset(event.data, searchTitle);
				    }
				    
				    if (event.data == 'Change detection completed successfully.') {
				    	//Next msg is the data
				    	lastMsg = true;
				    }			    
			    };
			}
			else {
				newElementMsg.className = "alert alert-danger";
		    	newElementMsg.innerHTML = "<strong>Error!</strong> Your browser does not support SSE. Try using Chrome!";
		    	document.getElementById('alertMsgChangeset').style.display = 'block';
		    	hideSpinner();
		    	hideSpinnerChangeset();
			}*/
		}
	}
			
	resetChangesetMapForm();
}

function parseChangeset(results, status, jqXHR) {	
	console.log(results);
	var newElementMsg = document.getElementById('alertMsgChangeset');	
	
	if (results == 'HTTP 401 Unauthorized') {
    	newElementMsg.className = "alert alert-danger";
    	newElementMsg.innerHTML = "<strong>Error!</strong> Unauthorized access to Copernicus SCIHUB.";
    	
    	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
    	hideSpinner();
    	hideSpinnerChangeset();
    	return;
    }
    else if (results == 'No images were found for the specified parameters.') {
    	newElementMsg.className = "alert alert-danger";
    	newElementMsg.innerHTML = "<strong>Error!</strong> " + results;
    	
    	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
    	hideSpinner();
    	hideSpinnerChangeset();
    	return;
    }
	
	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
	hideSpinner();
	hideSpinnerChangeset();
	
	results = '{"changeset":'+results+'}';
	objJSON = JSON.parse(results);
	
	var layerTitle = this.searchTitle;
					
	var tl = new Layer(layerTitle, '', true, 'changeset', '', '', '', '', '', '', '', '', '');
	mapLayers.push(tl); 
	addTableRow(layerTitle, 'changeset');
	
	var jsonObjects = {
		'type': 'FeatureCollection',
	    'crs': {
	    	'type': 'name',
	        'properties': {
	        	'name': 'EPSG:4326'
	        }
	    },
	    'features': []
	};
	
	for (var i=0; i<objJSON.changeset.length; i++) {
		addPolygon(objJSON.changeset[i], jsonObjects);
	}   
	console.log(jsonObjects);
	
	var opt_options = {
            'dataProjection': 'EPSG:4326',				
            'featureProjection': 'EPSG:3857'			
	};
	
	var vectorSource = new ol.source.Vector({
        features: (new ol.format.GeoJSON()).readFeatures(jsonObjects, opt_options)
    });
	
	//Image Vector layer to use WebGL rendering
	var jsonChangeset = new ol.layer.Image({
		title: layerTitle,
        source: new ol.source.ImageVector({
          source: vectorSource,
          style: defaultVectorStyle
        })
    });
	
	map.addLayer(jsonChangeset);
	addLayerToTimeline(jsonChangeset, objJSON.changeset[0].sourceDate, objJSON.changeset[0].targetDate);
	map.getView().fit(jsonChangeset.getSource().getSource().getExtent(), map.getSize());
	
	hideSpinnerChangeset();
	setTimeout(function() {$('#alertMsgChangeset').fadeOut('slow');}, fadeTime);
}

function addPolygon(jsonObject, jsonObjects) {
	console.log(jsonObject);
	var jsonObj = {
	           "type": "Feature", 
	           "properties": {
	        	   "name": jsonObject.area.name,
	        	   "Start Date": jsonObject.sourceDate,
	        	   "End Date": jsonObject.targetDate,
	        	   "objectType": "changeset",
	        	   "Image 1": jsonObject.sourceName,
	        	   "Image 2": jsonObject.targetName
	           },
	           "geometry": jsonObject.area.geometry
	};
	
	jsonObjects.features.push(jsonObj);
}

///////////////////////
function enableRegionChangeset() {
	var country = document.getElementById('layerSpatialFilterChangesetValue1').options[document.getElementById('layerSpatialFilterChangesetValue1').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterChangesetValue2');

	resetSelectForm(divRef, 'Region');
	resetSelectForm(document.getElementById('layerSpatialFilterChangesetValue3'), 'Region Unit');
	resetSelectForm(document.getElementById('layerSpatialFilterChangesetValue4'), 'City');
		
	var res = alasql('SELECT DISTINCT region FROM geodata WHERE country = "' + country + '"');
	res.forEach(function(i) {
		if (i.region != '') {
			element = document.createElement('option');
			element.value = i.region;
			element.innerHTML = i.region;
			divRef.appendChild(element);
		}		
	});		  
	
	document.getElementById('layerSpatialFilterChangesetValue2').disabled = false;
	document.getElementById('layerSpatialFilterChangesetValue3').disabled = true;
	document.getElementById('layerSpatialFilterChangesetValue4').disabled = true;
}

function enableRegionUnitChangeset() {
	var region = document.getElementById('layerSpatialFilterChangesetValue2').options[document.getElementById('layerSpatialFilterChangesetValue2').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterChangesetValue3');

	resetSelectForm(divRef, 'Region Unit');
	resetSelectForm(document.getElementById('layerSpatialFilterChangesetValue4'), 'City');
	
	var res = alasql('SELECT DISTINCT region_unit FROM geodata WHERE region = "' + region + '"');
	res.forEach(function(i) {
		if (i.region_unit != '') {
			element = document.createElement('option');
			element.value = i.region_unit;
			element.innerHTML = i.region_unit;
			divRef.appendChild(element);
		}			
	});		
	
	document.getElementById('layerSpatialFilterChangesetValue3').disabled = false;
	document.getElementById('layerSpatialFilterChangesetValue4').disabled = true;
}

function enableCityChangeset() {
	var regionUnit = document.getElementById('layerSpatialFilterChangesetValue3').options[document.getElementById('layerSpatialFilterChangesetValue3').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterChangesetValue4');

	resetSelectForm(divRef, 'City');	
	
	var res = alasql('SELECT DISTINCT city FROM geodata WHERE region_unit = "' + regionUnit + '"');
	res.forEach(function(i) {
		if (i.city != '') {
			element = document.createElement('option');
			element.value = i.city;
			element.innerHTML = i.city;
			divRef.appendChild(element);
		}
	});		
	
	document.getElementById('layerSpatialFilterChangesetValue4').disabled = false;
}

function enableCountriesChangeset() {
	var res = alasql('SELECT DISTINCT country FROM geodata');
    var divRef = document.getElementById('layerSpatialFilterChangesetValue1');   
    
    resetSelectForm(divRef, 'Country');
   
	res.forEach(function(i) {
			if (i.country != '') {
			var element = document.createElement('option');
			element.value = i.country;
			element.innerHTML = i.country;
			divRef.appendChild(element);
		}			
	});
    
}

function clearChangesetFilterForm() {
	$('#layerSpatialFilterChangesetValue1').get(0).selectedIndex = 0;
	$('#layerSpatialFilterChangesetValue2').get(0).selectedIndex = 0;
	$('#layerSpatialFilterChangesetValue3').get(0).selectedIndex = 0;
	$('#layerSpatialFilterChangesetValue4').get(0).selectedIndex = 0;
	document.getElementById('layerSpatialFilterChangesetValue2').disabled = true;
	document.getElementById('layerSpatialFilterChangesetValue3').disabled = true;
	document.getElementById('layerSpatialFilterChangesetValue4').disabled = true;
}

