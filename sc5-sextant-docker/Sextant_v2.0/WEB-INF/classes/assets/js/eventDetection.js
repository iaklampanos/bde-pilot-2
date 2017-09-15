var eventAreas = [];
var eventResources = [];

function loadMapEvents() {
	$('#modalEventDetection').on('shown.bs.modal', function () {
		trigger = false;
		document.getElementById('drawExtentMainMap').style.backgroundColor = 'rgba(185, 106, 139, 0.7)';
		
		resetEventDetectionMapForm();
		initSearchMapEventDetection();
		document.getElementById('searchBoxEvent').value = "";
	});
}

function initSearchMapEventDetection() {
	//document.getElementById('searchMapExtentFormEventDetection').style.display = 'block';
	//document.getElementById('drawExtentButtonEventDetection').disabled = true;
	
	//Initialize map
	//var currentView = map.getView().getCenter();
	var currentZoom = map.getView().getZoom();
	mapFilter = new ol.Map({
        layers: [bingAerialLabels, vector],
        target: 'mapExtentEventDetection',
        view: map.getView()
    });
	//mapFilter.getView().setZoom(currentZoom-1);
	
	document.getElementsByClassName('ol-zoom')[0].style.top = '10px';
	document.getElementsByClassName('ol-zoom')[0].style.left = '10px';
   
    addInteraction();
    
    document.getElementById('map_canvas').style.display = 'block';
	document.getElementById('map_canvas2').style.display = 'none';
}

function resetEventDetectionMapForm() {	
	//document.getElementById('searchMapExtentFormEventDetection').style.display = 'none';
	//document.getElementById('drawExtentButtonEventDetection').disabled = false;
	
	var divRef = document.getElementById('mapExtentEventDetection');
	while (divRef.firstChild) {
		divRef.removeChild(divRef.firstChild);
	}
	
	document.getElementById('eventDetectionSearchForm').reset();
	//clearEventDetectionFilterForm();
	
	//vector.getSource().clear();
}

function getEventDetectionResults() {
	var KEYS = [];
	var keyWords = document.getElementById('eventDetectionSearchKeys').value;
    if (keyWords != "") {
    	var keyArray = keyWords.split(' ');
	    for (var i=0; i<keyArray.length; i++) {
	    	KEYS.push(keyArray[i].toString()); 
	    }
    }
    else {
    	KEYS.push('null');
    }
    
	
	var eventDate = document.getElementById('startDateEventDetection').value;
    var referenceDate = document.getElementById('endDateEventDetection').value;
    
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
	
	
	var extent = 'null';
	if (vector.getSource().getFeatures().length > 0) {
		extent = vector.getSource().getExtent();
		var extent4326 = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
		extent = mapExtentToWKTLiteralWGS84(extent4326);	
	}
	
	if (extent == 'null' && eventDate == 'null' && referenceDate == 'null' && KEYS.toString() == 'null') {
		document.getElementById('alertMsgNoParams').style.display = 'block';
        setTimeout(function() {$('#alertMsgNoParams').fadeOut('slow');}, fadeTime);
	}
	else {	
		console.log('KEYS: '+KEYS.toString());
		console.log('START: '+eventDate);
		console.log('END: '+referenceDate);
		console.log('EXTENT: '+extent);
		
		var url = 'http://teleios4.di.uoa.gr:8080/changeDetection/event/search?';
		var firstParam = 0;
		if (extent != 'null') {
			url = url + 'extent='+encodeURIComponent(extent);
			firstParam = 1;
		}
		if (KEYS.toString() != 'null') {
			if (firstParam) {
				url = url + '&keys='+encodeURIComponent(KEYS.toString());
			}
			else {
				url = url + 'keys='+encodeURIComponent(KEYS.toString());
				firstParam = 1;
			}
		}
		if (eventDate != 'null') {
			if (firstParam) {
				url = url + '&event_date='+encodeURIComponent(eventDate);
			}
			else {
				url = url + 'event_date='+encodeURIComponent(eventDate);
				firstParam = 1;
			}
		}
		if (referenceDate != 'null') {
			if (firstParam) {
				url = url + '&reference_date='+encodeURIComponent(referenceDate);
			}
			else {
				url = url + 'reference_date='+encodeURIComponent(referenceDate);
				firstParam = 1;
			}
		}
		
		document.getElementById('alertMsgServerWait').style.display = 'block';
		showSpinner(colorSpin);
		$.ajax({
		    type: 'GET',
		    //url: 'http://popeye.di.uoa.gr:8080/changeDetection/event/search?extent='+encodeURIComponent(extent)+'&keys='+encodeURIComponent(KEYS.toString())+'&event_date='+encodeURIComponent(eventDate)+'&reference_date='+encodeURIComponent(referenceDate),		    
		    url: url,
		    crossDomain: true,		    
		    timeout: ajaxTimeout,
		    success: parseEventDetectionResults,
		    error: printError
		});
		
	}
			
	resetEventDetectionMapForm();
}

function parseEventDetectionResults(results, status, jqXHR) {
	if (results.length == 0) {
		hideSpinner();
	    setTimeout(function() {$('#alertMsgServerWait').fadeOut('slow');}, fadeTimeFast);
	    
	    document.getElementById('alertMsgNoEvent').style.display = 'block';
	    setTimeout(function() {$('#alertMsgNoEvent').fadeOut('slow');}, fadeTime);
	    return;
	}
	
	results = '{"events":'+JSON.stringify(results)+'}';
	objJSON = JSON.parse(results);
	
	for (var i=0; i<objJSON.events.length; i++) {
		addEventToPanel(objJSON.events[i]);
	}
	
	hideSpinner();
    setTimeout(function() {$('#alertMsgServerWait').fadeOut('slow');}, fadeTime);
    
}

function addEventToPanel(event) {
	var eventDate = event.eventDate;
	//var referenceDate = event.referenceDate;
	var title = event.title;
	var eventId = event.id;

	var areas = new areasOfEvent(eventId, event.areas);
	eventAreas.push(areas);
	
	var areaOptions = '<option value="" disabled selected>Event Areas</option>';
	for (var i=0; i<event.areas.length; i++) {
		areaOptions += '<option value="'+event.areas[i].name+'">'+event.areas[i].name+'</option>';
	}
	var divRef = document.getElementById('eventTableBody');

	divRef.innerHTML += '<tr id="'+eventId+'">'+
							'<td>'+
								'<div class="row">'+
				      	  	  	'	<div class="col-md-11 col-sm-11">'+
				      	  	    '  		<h4 class="eventTitle">'+title+'</h4>'+
				      	  	    '  	</div>'+
				      	  	    '  <div class="col-md-1 col-sm-1">'+
				       	  	  	'  		<a onClick="deleteEvent(\''+eventId+'\')" title="Delete"><i class="fa fa-times"></i></a>'+		      
				       	  	  	'  </div>'+
						      	'</div>'+
		
								'<div><p><b>Event Date:</b> '+eventDate+'</p></div>'+
								//'<div><p><b>Reference Date:</b> '+referenceDate+'</p></div>'+
								'<select title="Event Areas" id="eventAreaSelect'+eventId+'" class="form-control" onChange="showEventArea(this.id,\''+eventId+'\')">'+areaOptions+'</select>'+
								'<button type="submit" title="Retrieve related sources" class="form-control btn btn-block btn-default" onClick="populateEventResources(\''+eventId+'\')" style="margin-top:5px">View Sources</button>'+
							'</td>'+
						'</tr>';
	
    retrieveRelatedEvents(eventId);
}

function deleteEvent(eventId) {
	//Delete the layer from the map
	map.getLayers().forEach(function(layer) {
    	if (layer.get('title') == 'eventAreas') {
    		map.removeLayer(layer);
    	}
    });
	//CLose popup and clear selected features
    mapSelectInterraction.getFeatures().clear();
    clearPopup();
    
    //Delete event from panel
    var rowIndex = document.getElementById(eventId).rowIndex;    
    document.getElementById('eventTableBody').deleteRow(rowIndex);
    
    //Delete related resources
    var position = -1;
    for (var i=0; i<eventResources.length; i++) {
		if (eventResources[i].id == eventId) {	
			position = i;
			break;
		}
    }
    if (position != -1) {
        eventResources.splice(position, 1);
    }
}

function showEventArea(selectId, eventId) {
	var areaList = document.getElementById(selectId);
	var areaName = areaList.options[areaList.selectedIndex].value;
	var geom = null;
	for (var i=0; i<eventAreas.length; i++) {
		if (eventAreas[i].id == eventId) {
			for (var j=0; j<eventAreas[i].areas.length; j++) {
				if (eventAreas[i].areas[j].name == areaName) {
					geom = eventAreas[i].areas[j].geometry;
					break;
				}
			}
			break;
		}
	}
	
	var title = 'null';
	var description = 'null';
	var eventDate = 'null';
	for (var i=0; i<eventResources.length; i++) {
		if (eventResources[i].id == eventId) {	
			title = eventResources[i].title;
			description = eventResources[i].description;
			eventDate = eventResources[i].date;
			break;
		}
    }
	
	//Clear the  eventAreas layer. Add the new feature and zoom to it
	var jsonObj = {
            "type": "Feature", 
            "properties": {
            	"name": areaName,
            	"Title": title,
            	"Description": description,
            	"Date": eventDate,
            	"objectType": "event"
            },
            "geometry": geom
    };

	var opt_options = {
            'dataProjection': 'EPSG:4326',				
            'featureProjection': 'EPSG:3857'			
	};
	
	//Delete previous layer with events if it exists
	//Delete the layer from the map
	map.getLayers().forEach(function(layer) {
    	if (layer.get('title') == 'eventAreas') {
    		map.removeLayer(layer);
    	}
    });
	//CLose popup and clear selected features
    mapSelectInterraction.getFeatures().clear();
    clearPopup();
	
	var vectorSource = new ol.source.Vector({
        features: (new ol.format.GeoJSON()).readFeatures(jsonObj, opt_options)
    });
	
	//Image Vector layer to use WebGL rendering
	var layer = new ol.layer.Image({
		title: 'eventAreas',
        source: new ol.source.ImageVector({
          source: vectorSource,
          style: defaultVectorStyle
        })
    });
	
	map.addLayer(layer);
	map.getView().fit(layer.getSource().getSource().getExtent(), map.getSize());
	
}

/**
 * Query Semagrow to get the events that are related to the given eventId
 * and populate the Associated Events dropdown list with the results.
 * @param eventId
 */
function retrieveRelatedEvents(eventId) {
	//console.log(eventId);
	//eventId = '2-889345a9890393532b7fb2ac11a04e80-3805';
	//var testEventId = '002-6a14dd0f8222788523d422519dc356b4-4975';

	var query = encodeURIComponent('SELECT * WHERE {?s <http://cassandra.semagrow.eu/bde/events#event_id> ?o .}');
	var query2 = encodeURIComponent("SELECT ?description ?event_date ?event_source_urls ?place_mappings ?tweet_post_ids ?title"+ 
	"WHERE { "+
	"?s <http://cassandra.semagrow.eu/bde/events#event_id> \""+eventId+"\" . "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#title> ?title } "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#description> ?description } "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#event_date> ?event_date } "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#event_source_urls> ?event_source_urls } "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#place_mappings> ?place_mappings } "+
	"OPTIONAL { ?s <http://cassandra.semagrow.eu/bde/events#tweet_post_ids> ?tweet_post_ids } }");
	
	$.ajax({
        type: 'GET',
        url: rootURL + '/semagrow/events?eventId='+encodeURIComponent(eventId),
        //url: 'http://143.233.226.33:8090/SemaGrow/sparql?query='+query2,
        headers: {
        	//'Accept-Charset' : 'utf-8',
        	'Content-Type'   : 'text/plain; charset=utf-8',
        },
        eventId: eventId,
        timeout: ajaxTimeout,
        success: parseEvent,
        error: printError
    });
	
}

function parseEvent(results, status, jqXHR) {
	
	if (results == '') { 
		return;
		
		/*
		results = '{'+
					'"description" : "It also raised its average Brent forecast to $45 per barrel this year, up from $39, while it said West Texas Intermediate would average $45 per barrel this year, up from $38 previously.",'+
					'"eventDate" : "2016-05-23T08:27+0000",'+
					'"eventSourceURLs" : "[http://feeds.reuters.com/~r/reuters/businessNews/~3/_rtH46dTHxk/us-global-oil-idUSKCN0YE01A, http://feeds.reuters.com/~r/reuters/companyNews/~3/BOGbjXtBOQ4/global-oil-idUSL3N18K05Y]",'+
					'"placeMappings" : null,'+
					'"tweetPostIDs" : "[https://twitter.com/BDE/status/712607797809192960, https://twitter.com/BDE/status/712607797809192960]",'+
					'"title" : "Test Event"'+
				   '}';
				   
				   */
	}
	
	var obj = JSON.parse(results);
	
	var event = new eventCassandra(this.eventId, obj.title, obj.description, obj.eventDate, obj.eventSourceURLs, obj.tweetPostIDs);
	eventResources.push(event);	
}

function populateEventResources(eventId) {	
	var articles = null;
	var element1 = document.getElementById('articlesBody');
	element1.innerHTML = '';
	
	var tweeterFeeds = null;
	var element2 = document.getElementById('twitterBody');
	element2.innerHTML = '';
	
	for (var i=0; i<eventResources.length; i++) {
		if (eventResources[i].id == eventId) {			
			if (eventResources[i].articles != null) {
				articles = eventResources[i].articles.replace('[', '').replace(']','').replace('{', '').replace('}', '').split(', http://');
				for (var j=0; j<articles.length; j++) {					
					var parseTitle = articles[j].replace('http://', '').split('=');
					var feedTitle = 'http://'+parseTitle[0];
					element1.innerHTML +='<tr style="text-align: center;"><td><a class="eventResourceURL" href="'+feedTitle+'" target="_blank">'+parseTitle[1]+'</a></td></tr>';
				}
			}
			
			if (eventResources[i].twitterFeeds != null) {
				tweeterFeeds = eventResources[i].twitterFeeds.replace('[', '').replace(']','').replace('{', '').replace('}', '').split(', ');
				for (var j=0; j<tweeterFeeds.length; j++) {
					var parseTitle = tweeterFeeds[j].split('=');
					var author = 'https://twitter.com/BDE/status/'+parseTitle[0];
					
					element2.innerHTML +='<tr style="text-align: center;"><td><a class="eventResourceURL" href="'+author+'" target="_blank">'+parseTitle[1]+'</a></td></tr>';
				}
			}
			
			break;
		}
	}
	
	$('#modalEventResources').modal('show');

}

/*
 * TODO: Remove this function and create the option elements of the select as follows:
 * <option value=""><a class="associatedEventClass" href="event link to open in new tab" target="_blank">Event Title</a></option>
 */
function showAssociatedEvent(selectId) {
	var eventsList = document.getElementById(selectId);
	var eventName = eventsList.options[eventsList.selectedIndex].value;
}
///////////////////////
function enableRegionEventDetection() {
	var country = document.getElementById('layerSpatialFilterEventDetectionValue1').options[document.getElementById('layerSpatialFilterEventDetectionValue1').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterEventDetectionValue2');

	resetSelectForm(divRef, 'Region');
	resetSelectForm(document.getElementById('layerSpatialFilterEventDetectionValue3'), 'Region Unit');
	resetSelectForm(document.getElementById('layerSpatialFilterEventDetectionValue4'), 'City');
		
	var res = alasql('SELECT DISTINCT region FROM geodata WHERE country = "' + country + '"');
	res.forEach(function(i) {
		if (i.region != '') {
			element = document.createElement('option');
			element.value = i.region;
			element.innerHTML = i.region;
			divRef.appendChild(element);
		}		
	});		  
	
	document.getElementById('layerSpatialFilterEventDetectionValue2').disabled = false;
	document.getElementById('layerSpatialFilterEventDetectionValue3').disabled = true;
	document.getElementById('layerSpatialFilterEventDetectionValue4').disabled = true;
}

function enableRegionUnitEventDetection() {
	var region = document.getElementById('layerSpatialFilterEventDetectionValue2').options[document.getElementById('layerSpatialFilterEventDetectionValue2').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterEventDetectionValue3');

	resetSelectForm(divRef, 'Region Unit');
	resetSelectForm(document.getElementById('layerSpatialFilterEventDetectionValue4'), 'City');
	
	var res = alasql('SELECT DISTINCT region_unit FROM geodata WHERE region = "' + region + '"');
	res.forEach(function(i) {
		if (i.region_unit != '') {
			element = document.createElement('option');
			element.value = i.region_unit;
			element.innerHTML = i.region_unit;
			divRef.appendChild(element);
		}			
	});		
	
	document.getElementById('layerSpatialFilterEventDetectionValue3').disabled = false;
	document.getElementById('layerSpatialFilterEventDetectionValue4').disabled = true;
}

function enableCityEventDetection() {
	var regionUnit = document.getElementById('layerSpatialFilterEventDetectionValue3').options[document.getElementById('layerSpatialFilterEventDetectionValue3').selectedIndex].value;
	var divRef = document.getElementById('layerSpatialFilterEventDetectionValue4');

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
	
	document.getElementById('layerSpatialFilterEventDetectionValue4').disabled = false;
}

function enableCountriesEventDetection() {
	var res = alasql('SELECT DISTINCT country FROM geodata');
    var divRef = document.getElementById('layerSpatialFilterEventDetectionValue1');   
    
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

function clearEventDetectionFilterForm() {
	$('#layerSpatialFilterEventDetectionValue1').get(0).selectedIndex = 0;
	$('#layerSpatialFilterEventDetectionValue2').get(0).selectedIndex = 0;
	$('#layerSpatialFilterEventDetectionValue3').get(0).selectedIndex = 0;
	$('#layerSpatialFilterEventDetectionValue4').get(0).selectedIndex = 0;
	document.getElementById('layerSpatialFilterEventDetectionValue2').disabled = true;
	document.getElementById('layerSpatialFilterEventDetectionValue3').disabled = true;
	document.getElementById('layerSpatialFilterEventDetectionValue4').disabled = true;
}
