/**
 * Zoom to all layers
 */

var listener_ip = "http://127.0.0.1:5000/";

function zoomToAll(mode) {
    var first = true;
    var extent = null;
    map.getLayers().forEach(function(layer) {
        if (typeof(layer.get('title')) != 'undefined' && layer.get('title') != 'userInfo') {
            if (first) {
                if (getLayerType(layer.get('title')) != 'geotiff' && getLayerType(layer.get('title')) != 'wms') {
                    extent = layer.getSource().getSource().getExtent();
                    first = false;
                } else if (getLayerType(layer.get('title')) == 'geotiff') {
                    extent = getImageExtent(layer.get('title'));
                    first = false;
                } else {
                    //WMS
                    extent = getWMSExtent(layer.get('title'));
                    first = false;
                }

            } else {
                if (getLayerType(layer.get('title')) != 'geotiff' && getLayerType(layer.get('title')) != 'wms') {
                    ol.extent.extend(extent, layer.getSource().getSource().getExtent());
                } else if (getLayerType(layer.get('title')) == 'geotiff') {
                    ol.extent.extend(extent, getImageExtent(layer.get('title')));
                } else {
                    //WMS
                    ol.extent.extend(extent, getWMSExtent(layer.get('title')));
                }

            }
        }
    });

    if (mode == 1) {
        extent = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
        return extent;
    }

    if (extent != null) {
        map.getView().fit(extent, map.getSize());
    }
}

function getLayerType(name) {
    for (var i = 0; i < mapLayers.length; i++) {
        if (mapLayers[i].name == name) {
            if (mapLayers[i].type.substring(0, 3) != 'wms') {
                return mapLayers[i].type;
            } else {
                return 'wms';
            }
        }
    }
}

function getImageExtent(name) {
    for (var i = 0; i < mapLayers.length; i++) {
        if (mapLayers[i].name == name) {
            var parse = mapLayers[i].imageBbox.split(',');
            var extent = [Number(parse[2]), Number(parse[3]), Number(parse[4]), Number(parse[5])];
            return extent;
        }
    }
}

function getWMSExtent(name) {
    for (var i = 0; i < mapLayers.length; i++) {
        if (mapLayers[i].name == name) {
            return mapLayers[i].imageBbox;
        }
    }
}

/**
 * Show all layers
 */
function showAllLayers() {
    map.getLayers().forEach(function(layer) {
        if (typeof(layer.get('title')) != 'undefined') {
            if (layer.get('title') != 'overlayStyle') {
                layer.setVisible(true);
                document.getElementById("shBox" + layer.get('title')).checked = true;
            }
        }
    });
}

/**
 * Hide all layers
 */
function hideAllLayers() {
    map.getLayers().forEach(function(layer) {
        if (typeof(layer.get('title')) != 'undefined') {
            if (layer.get('title') != 'overlayStyle') {
                layer.setVisible(false);
                document.getElementById("shBox" + layer.get('title')).checked = false;
            }
        }
    });
}

function showColorPanel() {
    document.getElementById('colorPanel').style.display = 'block';
}

function closeColorPanel() {
    document.getElementById('colorPanel').style.display = 'none';
}

/**
 * Loading indicator
 */
function showSpinner(colorID) {
    colorID = '#F5F5F5';
    window.scrollTo(0, 0);
    var opts = {
        lines: 12, // The number of lines to draw
        length: 6, // The length of each line
        width: 3, // The line thickness
        radius: 8, // The radius of the inner circle
        rotate: 0, // The rotation offset
        color: colorID, // #rgb or #rrggbb
        speed: 1, // Rounds per second
        trail: 60, // Afterglow percentage
        shadow: false, // Whether to render a shadow
        hwaccel: false, // Whether to use hardware acceleration
        className: 'spinner', // The CSS class to assign to the spinner
        zIndex: 5000, // The z-index (defaults to 2000000000)
        top: 0, // Top position relative to parent in px
        left: 0 // Left position relative to parent in px
    };
    var spinner = new Spinner(opts).spin();
    $("#loadingWheel").append(spinner.el);
}

function hideSpinner() {
    $('.spinner').hide();
}

function showSpinnerDescribe(colorID) {
    //colorID = '#F5F5F5';
    window.scrollTo(0, 0);
    var opts = {
        lines: 12, // The number of lines to draw
        length: 6, // The length of each line
        width: 3, // The line thickness
        radius: 8, // The radius of the inner circle
        rotate: 0, // The rotation offset
        color: colorID, // #rgb or #rrggbb
        speed: 1, // Rounds per second
        trail: 60, // Afterglow percentage
        shadow: false, // Whether to render a shadow
        hwaccel: false, // Whether to use hardware acceleration
        className: 'spinner', // The CSS class to assign to the spinner
        zIndex: 5000, // The z-index (defaults to 2000000000)
        top: 0, // Top position relative to parent in px
        left: 0 // Left position relative to parent in px
    };
    var spinner = new Spinner(opts).spin();
    $("#loadingWheelDescribe").append(spinner.el);
}

function hideSpinnerDescribe() {
    $('.spinner').hide();
}

function showSpinnerChangeset(colorID) {
    colorID = '#26a69a';
    window.scrollTo(0, 0);
    var opts = {
        lines: 12, // The number of lines to draw
        length: 6, // The length of each line
        width: 3, // The line thickness
        radius: 8, // The radius of the inner circle
        rotate: 0, // The rotation offset
        color: colorID, // #rgb or #rrggbb
        speed: 1, // Rounds per second
        trail: 60, // Afterglow percentage
        shadow: false, // Whether to render a shadow
        hwaccel: false, // Whether to use hardware acceleration
        className: 'spinnerChangeset', // The CSS class to assign to the spinner
        zIndex: 1030, // The z-index (defaults to 2000000000)
        top: 0, // Top position relative to parent in px
        left: 0 // Left position relative to parent in px
    };
    var spinner = new Spinner(opts).spin();
    $("#loadingWheelChangeset").append(spinner.el);
}

function hideSpinnerChangeset() {
    $('.spinnerChangeset').hide();
}

/**
 * Set new server URL
 */
function setNewServer() {
    var newServer = document.getElementById('serverURL').value;
    rootURL = newServer + '/rest/service';

    var newHostArr = newServer.replace("http://", "").split("/");
    var parseNewRootURL = newHostArr[0].split(':');
    server = parseNewRootURL[0];

    element = document.createElement('option');
    element.id = newServer;
    element.value = newServer;
    element.text = newServer;
    document.getElementById('selectServerURL').appendChild(element);

    document.getElementById('serverURL').value = null;

    var selectList = document.getElementById('selectServerURL');
    for (var opt, i = 0; opt = selectList.options[i]; i++) {
        if (opt.value == newServer) {
            selectList.selectedIndex = i;
            break;
        }
    }
}

/**
 * Set server URL the selected from list
 */
function setSelectedServer() {
    var element = document.getElementById('selectServerURL');
    var newServer = element.options[element.selectedIndex].value;
    rootURL = newServer + '/rest/service';

    var newHostArr = newServer.replace("http://", "").split("/");
    var parseNewRootURL = newHostArr[0].split(':');
    server = parseNewRootURL[0];
}

/**
 * stRDF representation to WKT. Take as input the map extent in EPSG:4326
 * @param bbox
 */
function mapExtentToWKT(bbox) {
    var polygon = createPolygon(bbox);
    var extent = polygon + ';http://www.opengis.net/def/crs/EPSG/0/4326';
    return extent;
}

/**
 * GeoSPARQL representation to WKT. Take as input the map extent in EPSG:4326
 * @param bbox
 */
function mapExtentToWKTLiteral(bbox) {
    var polygon = createPolygon(bbox);
    return polygon;
}

function mapExtentToWKTLiteralWGS84(bbox) {
    var left = Number(bbox[0]);
    var bottom = Number(bbox[1]);
    var right = Number(bbox[2]);
    var top = Number(bbox[3]);

    var polygon = 'POLYGON((' + right + ' ' + bottom + ', ' +
        left + ' ' + bottom + ', ' +
        left + ' ' + top + ', ' +
        right + ' ' + top + ', ' +
        right + ' ' + bottom + '))';

    return polygon;
}

/**
 * Create a polygon string from the OpenLayers.Bounds object
 * @param bbox
 */
function createPolygon(bbox) {
    var left = Number(bbox[0]);
    var bottom = Number(bbox[1]);
    var right = Number(bbox[2]);
    var top = Number(bbox[3]);

    var polygon = 'POLYGON((' + bottom + ' ' + right + ', ' +
        bottom + ' ' + left + ', ' +
        top + ' ' + left + ', ' +
        top + ' ' + right + ', ' +
        bottom + ' ' + right + '))';

    return polygon;
}

var animateLegend = 0;
var animateTimeline = 0;
var animateSwefs = 0;
var animateStats = 0;

function animateLegendPanel() {
    if (animateLegend == 0) {
        document.getElementById('animatePanelButton').innerHTML = '<i class="fa fa-chevron-left fa-lg"></i>';
        document.getElementById('animatePanelButton').title = 'Show Panel';
        $("#legendPanel").animate({
            right: -($("#legendPanel").width())
        }, 500);
        animateLegend = 1;
    } else {
        /*
        if (animateStats == 1) {
        	animateStatsPanel();
        }*/
        document.getElementById('animatePanelButton').innerHTML = '<i class="fa fa-chevron-right fa-lg"></i>';
        document.getElementById('animatePanelButton').title = 'Hide Panel';
        $("#legendPanel").animate({
            right: 0
        }, 500);
        animateLegend = 0;
    }
}

function animateTimePanel() {
    if (animateTimeline == 0) {
        if (animateStats == 1) {
            animateStatsPanel();
        }
        document.getElementById('animateTimelineButton').innerHTML = '<i class="fa fa-chevron-right fa-lg"></i>';
        document.getElementById('animateTimelineButton').title = 'Hide Timeline';
        $("#tmContainer").animate({
            right: 0
        }, 500);
        animateTimeline = 1;
        hideWMStimePanel();
    } else {
        document.getElementById('animateTimelineButton').innerHTML = '<i class="fa fa-chevron-left fa-lg"></i>';
        document.getElementById('animateTimelineButton').title = 'Show Timeline';
        $("#tmContainer").animate({
            right: -($("#tmContainer").width())
        }, 500);
        animateTimeline = 0;
    }
}

function animateStatsPanel() {
    if (animateStats == 0) {
        if (animateLegend == 0) {
            animateLegendPanel();
        }
        if (animateTimeline == 1) {
            animateTimePanel();
        }
        document.getElementById('animateStatsPanelButton').innerHTML = '<i class="fa fa-chevron-right fa-lg"></i>';
        document.getElementById('animateStatsPanelButton').title = 'Hide Stats';
        $("#statsContainer").animate({
            right: 0
        }, 500);
        animateStats = 1;
        hideWMStimePanel();
    } else {
        document.getElementById('animateStatsPanelButton').innerHTML = '<i class="fa fa-chevron-left fa-lg"></i>';
        document.getElementById('animateStatsPanelButton').title = 'Show Stats';
        $("#statsContainer").animate({
            right: -($("#statsContainer").width()) - 500
        }, 500);
        animateStats = 0;
    }
}

function animateSwefsPanel() {
    if (animateSwefs == 0) {
        document.getElementById('animateSwefsPanelButton').innerHTML = '<i class="fa fa-chevron-right fa-lg"></i>';
        document.getElementById('animateSwefsPanelButton').title = 'Hide SWeFS Panel';
        var width = $("#map_canvas").width() * 0.8 - 20;
        $("#swefsPanel").animate({
            right: width
        }, 500);
        animateSwefs = 1;

    } else {
        document.getElementById('animateSwefsPanelButton').innerHTML = '<i class="fa fa-chevron-left fa-lg"></i>';
        document.getElementById('animateSwefsPanelButton').title = 'Show SWeFS Panel';
        $("#swefsPanel").animate({
            right: -($("#swefsPanel").width() + 200)
        }, 500);
        animateSwefs = 0;
    }
}

var layerSet = false;
var baseMapType = null;

function setBaseBingAerial() {
    var vecLayer = undefined;
    mapFilter.getLayers().forEach(function(layer) {
        vecLayer = layer;
    });
    if (bingMapsKey != null) {
        document.getElementById('coordinates').style.color = '#FFCC66';
        mapFilter.getLayers().setAt(1, bingMap);
        mapFilter.addLayer(vecLayer);
        baseMapType = 'bing';

        if (!layerSet) {
            mapFilter.addLayer(featureOverlay);
            featureOverlay.setZIndex(5);
            layerSet = true;
        }
    } else {
        document.getElementById('alertMsgBingKey').style.display = 'block';
        setTimeout(function() {
            $('#alertMsgBingKey').fadeOut('slow');
        }, 10000);
    }
}

function setBaseBingAerialLabels() {
    var vecLayer = undefined;
    mapFilter.getLayers().forEach(function(layer) {
        vecLayer = layer;
    });
    if (bingMapsKey != null) {
        document.getElementById('coordinates').style.color = '#FFCC66';
        mapFilter.getLayers().setAt(1, bingAerialLabels);
        mapFilter.addLayer(vecLayer);
        baseMapType = 'aerial';

        if (!layerSet) {
            mapFilter.addLayer(featureOverlay);
            featureOverlay.setZIndex(5);
            layerSet = true;
        }
    } else {
        document.getElementById('alertMsgBingKey').style.display = 'block';
        setTimeout(function() {
            $('#alertMsgBingKey').fadeOut('slow');
        }, 10000);
    }
}

function setBaseBingRoad() {
    var vecLayer = undefined;
    mapFilter.getLayers().forEach(function(layer) {
        vecLayer = layer;
    });
    if (bingMapsKey != null) {
        document.getElementById('coordinates').style.color = '#A30052';
        mapFilter.getLayers().setAt(1, bingRoads);
        mapFilter.addLayer(vecLayer);
        baseMapType = 'road';

        if (!layerSet) {
            mapFilter.addLayer(featureOverlay);
            featureOverlay.setZIndex(5);
            layerSet = true;
        }
    } else {
        document.getElementById('alertMsgBingKey').style.display = 'block';
        setTimeout(function() {
            $('#alertMsgBingKey').fadeOut('slow');
        }, 10000);
    }
}

function setBaseOSM() {
    var vecLayer = undefined;
    mapFilter.getLayers().forEach(function(layer) {
        vecLayer = layer;
    });
    document.getElementById('coordinates').style.color = '#A30052';
    mapFilter.getLayers().setAt(1, baseOSM);
    mapFilter.addLayer(vecLayer);
    baseMapType = 'OSM';

    if (!layerSet) {
        mapFilter.addLayer(featureOverlay);
        featureOverlay.setZIndex(5);
        layerSet = true;
    }
}

function baseGhyb() {
    /*
    map.setBaseLayer(ghyb);
    document.getElementById('coordinates').style.color = '#FFCC66';
    */
}

function disableFeatures(disableAll, disableSaveMap) {
    if (disableAll == true) {
        //disable functions: okBTNall class
        var okBTNs = document.getElementsByClassName('okBTNall');
        for (var i = 0; i < okBTNs.length; i++) {
            okBTNs[i].disabled = true;
        }
    }

    if (disableSaveMap == true) {
        //disable save map function: okBTNsaveMap class
        var okBTNsave = document.getElementsByClassName('okBTNsaveMap');
        for (var i = 0; i < okBTNsave.length; i++) {
            okBTNsave[i].disabled = true;
        }
    }
}

function showManualPages() {
    document.getElementById('manualPages').style.display = 'block';
}

function closeManualPages() {
    document.getElementById('manualPages').style.display = 'none';
}

function showHelpPage(pageId) {
    document.getElementById('manualFrame').src = "./assets/manual/" + pageId;
    showManualPages();
}


function isPollChecked() {
    var rlist = document.getElementById('poll');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return true;
        }
    }
    return false;
}

function pollcheckedVal() {
    var rlist = document.getElementById('poll');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return rlist[i].value;
        }
    }
}


function isMethodChecked() {
    var rlist = document.getElementById('clust');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return true;
        }
    }
    return false;
}

function methodcheckedVal() {
    var rlist = document.getElementById('clust');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return rlist[i].value;
        }
    }
}



function isMetricChecked() {
    var rlist = document.getElementById('est_met');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return true;
        }
    }
    return false;
}

function compVal() {
    var rlist = document.getElementById('compare');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return rlist[i].value;
        }
    }
}


function iscompChecked() {
    var rlist = document.getElementById('compare');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return true;
        }
    }
    return false;
}

function metriccheckedVal() {
    var rlist = document.getElementById('compare');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            return rlist[i].value;
        }
    }
}

function getBans() {
    var ret = [];
    var rlist = document.getElementById('stat_ban');
    for (var i = 0; i < rlist.length; i++) {
        if (rlist[i].checked) {
            ret.push(rlist[i].value);
        }
    }
    return ret;
}

function isBanned(bans, station){
   for (var i=0;i<bans.length;i++)
   {
      if (bans[i] == station)
      {
        return true;
      }
   }
   return false;
}


var geo = undefined;
var resp = undefined;

function estimateLocation() {
    var date = $('#datepicker').datepicker().val();
    var hourdiv = document.getElementById("hourpicker");
    var hour = hourdiv.options[hourdiv.selectedIndex].value;
    var timestamp = date + " " + hour + ":00:00";
    clearDispersion();
    vector.getSource().forEachFeature(function(feature) {
        var s = document.getElementById('stat_info');
        for (i = 0; i < s.childNodes.length; i++) {
            if (s.childNodes[i].id == feature.getId()) {
                vector.getSource().removeFeature(feature);
            }
        }
    });
    drawStations();
    var res = document.getElementById('source_result');
    res.innerHTML = '';
    if (isPollChecked() && isMethodChecked() && !!date && iscompChecked()) {
        var locs = [];
        vector.getSource().forEachFeature(function(feature) {
            try {
                var id = feature.getId();
                if (id.includes('detection')) {
                    var coord = ol.proj.transform(feature.getGeometry().getCoordinates(), 'EPSG:3857', 'EPSG:4326');
                    locs.push({
                        lat: String(coord[1]),
                        lon: String(coord[0])
                    });
                }
            } catch (e) { //do nothing
            }
        });
        if (locs.length > 0) {
            var loader = document.getElementById('loader_ic');
            var eheader = document.getElementById('estimate');
            var slider = document.getElementById('div_slider');
            var thres = document.getElementById('p_thres');
            var uri = document.getElementById('city_uri');
            loader.style.display = 'block';
            eheader.style.display = 'none';
            slider.style.display = 'none';
            thres.style.display = 'none';
            uri.style.display = 'none';
            if (methodcheckedVal().indexOf('classification') == -1) {
                var req = new XMLHttpRequest();
                req.open("POST", listener_ip + "detections/" + timestamp + "/" + pollcheckedVal() + "/cosine/" + methodcheckedVal() + '/' + compVal(), true);
                req.setRequestHeader('Content-Type', 'application/json; charset=utf-8');
                req.send(JSON.stringify(locs));
                req.onloadend = function() {
                    resp = JSON.parse(req.responseText);
                    bans = getBans();
                    if (resp["scores"][0] - resp["scores"][2] != 0) {
                        res_str = 'Estimated sources: <br> <table style="border-collapse: collapse;"><tr><th style="padding: 8px;">Station<br>name</th><th style="padding: 8px;">Score</th><th style="padding: 8px;">Draw</th></tr>';
                        for (var i = 0; i < resp['scores'].length; i++) {
                            if (resp['scores'][i] != 0 && !isBanned(bans,resp['stations'][i])) {
                                res_str += '<tr><td style="padding: 8px;">'+resp['stations'][i] + '</td><td style="padding: 8px;">' + resp['scores'][i] + '</td><td style="padding: 8px;"><form id="ui_form_'+i+'"><button type="button" class="btn btn-primary" onclick="drawDispersion('+i+')">Plume</button><button type="button" class="btn btn-primary" onclick="checkPop('+i+')">Affected areas</button></form><div id="loader_ic_'+i+'" class="loader" style="display:none;"></div></td></tr>';
                                }
                        }
                        res_str += '</table>';
                        res.innerHTML = res_str;
                        resp.affected = [{},{},{}];
                        loader.style.display = 'none';
                        eheader.style.display = 'block';
                    } else {
                        alert('Either detection points are out of grid or there is no overlap between detection points and calculated dispersions');
                        loader.style.display = 'none';
                        eheader.style.display = 'block';
                    }
                };
            } else {
                var req = new XMLHttpRequest();
                req.open("POST", listener_ip+"class_detections/" + timestamp + "/" + pollcheckedVal() + "/cosine/" + methodcheckedVal(), true);
                req.setRequestHeader('Content-Type', 'application/json; charset=utf-8');
                req.send(JSON.stringify(locs));
                req.onloadend = function() {
                    resp = JSON.parse(req.responseText);
                    checkClassProgress(resp['id']);
                };
            }
        } else {
            alert('You should mark some detection points before estimating the source\'s location');
        }
    } else {
        alert('You should choose a date, pollutant & clustering method before estimating the source\'s location');
    }
}

function checkClassProgress(id){
    var res = document.getElementById('source_result');
    var loader = document.getElementById('loader_ic');
    var eheader = document.getElementById('estimate');
    var req = new XMLHttpRequest();
    req.open("GET", listener_ip+"status/" + id, true);
    req.setRequestHeader('Content-Type', 'application/json; charset=utf-8');
    req.send();
    req.onloadend = function() {
        resp = JSON.parse(req.responseText);
        if (resp['state'] != 'PENDING' && resp['state'] != 'PROGRESS') {
             if ('result' in resp) {
                 resp = JSON.parse(resp['result']);
                 if (resp["scores"][0] - resp["scores"][2] != 0) {
                     res_str = 'Estimated sources: <br> <table style="border-collapse: collapse;"><tr><th style="padding: 8px;">Station<br>name</th><th style="padding: 8px;">Score</th><th style="padding: 8px;">Draw</th></tr>';
                     for (var i = 0; i < resp['scores'].length; i++) {
                         if (resp['scores'][i] != 0) {
                             res_str += '<tr><td style="padding: 8px;">'+resp['stations'][i] + '</td><td style="padding: 8px;">' + resp['scores'][i] + '</td><td style="padding: 8px;"><form id="ui_form_'+i+'"><button type="button" class="btn btn-primary" onclick="drawDispersion('+i+')">Plume</button><button type="button" class="btn btn-primary" onclick="checkPop('+i+')">Affected areas</button></form><div id="loader_ic_'+i+'" class="loader" style="display:none;"></div></td></tr>';
                             }
                     }
                     res_str += '</table>';
                     res.innerHTML = res_str;
                     resp.affected = [{},{},{}];
                     loader.style.display = 'none';
                     eheader.style.display = 'block';
                 } else {
                     alert('Either detection points are out of grid or there is no overlap between detection points and calculated dispersions');
                     loader.style.display = 'none';
                     eheader.style.display = 'block';
                 }
             }
        }
        else{
          setTimeout(function() {
                     checkClassProgress(id);
                 }, 2000);
        }
    };
}


function initPop(idx){
  affected = resp['affected'][idx];
  var max;
  for (var i = 0; i < affected['features'].length; i++) {
      if (!max || parseInt(affected['features'][i]['properties']['POP']) > max) {
          max = parseInt(affected['features'][i]['properties']['POP']);
      }
  }
  var min;
  for (var i = 0; i < affected['features'].length; i++) {
      if (!min || parseInt(affected['features'][i]['properties']['POP']) < min) {
          min = parseInt(affected['features'][i]['properties']['POP']);
      }
  }
  drawDispersion(idx);
  var slider = document.getElementById('p_slider');
  slider.min = min;
  slider.max = max;
  $('input[type="range"]').rangeslider('update', true);
}

function getPopulation(idx){
  var slider = document.getElementById('div_slider');
  var thres = document.getElementById('p_thres');
  var click = document.getElementById('ui_form_'+idx);
  var load = document.getElementById('loader_ic_'+idx);
  load.style.display = 'block';
  click.style.display = 'none';
  $.ajax({
      type: 'POST',
      url: listener_ip + "population/",
      data: JSON.stringify(resp.dispersions[idx]),
      success: function(result) {
        var task = JSON.parse(result);
        checkTaskProgress(task['id'],idx);
      },
      async: true
    });
  // $.ajax({
  //     type: 'POST',
  //     url: listener_ip + "population/",
  //     data: JSON.stringify(resp.dispersions[idx]),
  //     success: function(result) {
          // var pop_result = JSON.parse(result);
          // resp.affected[idx] = pop_result;
          // slider.style.display = 'block';
          // thres.style.display = 'block'
          // load.style.display = 'none';
          // click.style.display = 'block';
          // initPop(idx);
  //     },
  //     async: false
  // });
}

function checkTaskProgress(id,idx){
  var req = new XMLHttpRequest();
    req.open("GET", listener_ip+"status/" + id, true);
    req.setRequestHeader('Content-Type', 'application/json; charset=utf-8');
    req.send();
    req.onloadend = function() {
      var task = JSON.parse(req.responseText);
      if (task['state'] != 'PENDING' && task['state'] != 'PROGRESS') {
            var slider = document.getElementById('div_slider');
            var thres = document.getElementById('p_thres');
            var click = document.getElementById('ui_form_'+idx);
            var load = document.getElementById('loader_ic_'+idx);
            var pop_result = JSON.parse(task['result']);
            resp.affected[idx] = pop_result;
            slider.style.display = 'block';
            thres.style.display = 'block';
            load.style.display = 'none';
            click.style.display = 'block';
            initPop(idx);
      }
      else{
          setTimeout(function() {
                     checkTaskProgress(id,idx);
                 }, 2000);
      }
  };
}


function checkPop(idx){
    if (JSON.stringify(resp.affected[idx]) === JSON.stringify({})) {
      getPopulation(idx);
    }
    else{
      initPop(idx);
    }
}


function drawDispersion(idx) {
    var styling = null;
    var label = 'dispersion_' + idx;
    clearDispersion();
    vector.getSource().forEachFeature(function(feature) {
        var s = document.getElementById('stat_info');
        for (i = 0; i < s.childNodes.length; i++) {
            if (s.childNodes[i].id == feature.getId()) {
                var style = new ol.style.Style({
                    image: new ol.style.Icon({
                        src: './assets/images/map-pin-md.png',
                        size: [186, 297],
                        scale: 0.1
                    })
                });
                feature.setStyle(style);
            }
        }
    });
    vector.getSource().forEachFeature(function(feature) {
        var fid = feature.getId();
        if (fid == resp["stations"][idx]) {
            var style = new ol.style.Style({
                image: new ol.style.Icon({
                    src: './assets/images/pin_red.png',
                    size: [433, 692],
                    scale: 0.05
                })
            });
            feature.setStyle(style);
        }
    });
    var s = document.getElementById('stat_info');
    for (i = 0; i < s.childNodes.length; i++) {
        s.childNodes[i].style.display = 'none';
    }
    var div = document.getElementById(resp["stations"][idx]);
    div.style.display = 'block';
    geo = JSON.parse(resp['dispersions'][idx]);
    var features = new ol.format.GeoJSON().readFeatures(geo, {
        featureProjection: 'EPSG:3857'
    });
    var source = new ol.source.Vector({
        features: features
    });

    var layer = new ol.layer.Image({
        title: label,
        source: new ol.source.ImageVector({
            source: source,
            style: defaultVectorStyle
        })
    });

    mapFilter.addLayer(layer);

}


function filterPop(idx, thres) {
    var grid = resp.affected[idx];
    var filtered = grid.features.filter(function(prop) {
        return prop['properties']['POP'] >= thres;
    });
    var toReturn = {};
    toReturn.crs = grid.crs;
    toReturn.type = "FeatureCollection";
    toReturn.features = filtered;
    return toReturn;
}

function drawPopGrid(idx, thres) {
    clearPopGrid();
    var slider = document.getElementById('p_slider');
    var geojsonObject = filterPop(idx, thres);
    for (var i = 0 ; i<geojsonObject.features.length;i++){
        lnglt = [geojsonObject.features[i].geometry.coordinates[0],geojsonObject.features[i].geometry.coordinates[1]];
        var feat = new ol.Feature(new ol.geom.Point(ol.proj.transform(lnglt, 'EPSG:4326', 'EPSG:3857')));
        feat.setId('POP_'+i);
        var scale = Math.round(geojsonObject.features[i].properties['POP']/slider.max)*0.02+0.02;
        var style = new ol.style.Style({
                  image: new ol.style.Icon({
                      src: 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Ski_trail_rating_symbol-black_diamond.svg/1024px-Ski_trail_rating_symbol-black_diamond.svg.png',
                      size: [1024, 1024],
                      scale: scale
                  })
              });
        feat.set('uri',geojsonObject.features[i].properties['URI']);
        feat.set('name',geojsonObject.features[i].properties['NAME']);
        feat.setStyle(style);
        var vec = vector.getSource();
        vec.addFeature(feat);
      }
}


function drawWindDir() {
    var date = $('#datepicker').datepicker().val();
    var hourdiv = document.getElementById("hourpicker");
    var hour = hourdiv.options[hourdiv.selectedIndex].value;
    var pldiv = document.getElementById("pressure_level");
    var level = pldiv.options[pldiv.selectedIndex].value;
    var timestamp = date + " " + hour + ":00:00";
    clearWindDir();
    var req = new XMLHttpRequest();
    req.open("GET", listener_ip + "getClosestWeather/" + timestamp + "/" + level, true);
    req.setRequestHeader('Content-Type', 'plain/text; charset=utf-8');
    req.onreadystatechange = function() {
        if (req.readyState == XMLHttpRequest.DONE) {
            geobj = JSON.parse(req.responseText);
            if (geobj.hasOwnProperty('error')) {
                alert('date is out of bounds');
            } else {
                var styling = new ol.style.Style({
                    stroke: new ol.style.Stroke({
                        color: [136, 136, 136, 1],
                        width: 1
                    })
                });
                var label = 'wind_direction';
                var geojsonObject = {
                    'type': 'FeatureCollection',
                    'crs': {
                        'type': 'name',
                        'properties': {
                            'name': 'EPSG:4326'
                        }
                    },
                    'features': [geobj]
                };
                var features = new ol.format.GeoJSON().readFeatures(geojsonObject, {
                    featureProjection: 'EPSG:3857'
                });
                var source = new ol.source.Vector({
                    features: features
                });

                var layer = new ol.layer.Image({
                    title: label,
                    source: new ol.source.ImageVector({
                        source: source,
                        style: styling
                    })
                });
                mapFilter.addLayer(layer);
                var listenerKey = layer.getSource().on('change', function(e) {
                    if (layer.getSource().getState() == 'ready') {
                        updateLayerStats(label);

                        for (var i = 0; i < mapLayers.length; i++) {
                            if ((mapLayers[i].name === label) && (label != 'userInfo')) {
                                mapLayers[i].features = getLayerFeatureNames(layer);
                                break;
                            }
                        }

                        // map.getView().fit(layer.getSource().getSource().getExtent(), map.getSize());

                        //Unregister the "change" listener
                        layer.getSource().unByKey(listenerKey);
                    }
                });
            }
        }
    }
    req.send();
}
