#!/bin/bash
git clone http://github.com/iaklampanos/bde-pilot-2.git
rm -rf /tomcat/webapps/Sextant_v2.0/assets/
rm -rf /tomcat/webapps/Sextant_v2.0/index.html
# cp -rp /bde-pilot-2/Sextant_v2.0/assets/ /tomcat/webapps/Sextant_v2.0/
# cp /bde-pilot-2/Sextant_v2.0/index.html /tomcat/webapps/Sextant_v2.0/
# cp -rp /bde-pilot-2/Sextant_v2.0/data/ /tomcat/webapps/Sextant_v2.0/
# pip install -r /bde-pilot-2/listener_reqs.txt
cp -rp /bde-pilot-2/pilot/Sextant_v2.0/assets/ /tomcat/webapps/Sextant_v2.0/
cp /bde-pilot-2/pilot/Sextant_v2.0/index.html /tomcat/webapps/Sextant_v2.0/
cp -rp /bde-pilot-2/pilot/Sextant_v2.0/data/ /tomcat/webapps/Sextant_v2.0/
pip install -r /bde-pilot-2/pilot/requirements.txt
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -U theano
