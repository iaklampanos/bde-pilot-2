#!/bin/bash
# Script that installs and runs redis
# from this celery example
# https://blog.miguelgrinberg.com/post/using-celery-with-flask
if [ ! -d redis-stable/src ]; then
    wget http://download.redis.io/redis-stable.tar.gz
    tar xvzf redis-stable.tar.gz
    rm redis-stable.tar.gz
fi
cd redis-stable
make
src/redis-server
