#!/bin/bash
cp /pilot-sc5-cycle3/data_ingest/db_info.json .
cd /pilot-sc5-cycle3/backend/
rm -f *.zip
killall -9 redis-server
killall -9 celery
killall -9 gunicorn
./run-redis.sh &
celery worker -A listener.celery -f celery_log.txt --loglevel=info &
gunicorn -b 0.0.0.0:5000 -w 2 -t 500 listener:app --log-level debug --log-file gunicorn_log.txt &
