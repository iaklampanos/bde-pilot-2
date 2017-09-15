"""
   CLASS INFO
   ---------------------------------------------------------------------------
   Handling POSTGRESQL connection and queries
   ---------------------------------------------------------------------------
"""
from sqlalchemy import create_engine
import json
import base64
from sqlalchemy.orm import sessionmaker

class DBConn(object):

    instance = None
    engine = None

    def __init__(self):
        with open('db_info.json', 'r') as data_file:
            dbpar = json.load(data_file)
        engine = create_engine('postgresql+psycopg2://' + dbpar['user'] + ':' +
                 base64.b64decode(dbpar['pass']) + '@' + dbpar['host'] + '/'
                 + dbpar['dbname'] + '?sslmode=disable')
        engine.connect()
        self.engine = engine
        instance = engine

    def __new__(cls):
        if DBConn.instance is None:
            DBConn.instance = object.__new__(cls)
        return DBConn.instance

    def safequery(self, sql):
        maker = sessionmaker(bind=self.engine, autocommit=True)
        session = maker()
        return session.execute(sql)
