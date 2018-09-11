# coding:utf-8

import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb


SQLHOST = "172.29.3.56"
SQLUSER = "root"
SQLPASSWORD = "cdSB123"
# 数据库名
SQLDB = "HWCom"
SQLPORT = 3306

class MySQL():
    def __init__(self,sqldb="HWCom"):
        self.sqldb=sqldb


    def login(self):
        cursor=MySQLdb.cursors.SSCursor
        self.connect=MySQLdb.connect(host=SQLHOST,port=SQLPORT,user=SQLUSER,password=SQLPASSWORD,db=self.sqldb,charset="utf8",cursorclass=cursor)
        self._cursor=self.connect.cursor()

    def get_cursor(self):
        return self._cursor

    def close(self):
        self._cursor.close()
        self.connect.close()




