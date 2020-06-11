# -*-coding: utf-8 -*-

class Mysql():
    HOST = "192.168.31.14"
    HOST = "localhost"
    PORT = 3306
    DB_NAME = "ncov"
    USER_NAME = "root"
    PASSWORD = "root"
    TB_NAME = "patients"
    POOL_SIZE = 3

    
class Web():        
    PORT = 9400
    
    
class Config():
    mysql = Mysql
    web = Web
    

