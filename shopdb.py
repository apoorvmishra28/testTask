import mysql.connector

mydb = mysql.connector.connect(
            user="root",
            passwd="root"
        )

class shopDB:
    def create_database(self):
        mycursor = mydb.cursor()
        mycursor.execute(
            "CREATE DATABASE IF NOT EXISTS shopdb"
        )

        return True

    def connect(self):
        mydb = mysql.connector.connect(
            user="root",
            passwd="root",
            database="shopdb"
        )
        return mydb

shopdb = shopDB()
shopdb.create_database()