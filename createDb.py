import mysql.connector

try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="aiyugioh",
        password="tsumenohy12",
        database="aiyugioh"
    )
except Exception as ex:
    quit("Unable to connect to database")

print('connexion ok')