import mysql.connector
import numpy as np

mydb = mysql.connector.connect(
    host="localhost",
    user="banhxeo",
    password="rEi2019Wa-05VtJ$p",
    database="v4",
    port=3308
)
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="sysadm",
#     password="ek7F7ck3",
#     database="v4",
#     port=3308
# )
mycursor = mydb.cursor()

mycursor.execute('SELECT * FROM datasources limit 10;')

result = mycursor.fetchone()
print(result)
