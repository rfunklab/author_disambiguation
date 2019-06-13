import mysql.connector
import db_config

mydb = mysql.connector.connect(
    host="rfunksql.csom.umn.edu",
    user=db_config.DB_CONFIG['user'],
    passwd=db_config.DB_CONFIG['passwd'])

mycursor = mydb.cursor()
mycursor.execute("USE googlepatents")
mycursor.execute("SELECT * FROM uspatents_raw_20190228 LIMIT 25")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)
