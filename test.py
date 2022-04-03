import numpy as np
import ai as nn
import neuron as sn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
import random
import json
from json import JSONEncoder
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="aiyugioh",
  password="tsumenohy12",
  database="aiyugioh"
)


mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM cards")

myresult = mycursor.fetchall()

valI = []
dataTrain = []

for x in myresult:
    valI.append(1 if 'Monster' in x[2] else 0)
    dataTrain.append( np.array([len(x[1]),len(x[2])]))

dt = np.array(dataTrain)
y = np.array(valI)

mycursor.execute("SELECT * FROM cards where id in (%s,%s,%s,%s,%s)",[random.randint(1, 11601),random.randint(1, 11601),random.randint(1, 11601),random.randint(1, 11601),random.randint(1, 11601) ])

myHand = mycursor.fetchall()

new_hand = []
toShow = []

for a in myHand:
    new_hand.append(np.array([len(a[1]), len(a[2])]))
    toShow.append(np.array([a[1], a[2]]))

parameter = nn.neuron(dt.T, y.reshape((1, y.shape[0])), n1=32, n_iter=1000)

print(parameter)

print(nn.predict(np.array(new_hand).T, parameter).any())
####
# Update in database
###
# val = []

# sql = "INSERT INTO cards (name, type, description, race, archetype, atk, def, level, rank, linkval, linkmarkers, attributes_card) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

# with open("db.json") as json_file:
    # f = json.load(json_file)['data']

# for fa in f:
    # val.append((fa['name'], fa['type'], fa['desc'], fa['race'], fa['archetype'] if 'archetype' in fa else 0, fa['atk'] if 'atk' in fa else 0, fa['def'] if 'def' in fa else 0, fa['level']  if 'level' in fa else 0, fa['rank'] if 'rank' in fa else 0, fa['linkval'] if 'linkval' in fa else 0, ', '.join(fa['linkmarkers']) if 'linkmarkers' in fa else 0, fa['attributes'] if 'attributes' in fa else 0))
    
# mycursor.executemany(sql, val)
# mydb.commit()
# print(mycursor.rowcount, "was inserted.")
