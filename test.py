import numpy as np
import ai as nn
import matplotlib.pyplot as plt
import random
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

parameter = nn.neuron_network(dt.T, y.reshape((1, y.shape[0])), n_iter=100)

# print(parameter)

print(nn.predict(np.array(new_hand).T, parameter).any())

