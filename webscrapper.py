# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:31:23 2019

@author: ASUS
"""
import pandas as pd
import urllib3
import numpy as np
import json

mid=[]
time=[]
user=[]
cont=[]
sent=[]

http = urllib3.PoolManager()

for j in range(10):
   
    if len(mid)==0:
        url = "https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json"
    else:
        url = "https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json?max={}".format(min_ID)
        min_ID=min(mid)
    r = http.request('GET', url)
    data = json.loads(r.data)
#    if data["response"]["status"] == 200:
#        print("My request was accepted")
for element in data["messages"]:
    mid.append(element["id"])
    time.append(element["created_at"])
    user.append(element["user"]["username"])
    cont.append(element["body"])
    try:
        sent.append(element["entities"]["sentiment"]["basic"])
    except:
        sent.append(None)

df=pd.DataFrame(time,columns=['TimeStamp'])
df['ID']=mid
df['User']=user
df['Content']=cont
df['Sentiment']=sent


#%% Store data on MongoDB

import pymongo
from pymongo import MongoClient

# creat database 'Group_BTC' and collection 'messages'
client = MongoClient('localhost',27017)
db = client["Group_BTC"]
collection = db.messages_BTC


#create a loop to get 30 twits time by time

my_id=[]

for x in range(0,30):
    if len(my_id) == 0:
        url = "https://api.stocktwits.com/api/2/streams/symbol/MSFT.json"
    else:
        url = "https://api.stocktwits.com/api/2/streams/symbol/MSFT.json?max=" + str(min(my_id))
    print(url)
    data = json.loads(http.request('GET', url).data)

db.messages.drop()

symbol_list = []

for element in data['messages']:
    try:
        group_data = {'id':element['id'], 'created_at':element['created_at'], 'users' :element['user']['username'], 'content' :element['body'], 'symbol':element['symbols'], 'sentiment':(element["entities"]["sentiment"]['basic']=="Bullish")*2-1}
    except:
        group_data = {'id':element['id'], 'created_at':element['created_at'], 'users' :element['user']['username'], 'content' :element['body'], 'symbol':element['symbols'], 'sentiment':0}
    result = db.messages.insert_one(group_data)
    
    for s in element['symbols']:
        symbol_list.append(s['symbol'])

print(symbol_list)
symbol_list.groupby()



db.messages
    
a=db.messages
print(a)
print (a['sentiment'])

#%%
import re

! pip install -U textblob
from textblob import Textblob
