# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:08:35 2019

@author: Team BTC - Hu Liang, Le Hoang Viet, Tang Hongxia, Vu Thanh Lan, El Habr Corine
"""
#sorry the code isnt very efficient. because of time constraints and the number of people working on the project, we couldnt do all the automatizations we would have liked to do.                               
#Code in block comment should not be run as it will make change to the cloud database

# %% Importing libraries
# You may need to install dnspython in order to work with cloud server

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
import os
import time
import re
import copy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from datetime import timedelta
from pymongo import MongoClient

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

#os.chdir('H:/Documents/Alternance/Project/')


# %% Function to scrap data from Stocktwit and add to the cloud server
# The function have 2 inputs:
# - Symbol of the asset in string
# - Rate limit: number of requests per execution, in integer

def get_stwits_data(symbol,rate_limit):
    
    client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
    db=client['SorbonneBigData']
    
    exist=0
    for q in db['{}'.format(symbol)].aggregate([ 
        { "$group": { 
            "_id": None,
            "min": { "$min": "$ID" } 
        }}
    ]):
        exist=1
        min_prev_id=q['min']

    
    http = urllib3.PoolManager()
    
    mid=[]
    duplicates=0
    
    for j in tqdm(range(rate_limit)):
        if exist==0:
            url = "https://api.stocktwits.com/api/2/streams/symbol/{}.json".format(symbol)
        elif exist!=0 and len(mid)==0:
            url = "https://api.stocktwits.com/api/2/streams/symbol/{}.json?max={}".format(symbol,min_prev_id)
        else:
            min_ID=min(mid)    
            url = "https://api.stocktwits.com/api/2/streams/symbol/{}.json?max={}".format(symbol,min_ID)
        
        r = http.request('GET', url)
        try:
            data = json.loads(r.data)
        except:
            print('Decode error, retry again')
            continue
        
        if duplicates==1:
            print('\nThere are duplicates in the result. Other people are maybe running. \nPlease try again later.')
            break
        
        if data["response"]["status"] != 200:
            print("\nYour request was denied, retry in 1 hour")
            time.sleep(3600)
            continue
#            insert_element=[]
#            break
            
        for element in data["messages"]:
            mid.append(element["id"])
            symbol_list=[]
            for s in element['symbols']:
                symbol_list.append(s['symbol'])
            try:
                insert_element = {"ID": element["id"], "TimeStamp": element["created_at"], "User": element["user"]["username"], "Content": element["body"],"Sentiment": (element["entities"]["sentiment"]["basic"]=="Bullish")*2-1,'Symbols':symbol_list}
            except:
                insert_element = {"ID": element["id"], "TimeStamp": element["created_at"], "User": element["user"]["username"], "Content": element["body"],"Sentiment": 0,'Symbols':symbol_list}
            try:
                result = db['{}'.format(symbol)].insert_one(insert_element)
            except:
                duplicates=1
                break
            
    return insert_element

# %% Execution of the function

symbol='BTC.X'
rate_limit=2000
last_ele=get_stwits_data(symbol,rate_limit)

# %% #Creating custom lexicon
    
#%% Finding the time interval of the database
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']

#Getting the minimum id
for q in db['BTC.X'].aggregate([ 
        { "$group": { 
            "_id": None,
            "min": { "$min": "$ID" } 
        }}
    ]):
    minID=q['min']

#Getting the timestamp from the min ID

for post in db['BTC.X'].find({'ID':minID}):
    start_time=post['TimeStamp']
#Getting the max id
for q in db['BTC.X'].aggregate([ 
        { "$group": { 
            "_id": None,
            "max": { "$max": "$ID" } 
        }}
    ]):
    maxID=q['max']

#Getting the timestamp from the max ID

for post in db['BTC.X'].find({'ID':maxID}):
    end_time=post['TimeStamp']

start_time=dt.strptime(start_time,'%Y-%m-%dT%H:%M:%SZ')

end_time=dt.strptime(end_time,'%Y-%m-%dT%H:%M:%SZ')

period=np.arange(dt(start_time.year,start_time.month,start_time.day),dt(end_time.year,end_time.month,end_time.day),timedelta(days=1))

#%% Creating dictionary

#Creating function to find words in positive and negative function
def create_positive_dictionary_by_day(day):
    dictionary=pd.DataFrame(columns=['Word','Frequency'])
    client = MongoClient('mongodb+srv://Group_fintech:groupfintech@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
    db=client['SorbonneBigData']
    
    sentimental=1
    for documents in db['BTC.X'].find({'Sentiment':sentimental,"TimeStamp":{"$regex": u"{}-{:02d}-{:02d}".format(day.astype(object).year,day.astype(object).month,day.astype(object).day)}}):
        word_list=re.findall(r"[\w']+|[.,!?;$]", documents['Content'])
        word_list = [porter.stem(t) for t in word_list]
        for word in word_list:
            if word in dictionary['Word'].tolist():
                frq=copy.copy(dictionary.iloc[dictionary.index[dictionary['Word']==word].tolist()[0]][1])+1
                dictionary.at[dictionary.index[dictionary['Word']==word].tolist()[0],'Frequency']=frq
            else:
                dictionary=dictionary.append({'Word': word ,'Frequency':1}, ignore_index=True)
    return dictionary

def create_negative_dictionary_by_day(day):
    dictionary=pd.DataFrame(columns=['Word','Frequency'])
    client = MongoClient('mongodb+srv://Group_fintech:groupfintech@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
    db=client['SorbonneBigData']
    
    sentimental=-1
    for documents in db['BTC.X'].find({'Sentiment':sentimental,"TimeStamp":{"$regex": u"{}-{:02d}-{:02d}".format(day.astype(object).year,day.astype(object).month,day.astype(object).day)}}):
        word_list=re.findall(r"[\w']+|[.,!?;$]", documents['Content'])
        word_list = [porter.stem(t) for t in word_list]
        for word in word_list:
            if word in dictionary['Word'].tolist():
                frq=copy.copy(dictionary.iloc[dictionary.index[dictionary['Word']==word].tolist()[0]][1])+1
                dictionary.at[dictionary.index[dictionary['Word']==word].tolist()[0],'Frequency']=frq
            else:
                dictionary=dictionary.append({'Word': word ,'Frequency':1}, ignore_index=True)
    return dictionary

from multiprocessing import Pool
pool = Pool()

#creating positive dictionary
df=list(tqdm(pool.imap(create_positive_dictionary_by_day, period), total=len(period)))
positive_dictionary=df[0].set_index('Word')
for i in tqdm(range(1,len(df))):
    positive_dictionary=positive_dictionary.add(df[i].set_index('Word'), fill_value=0)    

#creating negative dictionary

df=list(tqdm(pool.imap(create_negative_dictionary_by_day, period), total=len(period)))
negative_dictionary=df[0].set_index('Word')
for i in tqdm(range(1,len(df))):
    negative_dictionary=negative_dictionary.add(df[i].set_index('Word'), fill_value=0)
    
negative_dictionary=negative_dictionary.sort_values('Frequency',ascending=False)
positive_dictionary=positive_dictionary.sort_values('Frequency',ascending=False)
positive_dictionary.columns=['Positive Freq']
negative_dictionary.columns=['Negative Freq']
positive_dictionary=positive_dictionary/db['BTC.X'].count_documents({'Sentiment':1})
negative_dictionary=negative_dictionary/db['BTC.X'].count_documents({'Sentiment':-1})

#Combining both dictionary
final_dict=positive_dictionary.add(negative_dictionary, fill_value=0).sort_values('Positive Freq',ascending=False)
final_dict['Pos over Neg']=final_dict['Positive Freq']/final_dict['Negative Freq']

#Removing stopwords from the dictionary
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
final_dict=final_dict.reset_index()

for i in final_dict['Word']:
    if i in stop_words:
        final_dict=final_dict[final_dict['Word']!=i]

#Removing words below the threshold
final_dic=final_dict.fillna(value=0)
final_dict=final_dict[(final_dict['Negative Freq']>0.0005) | (final_dict['Positive Freq']>0.0005)]

final_dict.fillna(value=0).sort_values('Pos over Neg',ascending=False).to_csv('Simple_Dictionary2.csv')


#%% Creating positive and negative word list from the lexicon
os.chdir('H:/Documents/Alternance/Project/')
lexicon=pd.read_csv('Simple_Dictionary2.csv')
lexicon=lexicon[['Word','Classification']]
neg_list=list(lexicon[lexicon['Classification']==-1]['Word'])
pos_list=list(lexicon[lexicon['Classification']==1]['Word'])

# Update lexicon result to the database
import nltk
porter = nltk.PorterStemmer()
import re
import copy

client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']

for i in range(32):
    for documents in tqdm(db['BTC.X'].find({'Custom_Lexicon_Sentiment':{ "$exists" : False }},limit=10000)):
        if documents['Sentiment']==0:
            score=0
            word_list=re.findall(r"[\w']+|[.,!?;$]", documents['Content'])
            word_list = [porter.stem(t) for t in word_list]
            for word in word_list:
                if word in neg_list:
                    score+=-1
                if word in pos_list:
                    score+=1
            if score >0:
                senti=1
            elif score <0:
                senti=-1
            else:
                senti=0
            db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Custom_Lexicon_Sentiment':senti}})
        else:
            db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Custom_Lexicon_Sentiment':documents['Sentiment']}})

#%% Creating positive and negative word list from the teacher lexicon
os.chdir('H:/Documents/Alternance/Project/')
lexicon=pd.read_csv('l2_lexicon.csv',sep=';')
neg_list=list(lexicon[lexicon['sentiment']=='negative']['keyword'])
pos_list=list(lexicon[lexicon['sentiment']=='positive']['keyword'])

# Update lexicon result to the database
pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\w+(?:\.\w+)?%?  # tickers
	  | \@?\w+(?:\.\w+)?%?  # users
      | \.\.\.              # ellipsis
      | [][.,;"'?!():_`-]    # these are separate tokens; includes ], [
    '''
	
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']
cursor=db['BTC.X'].find({'Prof_Lexicon_Sentiment':{ "$exists" : False }},limit=10000)
for i in range(32):
    for documents in tqdm(cursor):
        if documents['Sentiment']==0:
            score=0
            word_list=nltk.regexp_tokenize(documents['Content'], pattern)
    #        word_list=re.findall(r"[\w']+|[.,!?;$]", documents['Content'])
    #        word_list = [porter.stem(t) for t in word_list]
            for word in word_list:
                if word in neg_list:
                    score+=-1
                if word in pos_list:
                    score+=1
            if score >0:
                senti=1
            elif score <0:
                senti=-1
            else:
                senti=0
            db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Prof_Lexicon_Sentiment':senti}})
        else:
            db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Prof_Lexicon_Sentiment':documents['Sentiment']}})

#%% Adding Vader analysis value to the database
# Connecting to the database
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true')

db=client['SorbonneBigData']
collection= db['BTC.X']
# Applying Vader
analyser = SentimentIntensityAnalyzer()

for i in tqdm(range(31)):
    for documents in collection.find({'Vader_sentiment2':{ "$exists" : False }},limit=10000):
        doc_id = documents['_id']
        Vaderdoc = analyser.polarity_scores(documents['Content'])
        Vaderdoc= Vaderdoc.get('compound')
        if Vaderdoc> 0.33:
               Sentiment_vader=1
        elif Vaderdoc< -0.33:
               Sentiment_vader=-1
        else:
               Sentiment_vader=0
        print (Sentiment_vader)
#Insert Vader value to the database
        db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Vader_sentiment2':Sentiment_vader}})
        db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Vader_sentiment':Vaderdoc}})



#%% Adding Textblob analysis value to the database
# Connecting to the database

client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']
collection= db['BTC.X']
# Applying Vader
analyser = SentimentIntensityAnalyzer()

#Vader=[] 54452

for i in tqdm(range(31)):
    for documents in collection.find({'Textblob_Sentiment2':{'$exists':False}},limit=10000):
        doc_id = documents['_id']
        pola = TextBlob(documents['Content']).sentiment.polarity
    #    Vader.append(Vaderdoc)
        if pola> 0.33:
               Sentiment_txt=1
        elif pola< -0.33:
               Sentiment_txt=-1
        else:
               Sentiment_txt=0
        
        db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Textblob_Sentiment2':Sentiment_txt}})
        db['BTC.X'].update_one({'_id':documents['_id']},{'$set':{'Textblob_Sentiment':pola}})



#%% Econometric testing

#%% Import BTC price time series
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']

price=[]
for documents in db['BTC.Price'].find({}):
    price.append([documents['Time'],documents['Price']])
price=pd.DataFrame(price,columns=['Time','Price'])
price['Time']=pd.to_datetime(price['Time'])
price=price.set_index('Time')
price=price[price.index<=dt(2019,9,21,14)]
plt.figure()
price.plot()
price['r_btc'] = (price.Price - price.Price.shift(1)) / price.Price.shift(1)

#%% Import all sentiment time series 
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']

sentimental=[]
for documents in tqdm(db['BTC'].find({})):
    sentimental.append([documents['TimeStamp'],documents['Custom_Lexicon_Sentiment'],documents['Prof_Lexicon_Sentiment'],documents['Textblob_Sentiment'],documents['Textblob_Sentiment2'],documents['Vader_sentiment'],documents['Vader_sentiment2'],documents['Sentiment']])
    
sentimental=pd.DataFrame(sentimental,columns=['Time','Custom_Lexicon_Sentiment','Prof_Lexicon_Sentiment','Textblob_Sentiment_prob','Textblob_Sentiment_binary','Vader_sentiment_prob','Vader_sentiment_binary','Origin_sentiment'])
sentimental=sentimental.set_index('Time')
sentimental.index=pd.to_datetime(sentimental.index.tz_localize(None))

# Resample time series into hour
sentiment_1h=sentimental.resample('1H').mean()
sentiment_1h.plot()
sentiment_1h=sentiment_1h[sentiment_1h.index > dt(2019,1,1) ]

# Export the time series to database
for i in tqdm(range(len(sentiment_1h))):
    insert_element = {"Time": sentiment_1h.index[i], "{}".format(sentiment_1h.columns[0]): sentiment_1h["{}".format(sentiment_1h.columns[0])][i],"{}".format(sentiment_1h.columns[1]): sentiment_1h["{}".format(sentiment_1h.columns[1])][i], "{}".format(sentiment_1h.columns[2]): sentiment_1h["{}".format(sentiment_1h.columns[2])][i], "{}".format(sentiment_1h.columns[3]): sentiment_1h["{}".format(sentiment_1h.columns[3])][i], "{}".format(sentiment_1h.columns[4]): sentiment_1h["{}".format(sentiment_1h.columns[4])][i], "{}".format(sentiment_1h.columns[5]): sentiment_1h["{}".format(sentiment_1h.columns[5])][i], "{}".format(sentiment_1h.columns[6]): sentiment_1h["{}".format(sentiment_1h.columns[6])][i]}
    result = db['Time_series_Data'].insert_one(insert_element)
#  

sentiment_1h=[]
for documents in tqdm(db['Time_series_Data'].find({})):
    sentiment_1h.append([documents['Time'],documents['Custom_Lexicon_Sentiment'],documents['Prof_Lexicon_Sentiment'],documents['Textblob_Sentiment_prob'],documents['Textblob_Sentiment_binary'],documents['Vader_sentiment_prob'],documents['Vader_sentiment_binary'],documents['Origin_sentiment']])
    
sentiment_1h=pd.DataFrame(sentiment_1h,columns=['Time','Custom_Lexicon_Sentiment','Prof_Lexicon_Sentiment','Textblob_Sentiment_prob','Textblob_Sentiment_binary','Vader_sentiment_prob','Vader_sentiment_binary','Origin_sentiment'])
sentiment_1h=sentiment_1h.set_index('Time')
sentiment_1h.index=pd.to_datetime(sentiment_1h.index.tz_localize(None))


#%% Correlation Matrix
test_data=pd.concat([price,sentiment_1h],axis=1)
test_data=test_data.fillna(value=0)
corr_matrix=test_data.corr()

#==============================================================================

#%%Time series analysis for custom lexicon and professor's lexicon
#analyse each timeseries by plotting them
sentiment_1h=sentiment_1h.dropna()
sentiprof=sentiment_1h.iloc[:,1]
senticustom=sentiment_1h.iloc[:,0]
sentiprof=sentiprof.dropna()
senticustom=senticustom.dropna()
sentiprof.astype(float)
senticustom.astype(float)
plt.figure()
btweet= sentiprof.plot(title='One hour average sentiment value(sentiprof)')
plt.figure()
btweetc=senticustom.plot(title='One hour average sentiment value2(senticustom)')
#from this graph, we can find our two sentiment values fluctuates, but 'quite stable'.
sentiprof.mean()
senticustom.mean()
#sentiprof mean value is 0.3615, it is lower than senticustom mean value which is 0.44
#Through this grough,we can observe a positive sentiment of btcoin on tweet from janurary 2019.
price.astype(float)
plt.figure()
priceg= price.Price.plot(title='Price of Bitcoin since Jan 2019(one hour)')
#Through this graph, we can find price of Bitcoin has an increasing trend from Jan 2019 to July 2019)
preturn=(price.Price-price.Price.shift(1))/price.Price.shift(1)
preturn=preturn.dropna()
preturn.mean()
plt.figure()
preturn.plot(title='Price return of Bitcoin since Jan 2019(one hour)')
#From this graph of price return, we can find it has some fluctuations, but 'quite stable' for us.

#%%Stationarity test, Unitroot test
#Professor Lexicon
adfuller(sentiprof,regression='ct')
adfuller(sentiprof,regression='nc')
#p value is small enough, at 95% confidence interval, we can say there is no unitroot in sentiprof, the series is quite stationary.

#Custom Lexicon
adfuller(senticustom,regression='ct')
adfuller(senticustom,regression='nc')
##the p-value is low enough, at 95% confidence level, we can reject the null typothesis which there is a unitroot.
adfuller(price.Price,regression='ct')
##p value is high,0.83. like what we saw in the graph, it has an obvious increasing trend since Jan 2019.
adfuller(preturn,regression='ct')
adfuller(preturn,regression='nc')
#p value is very low to reject the null hypothesis, there is no unitroot for Bitcoin price return.
#%%Set the same datatime and merge all datas togther.
dates2 = pd.date_range('2018-12-22', '2019-09-24', freq='h')
ab=pd.DataFrame(index=dates2,data=sentiprof)
ad=pd.DataFrame(index=dates2,data=preturn)
ac=pd.DataFrame(index=dates2,data=senticustom)
btcdata= pd.concat([ad,ab,ac],axis=1)
btcdata=btcdata.dropna()
btcdata.columns=["preturn","sentiprof","senticustom"]
#%%Ols,for finding the relationship between btc price return and sentiment values.
#Professor Lexicon
btcregprof = smf.ols('preturn~sentiprof',data=btcdata).fit()
btcregprof.summary()
#both beta0=-0.0016 and beta1=0.0048 are significant, beta1 is positive, it means,  when the sentiment value increae 1 unit, the return will increase 0.0048,in 1 hour

#Custom Lexicon
btcregcustom=smf.ols('preturn~senticustom',data=btcdata).fit()
btcregcustom.summary()
#bith beta0=-0.0018 and beta1=0.0046 are significant, it means,when the sentiment value increae 1 unit, the return will increase 0.0046,in 1 hour
#%%AR(1)and find the order of AR
#Professor Lexicon
ab.columns=['sentiprof']
ab['dsentiprof']=ab.sentiprof.shift(1)
ab=ab.dropna()
arprof1=smf.ols('sentiprof~dsentiprof',data=ab).fit()
arprof1.summary()
#coeficients are both significant, 1 hour sentiment has relationship between 1 hour ago.
pacfprof=plot_pacf(sentiprof, lags=24)
pacfcustom=plot_pacf(senticustom, lags=24)
ARMAfit = sm.tsa.arma_order_select_ic(sentiprof, ic=['aic', 'bic'], trend='nc', max_ma=0)
ARMAfit.aic_min_order
ARMAfit.bic_min_order
#AR(4), it means one hour sentiment still have relationship between four hours ago.

#Custom Lexicon
ac.columns=['senticustom']
ac['dsenticustom']=ac.senticustom.shift(1)
ac=ac.dropna()
arcustom1=smf.ols('senticustom~dsenticustom',data=ac).fit()
arcustom1.summary()
#coeficients are both significant
ARMAfit = sm.tsa.arma_order_select_ic(senticustom, ic=['aic', 'bic'], trend='nc', max_ma=0)
ARMAfit.aic_min_order
ARMAfit.bic_min_order
#AR 4,
#%%VAR model and granger causaulity test
# VAR for professor Lexicon
# create  VAR
preturnprof=btcdata[['preturn','sentiprof']]
varmodelprof = VAR(preturnprof)
# select the order of the VAR
print(varmodelprof.select_order(maxlags = 20,trend = 'nc'))
print(varmodelprof.select_order(maxlags = 20,trend = 'c'))
#we choose 13 as our order
result_var = varmodelprof.fit(13)
result_var = varmodelprof.fit(13,trend = 'nc')
result_var.summary()

sum(result_var.roots <= 1) 
# notroots are greater or equal to one, our VAR is stable
# plot impulse response functions
result_var.irf().plot()
# Granger causality
print(result_var.test_causality('preturn','sentiprof'))
print(result_var.test_causality('sentiprof','preturn'))
#we need to reject the null hypothesis that sentiprof does not Granger-cause preturn. so there is granger causality between them. 
#%% VAR for the custom Lexicon
# create a VAR
preturncustom=btcdata[['preturn','senticustom']]
varmodelcustom = VAR(preturncustom)
# select the order of the VAR
print(varmodelcustom.select_order(maxlags = 20,trend = 'nc'))
print(varmodelcustom.select_order(maxlags = 20,trend = 'c'))
#we choose 13 as our order
result_var2 = varmodelcustom.fit(13)
result_var2 = varmodelcustom.fit(13,trend = 'nc')
result_var2.summary()

sum(result_var2.roots <= 1) 
# notroots are greater or equal to one, our VAR is stable
# plot impulse response functions
result_var2.irf().plot()
#A positive impact of return rate will impact sentiment in 1 h, and the effect will dissipate in the following 4 hours, remaining positive
#A positive impact of sentiment will impact return rate in 1 h, but the result is not persistent
# Granger causality
print(result_var2.test_causality('preturn','senticustom'))
print(result_var2.test_causality('senticustom','preturn'))
#H_0: preturn does not Granger-cause senticustom: fail to reject at 5% significance level.

#=====================================================================================
#%% Testing for Vader and Textblob
#%% Testing accuracy of Vader and Textblob
total_label=0
Vader_correct=0
Txtblb_correct=0
for documents in db['BTC.X'].find({'Sentiment':{'$ne':0}}):
    total_label+=1
    if documents['Sentiment']==documents['Vader_sentiment2']:
        Vader_correct+=1
    if documents['Sentiment']==documents['Textblob_Sentiment2']:
        Txtblb_correct+=1
print('Vader accuracy:{:.2%}'.format(Vader_correct/total_label))
print('Textblob accuracy:{:.2%}'.format(Txtblb_correct/total_label))


#%% Import textblog sentiment time series 
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']
#
#textblob_sent=[]
#for documents in db['BTC.X'].find({}):
#    textblob_sent.append([documents['TimeStamp'],documents['Textblob_Sentiment2'],documents['Textblob_Sentiment']])
#
#textblob_sent=pd.DataFrame(textblob_sent,columns=['Time','Textblob_thres','Textblob_Sentiment'])
#textblob_sent['Time']=pd.to_datetime(textblob_sent['Time'])
#textblob_sent=textblob_sent.set_index('Time')
#textblob_sent.index=pd.to_datetime(textblob_sent.index.tz_localize(None))
#
## Resample time series into hour
#textblob_1h=textblob_sent.resample('1H').mean()
#textblob_1h.iloc[:,1].plot()
#textblob_1h=textblob_1h[textblob_1h.index > dt(2019,1,1) ]

textblob_1h=[]
for documents in tqdm(db['Time_series_Data'].find({})):
    textblob_1h.append([documents['Time'],documents['Textblob_Sentiment_binary'],documents['Textblob_Sentiment_prob']])
    
textblob_1h=pd.DataFrame(textblob_1h,columns=['Time','Textblob_thres','Textblob_Sentiment'])
textblob_1h=textblob_1h.set_index('Time')
textblob_1h.index=pd.to_datetime(textblob_1h.index.tz_localize(None))
plt.figure()
textblob_1h.iloc[:,1].plot()

#%% the analysis of textblob
# Putting together both df
frames=[price, textblob_1h]
blob_analysis= pd.concat(frames, axis=1)
plt.figure()
(blob_analysis.Textblob_Sentiment).plot()
blob_analysis= blob_analysis.dropna()
# Testing for stationarity 
# Ho: there is a unit root
adfuller(blob_analysis.Textblob_Sentiment, regression='c')
#adfuller(blob_analysis.Textblob_Sentiment, regression='nc')
#adfuller(blob_analysis.Textblob_Sentiment, regression='ct')
# We reject H0, the series does not have a unit root. The fluctuation is around a mean a bit higher than 0 so we take the regression with 'c', but even for 'nc' and 'ct' still stationary. 

# Check if there is a auto regressive process for sentiments
# Fit an ARMA with the procedure and test with AIC or BIC
ARMATextblob = sm.tsa.arma_order_select_ic(blob_analysis['Textblob_Sentiment'], ic=['aic', 'bic'], trend='nc')
ARMATextblob.aic_min_order
ARMATextblob.bic_min_order

plot_pacf(blob_analysis['Textblob_Sentiment'], lags=24)
plot_acf(blob_analysis['Textblob_Sentiment'], lags=24)
model = ARIMA(blob_analysis['Textblob_Sentiment'], order=(2,0,2))
model_fit = model.fit(disp=0)
plt.figure()
model_fit.resid.plot()
# The sentiment has an AR(2). In an attempt to explain this, we can say that it is a kind of market behavior, in which a bad (good) comments leads to bad (good) comments up to 2 hours (maybe because of loss of confidence of the market).

# An OLS regression  
OLSrText= smf.ols('r_btc ~ Textblob_Sentiment',data = blob_analysis).fit()
OLSrText.summary()
# The sentiment is not significant for the explanation of the bitcoin returns

#Finding the lag for the VAR model
blob_analysis2= blob_analysis.loc[:,['r_btc','Textblob_Sentiment']]
modelblob = VAR(blob_analysis2)
print(modelblob.select_order(maxlags = 25,trend = 'c'))
# We go with the AIC criterion and take a lag of 2
result_blob = modelblob.fit(2)
result_blob.summary()
# Only the 2nd lag of the returns is significant in explaining the returns.
# The first and the second lags of the textblob are significant in explaining the sentiment 
###!!!! Important: the variables do not seem to affect each other!!!!

# Granger causality test between return rate r_btc and Sentimentblob
# H0: The Sentiment does not Granger cause the returns
print(result_blob.test_causality('r_btc','Textblob_Sentiment'))
# H0: The returns do not granger cause the sentiment
print(result_blob.test_causality('Textblob_Sentiment','r_btc'))
# high p-value, we cannot reject H0: There is no Granger causality between the 2 variables

# To check the stability of the VAR model
sum(result_blob.roots >= 1) 
dir(result_blob)

# The VAR model is not stable. Let's take a look at the residuals
print (modelblob.fit(2).resid)
residblob= pd.DataFrame(columns=['resid_r','resid_blob'])
residblob['resid_r']=modelblob.fit(2).resid['r_btc']
residblob['resid_blob']=modelblob.fit(2).resid['Textblob_Sentiment']
adfuller(residblob['resid_r'], regression='c')
# Stationary residuals
adfuller(residblob['resid_blob'], regression='nc')
# Stationary residuals
plot_pacf(residblob['resid_r'], lags=24)
# There seems to be some autocorrelation at the lag 10. Maybe this is causing the instability of the VAR model

# Check the IRF
result_blob.irf().plot()
# the retuns shock effect on sentiment and vice versa are not different from zero. The return shock effect on returns (and sentiment on sentiment), seems to have a response but it doesn't last = short term memory.

#%% Import Vader sentiment time series 
client = MongoClient('mongodb+srv://Group_fintech:groupfintech1@stocktwits-lnyl5.mongodb.net/test?retryWrites=true&w=majority')
db=client['SorbonneBigData']

#Vader_sent=[]
#for documents in db['BTC.X'].find({}):
#    Vader_sent.append([documents['TimeStamp'],documents['Vader_sentiment2'],documents['Vader_sentiment']])
#    
#Vader_sent=pd.DataFrame(Vader_sent,columns=['Time','Vader_thres','Vader_Sentiment'])
#Vader_sent['Time']=pd.to_datetime(Vader_sent['Time'])
#Vader_sent=Vader_sent.set_index('Time')
#Vader_sent.index=pd.to_datetime(Vader_sent.index.tz_localize(None))
## Resample time series into hour
#Vader_1h=Vader_sent.resample('1H').mean()
#Vader_1h.plot()
#Vader_1h=Vader_1h[Vader_1h.index > dt(2019,1,1) ]

Vader_1h=[]
for documents in tqdm(db['Time_series_Data'].find({})):
    Vader_1h.append([documents['Time'],documents['Vader_sentiment_binary'],documents['Vader_sentiment_prob']])
    
Vader_1h=pd.DataFrame(Vader_1h,columns=['Time','Vader_thres','Vader_Sentiment'])
Vader_1h=Vader_1h.set_index('Time')
Vader_1h.index=pd.to_datetime(Vader_1h.index.tz_localize(None))
plt.figure()
Vader_1h.plot()

#%% Econometrics for Vader
# Putting together both df
framesv=[price, Vader_1h]
vader_analysis= pd.concat(framesv, axis=1)
plt.figure()
vader_analysis['Vader_Sentiment'].plot()
vader_analysis= vader_analysis.dropna()
# Testing for stationarity (even though we know they are stationary)
# Ho: there is a unit root
adfuller(vader_analysis.Vader_Sentiment, regression='c')
#adfuller(vader_analysis.Vader_Sentiment, regression='nc')
#adfuller(vader_analysis.Vader_Sentiment, regression='ct')
# We consistently reject H0. The series has no unit root.

# Check if there is a auto regressive process for sentiments
# Fit an ARMA with the procedure and test with AIC or BIC
ARMATextblob = sm.tsa.arma_order_select_ic(vader_analysis['Vader_Sentiment'], ic=['aic', 'bic'], trend='nc')
ARMATextblob.aic_min_order
ARMATextblob.bic_min_order
plot_pacf(vader_analysis['Vader_Sentiment'])
plt.figure()
plot_acf(vader_analysis['Vader_Sentiment'])
# AIC< BIC and the plots do not show the same lags to take, but since this isn't the main objective of our work, we decide on the below arbitrarily
model = ARIMA(vader_analysis['Vader_Sentiment'], order=(4,0,1))
model_fit = model.fit(disp=0)
plt.figure()
model_fit.resid.plot() #The residuals seem stationary 

# An OLS regression  
OLSVader= smf.ols('r_btc ~ Vader_Sentiment',data = vader_analysis).fit()
OLSVader.summary()
# The coeff of Vader sentiment on returns is significant.

# Finding the lag for the VAR model
vader_analysis2= vader_analysis.loc[:,['r_btc','Vader_Sentiment']]
modelvader = VAR(vader_analysis2)
print(modelvader.select_order(maxlags = 25,trend = 'c'))
# We go with the AIC criterion and take a lag of 10
result_vader = modelvader.fit(10)
result_vader.summary()
# Significance for the returns: lag 1 sentiment, lag 9 sentiment lag2 returns and lag 10 returns
# Significance for the sentiment: lag1 sentiment, lag 2 sentiment, lag 3 sentiment, lag 8 sentiment, lag 9 sentiment, lag 2 returns, lag 4 returns, lag 7 returns lag 10 returns.
# The 2 variables seem to affect each other
#(always checking at the 5 perc level)

# Granger causality test between return rate r_btc and Sentimentblob
# H0: The Sentiment does not Granger cause the returns
print(result_vader.test_causality('r_btc','Vader_Sentiment'))
# we reject H0. Sentiment does cause returns
# H0: The return does not Granger cause the sentiment
print(result_vader.test_causality('Vader_Sentiment','r_btc'))
# we reject H0: the returns do granger cause the Vader Sentiment

# To check the stability of the VAR model
sum(result_vader.roots >= 1) 
# The VAR model is not stable. Let's take a look at the residuals
print (modelvader.fit(10).resid)
residvader= pd.DataFrame(columns=['resid_rv','resid_vader'])
residvader['resid_rv']=modelvader.fit(10).resid['r_btc']
residvader['resid_vader']=modelvader.fit(10).resid['Vader_Sentiment']
adfuller(residvader['resid_rv'], regression='c')
# Stationary residuals
adfuller(residvader['resid_rv'], regression='nc')
# Stationary residuals
plot_pacf(residvader['resid_rv'], lags=24)
# There seems to be some autocorrelation at the lag 10. Maybe this is causing the instability of the VAR model

# Check the IRF
result_vader.irf().plot()
# the retuns shock effect on sentiment and vice versa are not different from zero. The return shock effect on returns (and sentiment on sentiment), seems to have a response but it doesn't last = short term memory (sentiment lasts a bit more)