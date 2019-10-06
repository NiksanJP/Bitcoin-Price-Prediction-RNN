import numpy as np 
import pandas as pd 
import os 
import sys 
import requests
import json
import datetime
import time

import matplotlib.pyplot as plt

#Downloads all the datasets from Gemini website for per minute
def downloadAllDatasets():
    year = datetime.date.today().year
    
    for i in range(2015, int(year), 1):
        url = 'http://www.cryptodatadownload.com/cdd/gemini_BTCUSD_' + str(i) + '_1min.csv'
        response = requests.get(url, allow_redirects=True)
        open('gemini_BTCUSD_' + str(i) + '_1min.csv', 'wb').write(response.content)

#Updates to Latest Gemini Dataset
def downloadLatestDatasetsOnly():
    url = 'http://www.cryptodatadownload.com/cdd/gemini_BTCUSD_' + str(datetime.date.today().year) + '_1min.csv'
    response = requests.get(url, allow_redirects=True)
    open('gemini_BTCUSD_' + str(datetime.date.today().year) + '_1min.csv', 'wb').write(response.content)

#Joins Dataset that are downloaded
def joinDatasets():
    y15 = pd.read_csv("gemini_BTCUSD_2015_1min.csv", skiprows=range(0,1))
    y16 = pd.read_csv("gemini_BTCUSD_2016_1min.csv", skiprows=range(0,1))
    y17 = pd.read_csv("gemini_BTCUSD_2017_1min.csv", skiprows=range(0,1))
    y18 = pd.read_csv("gemini_BTCUSD_2018_1min.csv", skiprows=range(0,1))
    y19 = pd.read_csv("gemini_BTCUSD_2019_1min.csv", skiprows=range(0,1))
    
    bitcoinDataset = y19.merge(y18, how='outer')
    bitcoinDataset = bitcoinDataset.merge(y17, how='outer')
    bitcoinDataset = bitcoinDataset.merge(y16, how='outer')
    bitcoinDataset = bitcoinDataset.merge(y15, how='outer')
    
    bitcoinDataset.to_csv('bitcoinDataset.csv')
        
    b = pd.read_csv('bitcoinDataset.csv')
    print(b.head())
    
def joinDatasetsFrom2017():
    y17 = pd.read_csv("gemini_BTCUSD_2017_1min.csv", skiprows=range(0,1))
    y18 = pd.read_csv("gemini_BTCUSD_2018_1min.csv", skiprows=range(0,1))
    y19 = pd.read_csv("gemini_BTCUSD_2019_1min.csv", skiprows=range(0,1))
    
    bitcoinDataset = y19.merge(y18, how='outer')
    bitcoinDataset = bitcoinDataset.merge(y17, how='outer')
    
    bitcoinDataset.to_csv('bitcoinDataset.csv')
        
    b = pd.read_csv('bitcoinDataset.csv')
    print(b.head())

#Update Dataset using APIs
def updateUsingAPIs():
    bitcoinDataset = pd.read_csv("bitcoinDataset.csv")
    print(bitcoinDataset.shape)
    
    last = int(bitcoinDataset.loc[0, 'Unix Timestamp'] / 1000)
    now = int(time.time())
    sets = now - last
    
    minsDownloadNeeded = int((sets - sets%60) / 60)
    
    print("TIME : " , now)
    print("LAST : ", last)
    print("DIFF : ", sets)
    print("\n Minutes : ", minsDownloadNeeded)
    
    Set = int(minsDownloadNeeded / 2000)
    lastSet = int(minsDownloadNeeded % 2000)
    totalSet = [Set, lastSet]
    
    for i in range(len(totalSet)):
        if totalSet[i] <= 0:
            totalSet[i] = 0
    
    print(totalSet)
    
    for x in totalSet:
        if x != 0:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=' + str(x)
            response = requests.get(url, allow_redirects=True)
            
            response = str(response.json())
            start = response.index("[{'time'") + 1
            end = response.index("[{'time'") + response[start:].index("}],") + 1

            response = response[start:end]
            response = response.split('},')

            temp = pd.DataFrame(columns=bitcoinDataset.columns.values)

            for i in range(len(response)):
                response[i] = response[i].replace('{','')
                response[i] = response[i].replace(',','')
                response[i] = response[i].replace("'","")
                response[i] = response[i].split(' ')

                if i == 0:
                    date = str(datetime.datetime.fromtimestamp(int(response[i][1])))[:-3]
                    vol = int(float(response[i][13])) - int(float(response[i][11]))
                
                    #                       UNIX     date    sym        open            high                Low             Close       Volume
                    temp = temp.append({    'Unix Timestamp' : int(int(response[i][1]) * 1000), 
                                            'Date' : date, 
                                            'Symbol' : 'BTCUSD', 
                                            'Open' : response[i][9], 
                                            'High' : response[i][5], 
                                            'Low' : response[i][7], 
                                            'Close' : response[i][3], 
                                            'Volume' : str(vol)}, ignore_index=True)
                else:
                    date = str(datetime.datetime.fromtimestamp(int(response[i][2])))[:-3]
                    vol = int(float(response[i][14])) - int(float(response[i][12]))
                
                    #                       UNIX     date    sym        open            high                Low             Close       Volume
                    temp = temp.append({    'Unix Timestamp' : int(int(response[i][2]) * 1000), 
                                            'Date' : date, 
                                            'Symbol' : 'BTCUSD', 
                                            'Open' : response[i][10], 
                                            'High' : response[i][6], 
                                            'Low' : response[i][8], 
                                            'Close' : response[i][4], 
                                            'Volume' : str(vol)}, ignore_index=True)

            temp = temp.sort_values('Unix Timestamp', ascending=False)
            bitcoinDataset = pd.concat([temp, bitcoinDataset])
            if len(bitcoinDataset.columns) == 12:
                bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1], bitcoinDataset.columns[2], bitcoinDataset.columns[3]], axis = 'columns')
            elif len(bitcoinDataset.columns) == 11:
                bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1], bitcoinDataset.columns[2]], axis = 'columns')
            elif len(bitcoinDataset.columns) == 10:
                bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1]], axis = 'columns')
            elif len(bitcoinDataset.columns) == 9:
                bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0]], axis = 'columns')
            bitcoinDataset = bitcoinDataset.sort_values('Unix Timestamp', ascending=False)
            bitcoinDataset.to_csv('bitcoinDataset.csv')
            
def getLivePrice():
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=1'
    response = requests.get(url, allow_redirects=True)
    bitcoinDataset = pd.read_csv("bitcoinDataset.csv")
    
    response = str(response.json())
    start = response.index("[{'time'") + 1
    end = response.index("[{'time'") + response[start:].index("}],") + 1

    response = response[start:end]
    response = response.split('},')

    response = response[len(response) - 1]
    response = response.replace('{','')
    response = response.replace(',','')
    response = response.replace("'","")
    response = response.split(' ')
    
    date =  str(datetime.datetime.fromtimestamp(int(response[2])))[:-3]
    vol =   int(float(response[14])) - int(float(response[12]))
    
    print(str(int(response[2]) * 1000))
    
    bitcoinDataset = bitcoinDataset.append({'Unix Timestamp' : int(int(response[2]) * 1000), 
                                            'Date' : date, 
                                            'Symbol' : 'BTCUSD', 
                                            'Open' : response[10], 
                                            'High' : response[6], 
                                            'Low' : response[8], 
                                            'Close' : response[4], 
                                            'Volume' : str(vol)}, ignore_index=True
    )
    if len(bitcoinDataset.columns) == 12:
        bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1], bitcoinDataset.columns[2], bitcoinDataset.columns[3]], axis = 'columns')
    elif len(bitcoinDataset.columns) == 11:
        bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1], bitcoinDataset.columns[2]], axis = 'columns')
    elif len(bitcoinDataset.columns) == 10:
        bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0], bitcoinDataset.columns[1]], axis = 'columns')
    elif len(bitcoinDataset.columns) == 9:
        bitcoinDataset = bitcoinDataset.drop([bitcoinDataset.columns[0]], axis = 'columns')
        
    bitcoinDataset = bitcoinDataset.sort_values('Unix Timestamp', ascending=False)
    
    bitcoinDataset.to_csv('bitcoinDataset.csv')
    print(bitcoinDataset.head())
    print("SLEEPING")
    time.sleep(55)

def initializeOnly():
    print("DOWNLOAD ALL DATASETS")
    downloadAllDatasets()
    print("DOWNLOAD LATEST ONLY")
    downloadLatestDatasetsOnly()
    print("JOIN DATASET")
    joinDatasets()  
    print("GET MISSING DATASET")
    updateUsingAPIs()
    
def initializeOnly2017():
    print("DOWNLOAD ALL DATASETS")
    downloadAllDatasets()
    print("DOWNLOAD LATEST ONLY")
    downloadLatestDatasetsOnly()
    print("JOIN DATASET")
    joinDatasetsFrom2017()  
    print("GET MISSING DATASET")
    updateUsingAPIs()

def initializeAndRun():
    downloadAllDatasets()
    downloadLatestDatasetsOnly()
    joinDatasets()  
    updateUsingAPIs()

    while True:
        if (int(time.time()) % 60 == 0):
            while True:
                print("ADDING LIVE PRICE")
                getLivePrice()

def updateLatestAndRun():
    updateUsingAPIs()

    while True:
        if (int(time.time()) % 60 == 0):
            while True:
                print("ADDING LIVE PRICE")
                getLivePrice()
                
def buySellStayDataSorter():
    df = pd.read_csv('bitcoinDataset.csv')
    df = df.sort_values('Unix Timestamp', ascending=True)
    columns = df.columns.values
    columns = np.append(columns, ['Decision'])
    columns = columns[1:]
    print("COLUMNs :", columns)
    
    #temp = pd.DataFrame(columns=columns)
    temp = pd.read_csv('SortedBitcoinDataset.csv')
    
    for i in range(35000, df.shape[0]):
        # 0  is sell
        # 1 is buy
        # 2 is do nothing
        change = df.iloc[i+1]['Close'] - df.iloc[i]['Close']
        if abs(change) > 2.5:
            if change > 0 :
                decision = '0'
            else:
                decision = '1'
        else:
            decision = '2'
        
        temp = temp.append({    'Unix Timestamp' : df.iloc[i]['Unix Timestamp'], 
                                            'Date' : df.iloc[i]['Date'], 
                                            'Symbol' : 'BTCUSD', 
                                            'Open' : df.iloc[i]['Open'], 
                                            'High' : df.iloc[i]['High'], 
                                            'Low' : df.iloc[i]['Low'], 
                                            'Close' : df.iloc[i]['Close'], 
                                            'Volume' : df.iloc[i]['Volume'],
                                            'Decision' : decision
                            }, ignore_index=True)
        
        if i%1000 == 0:
            print(i)
        
        if i%10000:
            temp = temp.sort_values('Unix Timestamp', ascending=True)
            temp.to_csv('SortedBitcoinDataset.csv')
    
    temp = temp.sort_values('Unix Timestamp', ascending=True)
    temp.to_csv('SortedBitcoinDataset.csv')
    
def buildFinalTrainingData():
    print("LOADING DATA")

    #Load Data
    df = pd.read_csv('bitcoinDataset.csv')
    df = df[['Unix Timestamp', 'Open', 'High', 'Low', 'Close']]
    df = df.sort_values('Unix Timestamp', ascending=True)
    df = df.drop('Unix Timestamp', axis=1)
    df = df.dropna()

    #Convert to Numpy
    df = df.to_numpy()

    #Select all coluns for X and just one for Y
    trainX = df
    trainY = df[:, df.shape[1]-1]

    rows = int(trainX.shape[0])
    columns = int(trainX.shape[1])

    #Drop last and first to make first to a second value and Last X value to None
    trainX = trainX[:(rows-1)]
    trainY = trainY[1:rows]
    trainY = trainY.reshape(trainY.shape[0], 1)
    
    data = np.concatenate((trainX,trainY), axis=1)
    #print(data.shape)
    np.random.shuffle(data)
    
    trainX = data[:, :4]
    trainY = data[:, 4]
    
    #print(trainX.shape, trainY.shape)
    for i in range(10) : 
        print(trainX[i], trainY[i])

    #Create Decision TAB
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], 1)
    print(trainX.shape, trainY.shape)
    trainY = trainX[:,0, 3] - trainY[:,0]
    print("AFTER RESHAPE ", trainX.shape, trainY.shape)

    #Reshape to trainable values
    #trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], 1)

    #Declare Broker fee
    #########################################################
    brokerFees = 2.5
    #########################################################

    trainY[trainY>brokerFees] = 1000000
    trainY[trainY<-brokerFees] = 2000000
    trainY[(trainY<brokerFees)&(trainY>-brokerFees)] = 3000000

    #Convert 1 to 0 and 2 to 1 and 3 to 2
    # 0  is sell
    # 1 is buy
    # 2 is do nothing
    indexes = np.where(trainY==1000000)
    trainY[indexes] = int(0)
    indexes = np.where(trainY==2000000)
    trainY[indexes] = int(1)
    indexes = np.where(trainY==3000000)
    trainY[indexes] = int(2)

    #Display Data
    #plt.hist(trainY)
    #plt.savefig('dataDistribution.png')
    #plt.waitforbuttonpress()

    #TODO : Get the count of sell data
    #Get the same amount of Other Data
    #Combine them to have equal numbers for training
    #Convert others to test Data
    X = trainX
    Y = trainY


    if np.count_nonzero(Y==0) < np.count_nonzero(Y==1):
        count = np.count_nonzero(Y==0)
    else:
        count = np.count_nonzero(Y==1)

    sellIndexes = np.where(trainY==0)
    buyIndexes = np.where(trainY==1)
    DNsIndexes = np.where(trainY==2)

    buysX = X[buyIndexes]
    buysY = Y[buyIndexes]

    sellX = X[sellIndexes]
    sellY = Y[sellIndexes]

    DNsX = X[DNsIndexes]
    DNsY = Y[DNsIndexes]

    trainX = np.concatenate((buysX[:count], sellX[:count], DNsX[:count]), axis = 0)
    trainY = np.concatenate((buysY[:count], sellY[:count], DNsY[:count]), axis = 0)

    testX = np.concatenate((buysX[count:], sellX[count:], DNsX[count:]), axis = 0)
    testY = np.concatenate((buysY[count:], sellY[count:], DNsY[count:]), axis = 0)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainY = np.reshape(trainY, (trainY.shape[0], 1))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testY = np.reshape(testY, (testY.shape[0], 1))

    np.save('trainTestData/trainX', trainX)
    np.save('trainTestData/trainY', trainY)
    np.save('trainTestData/testX', testX)
    np.save('trainTestData/testY', testY)
        
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    
    plt.hist(testY)
    plt.hist(trainY)
    plt.savefig('dataDistribution.png')
    #plt.show()
    #plt.waitforbuttonpress()

#initializeAndRun()
buildFinalTrainingData()