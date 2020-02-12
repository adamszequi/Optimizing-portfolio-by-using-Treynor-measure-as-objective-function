# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:21:36 2020

@author: Dell
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import fmin



dataLocations=[r'C:\Users\Dell\Desktop\data\2010-2019\GOIL VWAP closing prces (2).xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\TOTAL VWAP closing prces (2).xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\TBL VWAP closing prces (2).xlsx']

uploadedData=[]
def retriveData(ticker):
        for tick in ticker:
            data=pd.read_excel(tick,parse_dates=[1]) 
            uploadedData.append(data)
        return uploadedData
    
#define function for returns of stocks used in analysis 
logReturnsList=[]
datesList=[]
def returnsGroupedByYear(tickers:list)->list:
    tickers=retriveData(tickers)
    for _ in tickers:
        logReturns=np.log(_['Closing Price VWAP (GHS)'][1:].values/\
                   _['Closing Price VWAP (GHS)'][:-1].values)
        logReturnsList.append(logReturns)
    for _ in range(len(logReturnsList[1])):
        datesList.append(tickers[1]['Date'][_][:4])
    finalReturnsGroupedByYear=pd.DataFrame(logReturnsList).T
    finalReturnsGroupedByYear.index=pd.Series(datesList)
    finalReturnsGroupedByYear.columns=['GOIL','TOTAL','TBL']
    return finalReturnsGroupedByYear.groupby(finalReturnsGroupedByYear.index).sum()

#declaring variables to be used in upcoming functions
returns=sp.array(returnsGroupedByYear(dataLocations))
rf=0.0003 
betaGiven=(0.8,0.4,0.3)

#estimating portfolio beta
def portfolioBeta(betaGiven:list,weights:list)->list:
    return sp.dot(betaGiven,weights)#sp.dot multiplies weight in a similar way 
                                    #as zip(listA,listB)

   
#estimating treynor ratio
def treynorMeasure(tickers,weights):
    Beta=portfolioBeta(betaGiven,weights)
    meanReturn=sp.mean(returns,axis=0)
    returnsArray=sp.array(meanReturn)
    return (sp.dot(weights,returnsArray)-rf)/Beta

def negativeTreynorForNminusOneStock(weights):
    weights2=sp.append(weights,1-sum(weights))#adds 1-sum(weights) to weights
    return -treynorMeasure(returns,weights2)

#for equally weighted portfolios
print('Efficient porfolio (Treynor ratio)')
print('Treynor ratio for an equal-weighted portfolio')
equalWeights=np.ones(len(dataLocations),dtype=float)*1/len(dataLocations)
print(treynorMeasure(returns,equalWeights))

#for n-1 portfolios with negative treynor
weight0=sp.ones(len(dataLocations)-1,dtype=float)*1/len(dataLocations)
weight1=fmin(negativeTreynorForNminusOneStock,weight0)
finalWeights=sp.append(weight1,1-sum(weight1))
finalTreynor=treynorMeasure(returns,finalWeights)
print ('Optimal weights are ')
print (finalWeights)
print ('final Sharpe ratio is ')
print(finalTreynor)
