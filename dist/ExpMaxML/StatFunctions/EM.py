from StatFunctions.StatObj import *
import numpy as np
from StatFunctions.PDF import *
import math
import matplotlib.pyplot as plt 

def EStep(data,cluster,Stat):
    ''' Calculates probability of each data for each cluster
    '''
    rValue = np.zeros((len(data),cluster))  
    for x,mean,var,pi in zip(range(cluster),Stat.mean,Stat.variance,Stat.pi):
        rValue[:,x] = pi* pdf(data,mean,var)
    for i in range(len(rValue)):
        rValue[i] = rValue[i]/(np.sum(Stat.pi)*np.sum(rValue,axis=1)[i])
    return rValue

def MStep(data,cluster,Stat,rValue):
    ''' 
        Calculate fraction point for each cluster
        Update all paramaters accordingly - Using M Step Formulas
    '''
    fractPnt = []
    for x in range(len(rValue[0])):
        fractPnt.append(np.sum(rValue[:,x]))
    for k in range(len(fractPnt)):
        Stat.pi[k] = (fractPnt[k]/np.sum(fractPnt))
    Stat.mean = np.sum(data.reshape(len(data),1)*rValue,axis=0)/fractPnt
    var_c = []
    for x in range(len(rValue[0])):
        var_c.append((1/fractPnt[x])*np.dot(((np.array(rValue[:,x]).reshape(len(data),1))*(data.reshape(len(data),1)-Stat.mean[x])).T,(data.reshape(len(data),1)-Stat.mean[x])))
    #for x in range(len(var_c)):
    #    Stat.variance[x] = (len(data)-1 /len(data))* var_c[x][0][0]
    return Stat


def LogLikelyHood(data,Stat: Stat):
    ''' 
        Find Likelihood for the given data and stat paramaters
    '''
    logLik = 0
    for x in range(len(Stat.mean)):
        varS = Stat.variance[x] * Stat.variance[x]
        term1 = ((len(data)) * math.log(2 * math.pi))/(-1*2)
        term2 = ((len(data)) * math.log(varS))/(-1*2)
        term3 = np.sum((np.power(np.subtract(data,Stat.mean[x]),2))/(-1*2*varS),axis=0)
        logLik += term1+term2+term3
    return logLik

def FindOptimalClusters(dataArr):
    '''
        Finds Optimal Clusters for the given Data Set
        1) Run for each cluster starting from 2 to 9
        2) Execute EStep and Mstep for 10 times
        3) Record Log likelihood for each iterations
        4) If the log likelihood values are monotonically increasing, it is the optimal cluster
        5) Else continue with next cluster
    '''
    optCluster = 0
    for cluster in range(2,10):
        logs=[]
        print('Checking feasibility of Cluster:' + str(cluster))
        data= np.empty(len(dataArr))
        np.copyto(data,dataArr)
        stat = Stat()
        stat.SetInitialValues(cluster,data)
        for x in range(10):
            rValue = EStep(data,cluster,stat)
            stat = MStep(data,cluster,stat,rValue)
            logLikely = LogLikelyHood(data,stat)
            logs.append(logLikely)
        if all(x<y for x, y in zip(logs, logs[1:])):
            optCluster = cluster
            break
    return optCluster