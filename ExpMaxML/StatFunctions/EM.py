import StatFunctions.StatObj
import numpy as np
from StatFunctions.PDF import *


def EStep(data,cluster,Stat):
        rValue = np.zeros((len(data),cluster))  
        for x,mean,var,pi in zip(range(cluster),Stat.mean,Stat.variance,Stat.pi):
            rValue[:,x] = pi* pdf(data,mean,var)
        for i in range(len(rValue)):
            rValue[i] = rValue[i]/(np.sum(Stat.pi)*np.sum(rValue,axis=1)[i])
        return rValue

def MStep(data,cluster,Stat,rValue):
        fractPnt = []
        for x in range(len(rValue[0])):
            fractPnt.append(np.sum(rValue[:,x]))
        for k in range(len(fractPnt)):
            Stat.pi[k] = (fractPnt[k]/np.sum(fractPnt))
        Stat.mean = np.sum(data.reshape(len(data),1)*rValue,axis=0)/fractPnt
        var_c = []
        for x in range(len(rValue[0])):
            var_c.append((1/fractPnt[c])*np.dot(((np.array(rValue[:,x]).reshape(len(data),1))*(data.reshape(len(data),1)-Stat.mean[x])).T,(data.reshape(len(data),1)-Stat.mean[x])))
        #for x in range(len(var_c)):
        #    Stat.variance[x] = var_c[x][0][0]
        return Stat


