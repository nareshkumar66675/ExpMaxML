import numpy as np

class Stat(object):
    ''' Statistic Object for data transfer
    '''
    def __init__(self, mean = [],variance =[],pi=[]):
        self.mean = mean
        self.variance=variance
        self.pi=pi

    def SetInitialValues(self,cluster,data):
        ''' Initialize values based on data set
            1) Sort Data and split into group based on cluster value
            2) Mean,Variance - for each group
            3) pi - 1/Cluster
        '''
        spltdData = list(self.split(np.sort(data),cluster))
        self.pi = []
        self.mean = []
        self.variance = []
        for x in range(cluster):
            self.pi.append(1/cluster)
            self.mean.append(np.mean(spltdData[x]))
            self.variance.append(np.var(spltdData[x]))

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

