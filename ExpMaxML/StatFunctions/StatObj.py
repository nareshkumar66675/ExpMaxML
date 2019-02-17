import numpy as np

class Stat(object):

    def __init__(self, mean = [],variance =[],pi=[]):
        self.mean = mean
        self.variance=variance
        self.pi=pi

    def SetInitialValues(self,cluster,data):
        spltdData = list(self.split(np.sort(data),cluster))
        for x in range(cluster):
            #temp = yield spltdData
            self.pi.append(1/cluster)
            self.mean.append(np.mean(spltdData[x]))
            self.variance.append(np.var(spltdData[x]))

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

