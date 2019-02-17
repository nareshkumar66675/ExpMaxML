import pandas as pd
from StatFunctions.PDF import *
from StatFunctions.EM import *
import StatFunctions.StatObj as StatData
import matplotlib.pyplot as plt 
import numpy as np

liverFilePath = r'C:\Users\kumar\Downloads\indian-liver-patient-records\indian_liver_patient.csv'
liverDF = pd.read_csv(liverFilePath)

albumin=liverDF.iloc[:,8].values


#stat = StatData.Stat([0,1,1.5],[1,0.5,0.2],[1/3,1/3,1/3])

stat = StatData.Stat()
stat.SetInitialValues(3,albumin)

logs = []
delts=[]
for x in range(30):
    print('Iteration '+str(x))
    rValue = EStep(albumin,3,stat)
    stat = MStep(albumin,3,stat,rValue)
    print('Printing Mean')
    print(*stat.mean, sep = ", ")
    #print('Printing Variance')
    #print(*stat.variance, sep = ", ")
    logLikely = LogLikelyHood(albumin,stat)
    print('Printing Log Likelyhood')
    print(logLikely)
    logs.append(logLikely)
    if x!=0:
        delts.append(logs[x]-logs[x-1])
        if len(delts)>1 and (delts[len(delts)-2]-delts[len(delts)-1])/delts[len(delts)-2]<.1:
            break

plt.title('Likelihood Convergence')
plt.plot(logs)
plt.ylabel('Likelihood')
plt.show()