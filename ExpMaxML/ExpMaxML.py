import pandas as pd
from StatFunctions.PDF import *
from StatFunctions.EM import *
import StatFunctions.StatObj as StatData
import matplotlib.pyplot as plt 
import numpy as np
import sys

liverFilePath = r'C:\Users\kumar\Downloads\indian-liver-patient-records\indian_liver_patient.csv'
liverDF = pd.read_csv(liverFilePath)
liverDF.replace('', np.nan, inplace=True)
liverDF = liverDF.dropna()
#dataArr=liverDF.iloc[:,8].values


while True:
    numCols = list(liverDF.select_dtypes([np.number]).columns.values)

    print('Available Columns')
    for i in range(len(numCols)):
        print(str(i)+"  "+numCols[i])

    colNo = input("Select one Column(Enter number): ")

    dataArr=liverDF.iloc[:,liverDF.columns.get_loc(numCols[int(colNo)])].values
    #stat = StatData.Stat([0,1,1.5],[1,0.5,0.2],[1/3,1/3,1/3])

    cluster = FindOptimalClusters(dataArr) #int(input("Enter number of clusters"))
    #cluster = int(input("Enter number of clusters"))
    if cluster == 0:
        print('Cannot find Optimal number of Clusters')
        try:
            cluster = int(input('Enter Cluster to continue(integer to continue string to restart:'))
        except (ValueError, TypeError):
            continue
        
    else:
        print('Optimal Cluster is'+str(cluster))
    

    stat = StatData.Stat()
    stat.SetInitialValues(cluster,dataArr)

    logs = []
    delts=[]
    for x in range(25):
        print('Iteration '+str(x))
        rValue = EStep(dataArr,cluster,stat)
        stat = MStep(dataArr,cluster,stat,rValue)
        print('Printing Mean')
        print(*stat.mean, sep = ", ")
        #print('Printing Variance')
        #print(*stat.variance, sep = ", ")
        logLikely = LogLikelyHood(dataArr,stat)
        print('Printing Log Likelyhood')
        print(logLikely)
        logs.append(logLikely)
        if x!=0:
            delts.append(logs[x]-logs[x-1])
            print((delts[len(delts)-2]-delts[len(delts)-1])/delts[len(delts)-2])
            if len(delts)>1:
                if delts[len(delts)-2] != 0 and ((delts[len(delts)-2]-delts[len(delts)-1])/delts[len(delts)-2]<.2):    
                    break

    plt.title('Likelihood Convergence')
    plt.plot(logs)
    plt.ylabel('Likelihood')
    plt.show()

    print('Final Mean Values for each cluster:')
    for i in range(len(stat.mean)):
        print('Cluster '+str(i)+"  : "+str(stat.mean[i]))

    choice = input('Press Y to Restart and N to Exit: ')

    if str.lower(choice) == 'n':
        sys.exit()

