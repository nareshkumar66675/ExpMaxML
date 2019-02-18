import pandas as pd
from StatFunctions.PDF import *
from StatFunctions.EM import *
import StatFunctions.StatObj as StatData
import matplotlib.pyplot as plt 
import numpy as np
import sys

liverFilePath = r'C:\Users\kumar\Downloads\indian-liver-patient-records\indian_liver_patient.csv'
wineFilePath = r"C:\Users\kumar\OneDrive\Documents\Projects\ExpMaxML\Dataset\winequality-red.csv"

liverDF = pd.read_csv(liverFilePath)
wineDF = pd.read_csv(wineFilePath)

liverDF.replace('', np.nan, inplace=True)
liverDF = liverDF.dropna()


wineDF.replace('', np.nan, inplace=True)
wineDF = wineDF.dropna()


while True:

    
    print('1.Liver')
    print('2.Wine')
    print('3.Custom(File Path Needed)')
    dataChoice = int(input('Select one Dataset from above : '))

    if dataChoice == 1:
        selectedDF = liverDF
    elif dataChoice == 2:
         selectedDF = wineDF
    elif dataChoice == 3:
        customFilePath = str(input('Enter full file Path : '))
        customDF = pd.read_csv(customFilePath)
        customDF.replace('', np.nan, inplace=True)
        customDF = liverDF.dropna()
        selectedDF = customDF
    else:
        choice = input('Enter Valid Option. Press Y to Restart and N to Exit: ')
        if str.lower(choice) == 'n':
            sys.exit()
        else:
            continue


    numCols = list(selectedDF.select_dtypes([np.number]).columns.values)

    print('Available Columns')
    for i in range(len(numCols)):
        print(str(i)+"  "+numCols[i])

    colNo = input("Select one Column(Enter number): ")

    dataArr=selectedDF.iloc[:,selectedDF.columns.get_loc(numCols[int(colNo)])].values
    #stat = StatData.Stat([0,1,1.5],[1,0.5,0.2],[1/3,1/3,1/3])

    print('Close the Graph to continue.')

    plt.hist(dataArr, bins='auto')
    plt.title("Histogram for Column : " +numCols[int(colNo)])
    plt.plot()
    plt.show()

    cluster = FindOptimalClusters(dataArr) #int(input("Enter number of clusters"))
    #cluster = int(input("Enter number of clusters"))
    if cluster == 0:
        print('Cannot find Optimal number of Clusters')
        try:
            cluster = int(input('Enter Cluster to continue(integer to continue string to restart:'))
        except (ValueError, TypeError):
            continue
        
    else:
        print('Optimal Cluster is '+str(cluster))
    

    stat = StatData.Stat()
    stat.SetInitialValues(cluster,dataArr)

    logs = []
    delts=[]
    for x in range(25):
        print('*******  Iteration '+str(x) +'   *********')
        rValue = EStep(dataArr,cluster,stat)
        stat = MStep(dataArr,cluster,stat,rValue)
        print('Printing Mean')
        print(*stat.mean, sep = ", ")
        #print('Printing Variance')
        #print(*stat.variance, sep = ", ")
        logLikely = LogLikelyHood(dataArr,stat)
        print('Log Likelyhood :'+ str(logLikely))
        logs.append(logLikely)
        if x!=0:
            delts.append(logs[x]-logs[x-1])
            print((delts[len(delts)-2]-delts[len(delts)-1])/delts[len(delts)-2])
            if len(delts)>1:
                if delts[len(delts)-2] != 0 and ((delts[len(delts)-2]-delts[len(delts)-1])/delts[len(delts)-2]<.1):    
                    break

    print('Final Mean Values for each cluster:')
    for i in range(len(stat.mean)):
        print('Cluster '+str(i)+"  : "+str(stat.mean[i]))

    print('Close the Graph to continue.')

    plt.title('Likelihood Convergence: ' + numCols[int(colNo)])
    plt.plot(logs)
    plt.ylabel('Likelihood')
    plt.show()



    choice = input('Press Y to Restart and N to Exit: ')

    if str.lower(choice) == 'n':
        sys.exit()

