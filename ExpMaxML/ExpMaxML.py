import pandas as pd
from StatFunctions.PDF import *
from StatFunctions.EM import *
import StatFunctions.StatObj as StatData
import matplotlib.pyplot as plt 
import numpy as np
import sys
import os.path

try:
    #EM with GMM Algorithm
    print('Expectation Maximization using Gaussian Mixture Model')

    #Read files from Dataset folder
    dataSetFolder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    liverFilePath = dataSetFolder+ os.path.join('\Dataset\indian_liver_patient.csv')
    wineFilePath = dataSetFolder+ os.path.join('\Dataset\winequality-red.csv')

    liverDF = pd.read_csv(liverFilePath)
    wineDF = pd.read_csv(wineFilePath)

    liverDF.replace('', np.nan, inplace=True)
    liverDF = liverDF.dropna()
    wineDF.replace('', np.nan, inplace=True)
    wineDF = wineDF.dropna()

    #Repeat until user wants to exit
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
            customDF = customDF.dropna()
            selectedDF = customDF
        else:
            choice = input('Enter Valid Option. Press Y to Restart and N to Exit: ')
            if str.lower(choice) == 'n':
                sys.exit()
            else:
                continue

        #Get Numeric Columns from Dataset
        numCols = list(selectedDF.select_dtypes([np.number]).columns.values)

        print('Available Columns')
        for i in range(len(numCols)):
            print(str(i) + "  " + numCols[i])

        colNo = input("Select one Column(Enter number): ")

        #Retrieve Column data based on user input
        dataArr = selectedDF.iloc[:,selectedDF.columns.get_loc(numCols[int(colNo)])].values
        #stat = StatData.Stat([0,1,1.5],[1,0.5,0.2],[1/3,1/3,1/3])

        print('Close the Graph to continue.')

        #Shows the Histogram of Data
        plt.hist(dataArr, bins='auto')
        plt.title("Histogram for Column : " + numCols[int(colNo)])
        plt.plot()
        plt.show()

        # Finds Optimal Clusters based on Likelihood
        cluster = FindOptimalClusters(dataArr) #int(input("Enter number of clusters"))

        if cluster == 0:
            print('Cannot find Optimal number of Clusters')
            try:
                cluster = int(input('Enter Cluster to continue(integer to continue string to restart:'))
            except (ValueError, TypeError):
                continue      
        else:
            print('Optimal Cluster is ' + str(cluster))
    
        # Initiliaze Stat Object - Contains mean,variance and pi
        # Predict inital stat values based on dataset
        stat = StatData.Stat()
        stat.SetInitialValues(cluster,dataArr)


        ''' Execute the Alogorithm 
            1. Execute E Step - Calculates probability of each data for each cluster
            2. Execute M Step - Calculate the weigh and update mean,stat and pi
            3. Find Log likelihood
            4. If Converges Exit, else Repeat
        '''
        logs = []
        delts = []
        for x in range(25):
            print('*******  Iteration ' + str(x) + '   *********')
            rValue = EStep(dataArr,cluster,stat)
            stat = MStep(dataArr,cluster,stat,rValue)
            print('Printing Mean')
            print(*stat.mean, sep = ", ")
            logLikely = LogLikelyHood(dataArr,stat)
            print('Log Likelyhood :' + str(logLikely))
            logs.append(logLikely)

            ''' To find if likelihood converges
                Find the delta of each log likelihood values
                For each delta, find the increase in value.
                if it is less than 10% Stop, else Repeat
            '''

            if x != 0:
                delts.append(logs[x] - logs[x - 1])
                print((delts[len(delts) - 2] - delts[len(delts) - 1]) / delts[len(delts) - 2])
                if len(delts) > 1:
                    if delts[len(delts) - 2] != 0 and ((delts[len(delts) - 2] - delts[len(delts) - 1]) / delts[len(delts) - 2] < .1):    
                        break

        print('Final Mean Values for each cluster:')
        for i in range(len(stat.mean)):
            print('Cluster ' + str(i) + "  : " + str(stat.mean[i]))

        print('Close the Graph to continue.')

        plt.title('Likelihood Convergence: ' + numCols[int(colNo)])
        plt.plot(logs)
        plt.ylabel('Likelihood')
        plt.show()

        choice = input('Press Y to Restart and N to Exit: ')

        if str.lower(choice) == 'n':
            sys.exit()

except Exception as e:
    print("Error Occurred. Please Restart")
    print(e)