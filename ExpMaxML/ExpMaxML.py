import pandas as pd

from StatFunctions.PDF import *
from StatFunctions.EM import *
import StatFunctions.StatObj as StatData

import numpy as np

liverFilePath = r'C:\Users\kumar\Downloads\indian-liver-patient-records\indian_liver_patient.csv'
liverDF = pd.read_csv(liverFilePath)

albumin=liverDF.iloc[:,8].values

gauss = None

stat = StatData.Stat([0,1,1.5],[1,0.5,0.2],[1/3,1/3,1/3])

for x in range(10):
    print('Iteration '+str(x))
    rValue = EStep(albumin,3,stat)
    stat = MStep(albumin,3,stat,rValue)
    print('Printing Mean')
    print(*stat.mean, sep = ", ")
    print('Printing Variance')
    print(*stat.variance, sep = ", ")
