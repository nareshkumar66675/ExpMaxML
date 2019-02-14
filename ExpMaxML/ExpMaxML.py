import pandas as pd
import matplotlib.pyplot as plt
from StatFunctions.PDF import *
#from matplotlib import style
#style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
#np.random.seed(0)
#X = np.linspace(-5,5,num=20)
#X0 = X*np.random.rand(len(X))+10 # Create data cluster 1
#X1 = X*np.random.rand(len(X))-10 # Create data cluster 2
#X2 = X*np.random.rand(len(X)) # Create data cluster 3
liverFilePath = r'C:\Users\kumar\Downloads\indian-liver-patient-records\indian_liver_patient.csv'
liverDF = pd.read_csv(liverFilePath)

albumin=liverDF.iloc[:,8].values



#X_tot = np.stack((X0,X1,X2)).flatten() # Combine the clusters to get the random datapoints from above
"""Create the array r with dimensionality nxK"""
r = np.zeros((len(albumin),3))  
print('Dimensionality','=',np.shape(r))
"""Instantiate the random gaussians"""
gauss = [[0,1],[1,0.5],[1.5,0.2]]
"""Instantiate the random pi_c"""
pi = np.array([1/3,1/3,1/3]) # We expect to have three clusters 
"""
Probability for each datapoint x_i to belong to gaussian g 
"""
for c,g,p in zip(range(3),gauss,pi):
    r[:,c] = p* pdf(albumin,g[0],g[1]) # Write the probability that x belongs to gaussian c in column c. 
                          # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
"""
Normalize the probabilities such that each row of r sums to 1 and weight it by pi_c == the fraction of points belonging to 
cluster c
"""
for i in range(len(r)):
    r[i] = r[i]/(np.sum(pi)*np.sum(r,axis=1)[i])
"""In the last calculation we normalized the probabilites r_ic. So each row i in r gives us the probability for x_i 
to belong to one gaussian (one column per gaussian). Since we want to know the probability that x_i belongs 
to gaussian g, we have to do smth. like a simple calculation of percentage where we want to know how likely it is in % that
x_i belongs to gaussian g. To realize this we must dive the probability of each r_ic by the total probability r_i (this is done by 
summing up each row in r and divide each value r_ic by sum(np.sum(r,axis=1)[r_i] )). To get this,
look at the above plot and pick an arbitrary datapoint. Pick one gaussian and imagine the probability that this datapoint
belongs to this gaussian. This value will normally be small since the point is relatively far away right? So what is
the percentage that this point belongs to the chosen gaussian? --> Correct, the probability that this datapoint belongs to this 
gaussian divided by the sum of the probabilites for this datapoint and all three gaussians. Since we don't know how many
point belong to each cluster c and threwith to each gaussian c, we have to make assumptions and in this case simply said that we 
assume that the points are equally distributed over the three clusters."""
    
print(r)
print(np.sum(r,axis=1)) # As we can see, as result each row sums up to one, just as we want it.

"""M-Step"""
"""calculate m_c"""
m_c = []
for c in range(len(r[0])):
    m = np.sum(r[:,c])
    m_c.append(m) # For each cluster c, calculate the m_c and add it to the list m_c
    
"""calculate pi_c"""
pi_c = []
for m in m_c:
    pi_c.append(m/np.sum(m_c)) # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
"""calculate mu_c"""
mu_c = np.sum(albumin.reshape(len(albumin),1)*r,axis=0)/m_c
"""calculate var_c"""
var_c = []
for c in range(len(r[0])):
    var_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape(len(albumin),1))*(albumin.reshape(len(albumin),1)-mu_c[c])).T,(albumin.reshape(len(albumin),1)-mu_c[c])))
  
    
    
"""Update the gaussians"""
gauss = [[mu_c[0],var_c[0]],
		 [mu_c[1],var_c[1]],
		 [mu_c[2],var_c[2]]]

"""Plot the data"""
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for i in range(len(r)):
    ax0.scatter(albumin[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100) 
"""Plot the gaussians"""
for g,c in zip([pdf(np.sort(albumin).reshape(len(albumin),1),gauss[0][0],gauss[0][1]),pdf(np.sort(albumin).reshape(len(albumin),1),gauss[1][0],gauss[1][1]),
				pdf(np.sort(albumin).reshape(len(albumin),1),gauss[2][0],gauss[2][1])],['r','g','b']):
    ax0.plot(np.sort(albumin),g,c=c)
    
    
plt.show()