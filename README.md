# ExpMaxML

It is an ML algorithm using **__Expectation Maximization using Gaussian Mixture Models__**.


# Overview

  - Given a Dataset, and selected attribute, algorithm clusters and gives the mean for each clusters.
  - It uses Gaussian Distribution for finding the probability.
  - It uses Expectation Maximization along with log likelihood to find a mean.

# Dataset Used
- Indian Liver Test : https://www.kaggle.com/uciml/indian-liver-patient-records
- Wine Quality : https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

# Algorithm

![FlowChart](https://raw.githubusercontent.com/nareshkumar66675/ExpMaxML/master/Others/ExpMaxMLFlow.png "FlowChart") 

*Created using Draw.io*

# Installation
```
1. Clone the Repository or Download the Project
2. Navigate to folder ExpMaxML
3. Execute 'python ExpMaxML.py'
```

# Sample Execution

#### 1. Select DataSet
```
Expectation Maximization using Gaussian Mixture Model
1.Liver
2.Wine
3.Custom(File Path Needed)
Select one Dataset from above : 1
```
#### 2. Select Column
```
Available Columns
0  Age
1  Total_Bilirubin
2  Direct_Bilirubin
3  Alkaline_Phosphotase
4  Alamine_Aminotransferase
5  Aspartate_Aminotransferase
6  Total_Protiens
7  Albumin
8  Albumin_and_Globulin_Ratio
9  Dataset
Select one Column(Enter number): 7
```
#### 3. Show Histogram and Find Optimal Cluster
```
Close the Graph to continue.
Checking feasibility of Cluster:2
Checking feasibility of Cluster:3
Optimal Cluster is 3
```
![AlbuminHistogram](https://raw.githubusercontent.com/nareshkumar66675/ExpMaxML/master/Others/AlbuminHist.png "AlbuminHistogram") 

#### 4. EM Iteration using GMM
```
*******  Iteration 0   *********
Printing Mean
2.355348573822168, 3.1193685474989397, 3.9317409062682755
Log Likelyhood :-151537.16537436945
*******  Iteration 1   *********
Printing Mean
2.356583558295428, 3.117896594822594, 3.928358017839534
Log Likelyhood :-151426.94519112387
0.0
*******  Iteration 2   *********
Printing Mean
2.356504406993411, 3.117180941618275, 3.9271702188672144
Log Likelyhood :-151399.42224758756
0.7502912558674304
..
...
.....
```

#### 5. Final Result and Likelihood Graph
```
Final Mean Values for each cluster:
Cluster 0  : 2.356364851826904
Cluster 1  : 3.116288502073406
Cluster 2  : 3.925730588575203
```
![Likelihood](https://raw.githubusercontent.com/nareshkumar66675/ExpMaxML/master/Others/Convergence.png "Likelihood") 

# Project Struture
##### ExpMaxML
- **ExpMaxML.py** - Main Startup File.
- **/StatFunctions**
    - EM - Methods related to Expectation Maximization
    - PDF - Distribution Implemenataion
    - StatObj - Custom Stat Class
##### Notebooks
- **Liver Analysis** - Very Basic Liver Analysis
##### DataSet
- indian_liver_patient.csv
- winequality-red.csv


  
