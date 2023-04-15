# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:52:37 2023

@author: kailas
"""
Q 20) Calculate probability from the given dataset for the below cases

Data _set: Cars.csv
Calculate the probability of MPG  of Cars for the below cases.
       MPG <- Cars$MPG
a.	P(MPG>38)
b.	P(MPG<40)
c.    P (20<MPG<50)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
cars=pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Cars (2).csv")
cars

sns.boxplot(cars.MPG)

# P(MPG>38)
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())
0.3475939251582705
# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())
0.7293498762151616
# P (20<MPG<50)
stats.norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std())          
1.2430968797327613e-05
 